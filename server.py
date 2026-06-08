import base64
import io
import modal
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import os

# --- Configuration & App Definition ---
APP_NAME = "qwen3-tts-finetune"
app = modal.App(APP_NAME)

vol_data = modal.Volume.from_name("tts-data", create_if_missing=True)
vol_model = modal.Volume.from_name("model-cache", create_if_missing=True)

FINE_TUNED_PATH = "/model/finetuned"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_REF_WAV = "/data/out/dataset/wavs/0027.wav"

# --- Helper for Pre-downloading ---
def download_models():
    """Bake models into the image during build phase for instant startup."""
    import torch
    import whisper
    from qwen_tts import Qwen3TTSModel
    
    print("Pre-downloading Whisper...")
    whisper.load_model("base")
    
    print("Pre-downloading Qwen3 models...")
    Qwen3TTSModel.from_pretrained(CUSTOM_VOICE_MODEL, dtype=torch.bfloat16)
    Qwen3TTSModel.from_pretrained(VOICE_DESIGN_MODEL, dtype=torch.bfloat16)

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "htop", "wget", "sox")
    .pip_install("torch", "torchaudio")
    .pip_install(
        "transformers>=4.37.0",
        "accelerate",
        "modelscope",
        "soundfile",
        "librosa",
        "scipy",
        "numpy",
        "pandas",
        "hf-transfer",
        "tiktoken",
        "einops",
        "transformers_stream_generator",
        "six",
        "openai-whisper",
        "pydub",
        "fastapi",
        "python-multipart"
    )
    .pip_install("git+https://github.com/QwenLM/Qwen3-TTS.git") 
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_models)
)

# ---------------------------------------------------------------------------
# Fish Audio S2-Pro Image  (separate from Qwen3 image to avoid conflicts)
# Hosts: fishaudio/s2-pro open-weight model via the fish-speech HTTP server
# GPU:   A10G (~24 GB VRAM needed for 4B + 400M Dual-AR model)
# ---------------------------------------------------------------------------
fish_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsndfile1", "sox", "curl", "portaudio19-dev", "libsox-dev")
    # Clone fish-speech and install in editable mode
    .run_commands(
        "git clone https://github.com/fishaudio/fish-speech /opt/fish-speech",
        "pip install --upgrade pip uv",
        "cd /opt/fish-speech && uv sync --python 3.12 --extra cu128 --no-dev",
        "cd /opt/fish-speech && uv pip install huggingface_hub hf-transfer requests",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PATH": "/opt/fish-speech/.venv/bin:$PATH",
        "VIRTUAL_ENV": "/opt/fish-speech/.venv",
        "PYTHONPATH": "/opt/fish-speech",
    })
)

# --- Transcriber Service ---
@app.cls(
    image=image,
    volumes={"/model": vol_model},
    gpu="T4", 
    min_containers=0,
)
class Transcriber:
    @modal.enter()
    def enter(self):
        import whisper
        print("Loading Whisper model...")
        self.model = whisper.load_model("base") 
        print("Whisper loaded.")

    @modal.method()
    def transcribe(self, audio_data: bytes):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name
        
        result = self.model.transcribe(tmp_path)
        return result["text"].strip()

# --- Voice Design Service ---
@app.cls(
    image=image,
    gpu="A10G",
    min_containers=0,
)
class VoiceDesigner:
    @modal.enter()
    def enter(self):
        print("Initializing Voice Designer...")
        from qwen_tts import Qwen3TTSModel
        import torch
        self.model = Qwen3TTSModel.from_pretrained(
            VOICE_DESIGN_MODEL,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        print("Designer loaded.")

    @modal.method()
    def design(self, text: str, emotion_instruct: str, language: str = "English"):
        print(f"Designing voice for instruction: '{emotion_instruct}'")
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=emotion_instruct
        )
        
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        return buffer.getvalue(), sr

# --- TTS Model Service ---
@app.cls(
    image=image,
    volumes={"/model": vol_model, "/data": vol_data},
    gpu="A10G",
    min_containers=1,
)
class Model:
    @modal.enter()
    def enter(self):
        print("Initializing Qwen3-TTS...")
        from qwen_tts import Qwen3TTSModel
        import torch
        
        model_path = CUSTOM_VOICE_MODEL
        if Path(FINE_TUNED_PATH).exists() and any(Path(FINE_TUNED_PATH).iterdir()):
            print(f"Found fine-tuned model at {FINE_TUNED_PATH}, loading...")
            model_path = FINE_TUNED_PATH
        else:
            print(f"Loading CustomVoice model: {CUSTOM_VOICE_MODEL}")

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        print("Model loaded.")

    def _lookup_ref_text(self, audio_path: str):
        meta_path = Path("/data/out/dataset/metadata.csv")
        if not meta_path.exists():
            print(f"Metadata file not found at {meta_path}")
            return None
        
        fname = Path(audio_path).stem
        print(f"Looking up metadata for id: {fname}")
        
        import csv
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for i, row in enumerate(reader):
                    if len(row) >= 2 and row[0] == fname:
                        print(f"Auto-found ref_text: {row[1][:50]}...")
                        return row[1]
                print(f"ID {fname} not found in metadata ({i+1} rows scanned).")
        except Exception as e:
            print(f"Error looking up metadata: {e}")
        return None

    @modal.method()
    def generate(self, text: str, ref_audio_resource: str = None, ref_text: str = None, 
                 language: str = "English", instruct: str = None):
        print(f"Generating for: '{text}' [Lang: {language}] [Instruct: {instruct}]")
        
        if ref_audio_resource:
            print(f"Using Voice Clone with provided resource (len={len(ref_audio_resource)})")
            if not ref_text and len(ref_audio_resource) < 256 and Path(ref_audio_resource).exists():
                 found_text = self._lookup_ref_text(ref_audio_resource)
                 if found_text:
                     ref_text = found_text
        else:
            # Automatic default: use 0027.wav if it exists
            if Path(DEFAULT_REF_WAV).exists():
                print(f"No reference audio provided. Using default: {DEFAULT_REF_WAV}")
                ref_audio_resource = DEFAULT_REF_WAV
                ref_text = self._lookup_ref_text(ref_audio_resource)
        
        if ref_audio_resource:
            print(f"Generating voice clone with {ref_audio_resource}. Instruct: {instruct}")
            print(f"Ref Text: {ref_text}")
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_resource, 
                ref_text=ref_text,
                instruct=instruct
            )
        else:
            print("Using Standard Generation (No reference found)")
            kwargs = {}
            if instruct: kwargs["instruct"] = instruct
            
            try:
                wavs, sr = self.model.generate_custom_voice(text=text, language=language, speaker="Vivian", **kwargs)
            except Exception as e:
                print(f"Error in default gen: {e}")
                # Ensure speaker is always provided as it is a required positional argument
                wavs, sr = self.model.generate_custom_voice(text=text, language=language, speaker="Speaker 1", **kwargs)

        audio_data = wavs[0]
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        return buffer.getvalue()

@app.cls(
    image=fish_image,
    gpu="A10G",
    volumes={"/model": vol_model},
    min_containers=0,
    timeout=1800,
)
class FishTTS:
    @modal.enter()
    def enter(self):
        import subprocess
        import sys
        import time
        from pathlib import Path

        import requests
        from huggingface_hub import snapshot_download

        self.api_url = "http://127.0.0.1:8080/v1/tts"
        model_path = Path("/model/fish-s2-pro")
        expected_files = [
            model_path / "config.json",
            model_path / "model.pth",
            model_path / "codec.pth",
        ]

        if not all(path.exists() for path in expected_files):
            print("[Fish S2-Pro] Downloading fishaudio/s2-pro from HuggingFace...")
            snapshot_download(
                repo_id="fishaudio/s2-pro",
                local_dir=str(model_path),
                force_download=True,
                resume_download=False,
            )
            print("[Fish S2-Pro] Download complete.")
        else:
            print(f"[Fish S2-Pro] Using cached weights at {model_path}")

        print("[Fish S2-Pro] Starting local fish-speech API server...")
        self.process = subprocess.Popen(
            [
                "/opt/fish-speech/.venv/bin/python", "-m", "tools.api_server",
                "--listen", "0.0.0.0:8080",
                "--llama-checkpoint-path", str(model_path),
                "--decoder-checkpoint-path", str(model_path / "codec.pth"),
                "--decoder-config-name", "modded_dac_vq",
            ],
            cwd="/opt/fish-speech",
        )

        health_url = "http://127.0.0.1:8080/v1/health"
        last_error = None
        for _ in range(90):
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    print("[Fish S2-Pro] API server is ready.")
                    return
            except Exception as e:
                last_error = e
            time.sleep(2)

        raise RuntimeError(f"Fish S2-Pro API server did not become ready: {last_error}")

    @modal.method()
    def generate(self, text: str, ref_audio_b64: str = None, ref_text: str = None):
        import io
        import re

        import numpy as np
        import requests
        import soundfile as sf

        def split_text(value, max_chars=500):
            sentences = re.split(r"(?<=[.!?])\s+", value.strip())
            chunks = []
            current = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                while len(sentence) > max_chars:
                    split_at = sentence.rfind(" ", 0, max_chars + 1)
                    split_at = split_at if split_at > 0 else max_chars
                    part, sentence = sentence[:split_at].strip(), sentence[split_at:].strip()
                    if current:
                        chunks.append(current)
                        current = ""
                    if part:
                        chunks.append(part)

                candidate = f"{current} {sentence}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = sentence
                else:
                    current = candidate

            if current:
                chunks.append(current)
            return chunks

        text_chunks = split_text(text)
        print(f"[Fish S2-Pro] Generating {len(text_chunks)} text chunk(s).")
        audio_chunks = []
        sample_rate = None

        for index, chunk in enumerate(text_chunks, start=1):
            print(f"[Fish S2-Pro] Chunk {index}/{len(text_chunks)}: {len(chunk)} characters")
            payload = {
                "text": chunk,
                "format": "wav",
                "chunk_length": 300,
                "normalize": True,
                "max_new_tokens": 0,
                "references": [],
            }

            if ref_audio_b64 and ref_text:
                payload["references"].append({
                    "audio": ref_audio_b64,
                    "text": ref_text,
                })

            resp = requests.post(self.api_url, json=payload, timeout=900)
            resp.raise_for_status()
            audio, chunk_rate = sf.read(io.BytesIO(resp.content), dtype="float32")
            if sample_rate is None:
                sample_rate = chunk_rate
            elif chunk_rate != sample_rate:
                raise RuntimeError(
                    f"Fish returned inconsistent sample rates: {sample_rate} and {chunk_rate}"
                )
            audio_chunks.append(audio)

        pause_shape = (int(sample_rate * 0.2),) + audio_chunks[0].shape[1:]
        pause = np.zeros(pause_shape, dtype=np.float32)
        combined = []
        for index, audio in enumerate(audio_chunks):
            if index:
                combined.append(pause)
            combined.append(audio)

        output = io.BytesIO()
        sf.write(output, np.concatenate(combined), sample_rate, format="WAV")
        return output.getvalue()

@app.function(image=image, timeout=1800)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse

    web_app = FastAPI()

    @web_app.post("/transcribe")
    async def transcribe_route(request: Request):
        data = await request.json()
        b64_audio = data.get("audio")
        if not b64_audio: return JSONResponse({"error": "audio required"}, status_code=400)
        
        audio_bytes = base64.b64decode(b64_audio)
        from server import Transcriber
        transcriber = Transcriber()
        text = transcriber.transcribe.remote(audio_bytes)
        return {"text": text}

    @web_app.post("/design")
    async def design_route(request: Request):
        data = await request.json()
        text = data.get("text")
        instruct = data.get("instruct")
        lang = data.get("language", "English")
        if not text or not instruct:
            return JSONResponse({"error": "text and instruct required"}, status_code=400)
        
        from server import VoiceDesigner
        designer = VoiceDesigner()
        wav_bytes, sr = designer.design.remote(text, instruct, lang)
        return Response(content=wav_bytes, media_type="audio/wav")

    @web_app.post("/generate")
    async def generate_route(request: Request):
        data = await request.json()
        text = data.get("text")
        if not text: return JSONResponse({"error": "text required"}, status_code=400)
        
        ref_audio = data.get("ref_audio")
        ref_text = data.get("ref_text")
        lang = data.get("language", "English")
        instruct = data.get("instruct")
        
        from server import Model
        model = Model()
        wav_bytes = model.generate.remote(text, ref_audio, ref_text, lang, instruct)
        return Response(content=wav_bytes, media_type="audio/wav")

    @web_app.post("/generate_fish")
    async def generate_fish_route(request: Request):
        data = await request.json()
        text = data.get("text")
        ref_audio = data.get("ref_audio")
        ref_text = data.get("ref_text", "")
        if not text:
            return JSONResponse({"error": "text required"}, status_code=400)

        from server import FishTTS
        fish = FishTTS()
        try:
            wav_bytes = fish.generate.remote(text, ref_audio, ref_text)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return Response(content=wav_bytes, media_type="audio/wav")

    return web_app

# ---------------------------------------------------------------------------
# Fish Audio S2-Pro — Modal Web Server
# ---------------------------------------------------------------------------
# Legacy helper kept for manual local testing inside Modal containers.
# The GUI uses the ASGI /generate_fish route above.
#
# API contract (fish-speech standard):
#   POST /v1/tts
#   Content-Type: application/json
#   Body: {
#     "text": "[excited] Hello world!",
#     "references": [{"audio": "<base64_wav>", "text": "ref transcript"}],
#     "format": "wav"
#   }
#   Response: audio/wav bytes
# ---------------------------------------------------------------------------
def fish_tts_server():
    """
    Starts the fish-speech HTTP API server (S2-Pro) inside a Modal A10G container.
    Downloads fishaudio/s2-pro weights from HuggingFace on first run
    (cached in the 'model-cache' volume for subsequent cold starts).
    """
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download

    model_path = Path("/model/fish-s2-pro")

    # Download weights on first run; skip if already cached in the volume
    if not model_path.exists() or not any(model_path.iterdir()):
        print("[Fish S2-Pro] Downloading fishaudio/s2-pro from HuggingFace...")
        snapshot_download(
            repo_id="fishaudio/s2-pro",
            local_dir=str(model_path),
        )
        print("[Fish S2-Pro] Download complete.")
    else:
        print(f"[Fish S2-Pro] Using cached weights at {model_path}")

    print("[Fish S2-Pro] Starting API server on 0.0.0.0:8080 ...")
    subprocess.Popen(
        [
            "/opt/fish-speech/.venv/bin/python", "-m", "tools.api_server",
            "--listen", "0.0.0.0:8080",
            "--llama-checkpoint-path", str(model_path),
            "--decoder-checkpoint-path", str(model_path / "codec.pth"),
            "--decoder-config-name", "modded_dac_vq",
        ],
        cwd="/opt/fish-speech",
    )


@app.local_entrypoint()
def test_inference(text: str = "Hello world"):
    model = Model()
    res = model.generate.remote(text)
    with open("test_output.wav", "wb") as f:
        f.write(res)
    print("Saved test_output.wav")
