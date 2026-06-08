import base64
import io
import modal
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

try:
    from src.common import app, image, vol_model, vol_data
except ImportError:
    try:
        from .common import app, image, vol_model, vol_data
    except ImportError:
        # Fallback for flattened structure or direct script execution
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from common import app, image, vol_model, vol_data

# Path to fine-tuned model (if it exists)
FINE_TUNED_PATH = "/model/finetuned"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

@app.cls(
    image=image,
    volumes={"/model": vol_model},
    gpu="T4", # Whisper runs fine on T4
    min_containers=0, # Scale to 0 when not used
)
class Transcriber:
    @modal.enter()
    def enter(self):
        import whisper
        import os
        print("Loading Whisper model...")
        # Cache whisper models in the volume
        os.makedirs("/model/whisper", exist_ok=True)
        self.model = whisper.load_model("base", download_root="/model/whisper") 
        print("Whisper loaded.")

    @modal.method()
    def transcribe(self, audio_data: bytes):
        import tempfile
        # Write bytes to temp file for whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name
        
        result = self.model.transcribe(tmp_path)
        return result["text"].strip()

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
        import os
        
        # Ensure cache dir exists
        cache_dir = "/model/qwen"
        os.makedirs(cache_dir, exist_ok=True)

        # Check if fine-tuned model exists
        model_path = BASE_MODEL
        if Path(FINE_TUNED_PATH).exists() and any(Path(FINE_TUNED_PATH).iterdir()):
            print(f"Found fine-tuned model at {FINE_TUNED_PATH}, loading...")
            model_path = FINE_TUNED_PATH
        else:
            print(f"Loading base model: {BASE_MODEL}")

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=cache_dir 
        )
        print("Model loaded.")

    def _lookup_ref_text(self, audio_path: str):
        # Try to find metadata.csv to look up text for the audio
        meta_path = Path("/data/out/dataset/metadata.csv")
        if not meta_path.exists():
            print(f"Metadata file not found at {meta_path}")
            return None
        
        # Extract filename id
        fname = Path(audio_path).stem
        print(f"Looking up metadata for id: {fname}")
        
        import csv
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for i, row in enumerate(reader):
                    # row: [id, text, norm_text]
                    if len(row) >= 2 and row[0] == fname:
                        print(f"Auto-found ref_text: {row[1][:50]}...")
                        return row[1]
                print(f"ID {fname} not found in metadata ({i+1} rows scanned).")
        except Exception as e:
            print(f"Error looking up metadata: {e}")
        return None

    @modal.method()
    def generate(self, text: str, ref_audio_resource: str = None, ref_text: str = None, 
                 language: str = "English", instruct: str = None,
                 bg_audio_resource: str = None, bg_volume: float = -20.0, bg_loop: bool = True):
        print(f"Generating for: '{text}' [Lang: {language}] [Instruct: {instruct}]")
        
        # If reference audio is provided, use Voice Clone mode (Zero-Shot)
        if ref_audio_resource:
            print(f"Using Voice Clone with provided resource (len={len(ref_audio_resource)})")
            
            # Auto-lookup ref_text if missing
            if not ref_text and len(ref_audio_resource) < 256 and Path(ref_audio_resource).exists():
                 found_text = self._lookup_ref_text(ref_audio_resource)
                 if found_text:
                     ref_text = found_text
            
            print(f"Ref Text: {ref_text}")
            
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_resource, 
                ref_text=ref_text,
            )
        else:
            print("Using Standard Generation (Preset/Fine-tuned default)")
            # Try to use instruct if provided, typically for CustomVoice or VoiceDesign
            kwargs = {}
            if instruct:
                kwargs["instruct"] = instruct
            
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker="Vivian",
                    **kwargs
                )
            except Exception as e:
                print(f"Error in default gen: {e}. Trying without speaker...")
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language,
                    **kwargs
                )

        # Convert output (numpy array) to bytes
        audio_data = wavs[0]
        
        # Log duration
        duration = len(audio_data) / sr
        print(f"Generated audio duration: {duration:.2f}s")
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        audio_bytes = buffer.getvalue()

        # --- Background Mixing ---
        if bg_audio_resource:
            try:
                from pydub import AudioSegment
                import base64
                
                print(f"Mixing background audio (loop={bg_loop}, volume={bg_volume}dB)")
                
                # Load TTS audio
                tts_audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                
                # Load BG audio
                if len(bg_audio_resource) > 512: # Likely base64
                    bg_bytes = base64.b64decode(bg_audio_resource)
                    bg_audio = AudioSegment.from_file(io.BytesIO(bg_bytes))
                elif Path(bg_audio_resource).exists():
                    bg_audio = AudioSegment.from_file(bg_audio_resource)
                else:
                    print(f"Error: Background audio resource not found: {bg_audio_resource}")
                    return audio_bytes

                # Apply volume
                bg_audio = bg_audio + bg_volume
                
                # Handle Looping
                if bg_loop:
                    # Loop bg to match or exceed tts length
                    loops = int(tts_audio.duration_seconds / bg_audio.duration_seconds) + 1
                    bg_audio = bg_audio * loops
                
                # Truncate bg to match tts length if it's too long
                bg_audio = bg_audio[:len(tts_audio)]
                
                # Overlay
                mixed_audio = bg_audio.overlay(tts_audio)
                
                # Export back to wav bytes
                out_buffer = io.BytesIO()
                mixed_audio.export(out_buffer, format="wav")
                print("Mixing complete.")
                return out_buffer.getvalue()
            except Exception as e:
                print(f"Error mixing background: {e}")
                return audio_bytes

        return audio_bytes

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def transcribe_endpoint(data: dict):
    # {"audio": "base64..."}
    b64_audio = data.get("audio")
    if not b64_audio:
        return {"error": "audio required"}
    
    import base64
    try:
        audio_bytes = base64.b64decode(b64_audio)
    except:
        return {"error": "Failed to decode base64"}
        
    transcriber = Transcriber()
    text = transcriber.transcribe.remote(audio_bytes)
    return {"text": text}

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def api(data: dict):
    # Expected JSON: {"text": "...", "ref_audio": "...", "ref_text": "...", "language": "en", 
    #                 "instruct": "...", "bg_audio": "...", "bg_volume": -20.0, "bg_loop": true}
    text = data.get("text")
    if not text: return {"error": "text required"}
    
    ref_audio = data.get("ref_audio") # Can be path or base64
    ref_text = data.get("ref_text")
    lang = data.get("language", "English")
    instruct = data.get("instruct")
    
    # Background options
    bg_audio = data.get("bg_audio")
    bg_volume = data.get("bg_volume", -20.0)
    bg_loop = data.get("bg_loop", True)
    
    model = Model()
    # Pass ref_audio as ref_audio_resource
    wav_bytes = model.generate.remote(text, ref_audio, ref_text, lang, instruct,
                                     bg_audio, bg_volume, bg_loop)
    
    from fastapi.responses import Response
    return Response(content=wav_bytes, media_type="audio/wav")

@app.local_entrypoint()
def test_inference():
    # Simple local test
    model = Model()
    # Use a dummy path or omit ref_audio to test default gen
    res = model.generate.remote("Hello world, this is a test.", language="English")
    with open("test_output.wav", "wb") as f:
        f.write(res)
    print("Saved test_output.wav")
