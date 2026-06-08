import os
# Optimize CUDA memory allocation to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch

# --- Monkeypatch for incompat. between Torch 2.1.2 and newer Transformers ---
import torch.utils._pytree
if not hasattr(torch.utils._pytree, 'register_pytree_node') and hasattr(torch.utils._pytree, '_register_pytree_node'):
    _orig_register = torch.utils._pytree._register_pytree_node
    def _wrapper(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
        return _orig_register(cls, flatten_fn, unflatten_fn)
    torch.utils._pytree.register_pytree_node = _wrapper
# --------------------------------------------------------------------------

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import base64
import io
import torch
from src.local_inference import LocalVoiceEngine
import soundfile as sf
import os

# Initialize App
app = FastAPI(title="Okoni Local Qwen3-TTS Server")

# Initialize the 0.6B model locally. It fits comfortably on an 8GB RTX 4060
# with reduced precision, while still falling back to CPU when CUDA is absent.
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
print(f"Initializing Local Engine with {MODEL_NAME}...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
engine = LocalVoiceEngine(model_name=MODEL_NAME, device=DEVICE)

@app.post("/transcribe")
async def transcribe_route(request: Request):
    data = await request.json()
    b64_audio = data.get("audio")
    if not b64_audio: return JSONResponse({"error": "audio required"}, status_code=400)
    
    try:
        audio_bytes = base64.b64decode(b64_audio)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        
        text = engine.transcribe(tmp_path)
        os.remove(tmp_path) # Cleanup
        return {"text": text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/design")
async def design_route(request: Request):
    """
    Note: The 0.6B Base model might not support "Voice Design" (instruct-based generation) 
    as well as the VoiceDesign specific model, but we can try to use the base model capabilities 
    or just inform the user.
    The local_inference.py doesn't currently expose a specific 'design' method distinct from cloning,
    but the Qwen3 codebase usually allows instruct.
    For now, we'll map this to the standard generation with instruct.
    """
    data = await request.json()
    text = data.get("text")
    instruct = data.get("instruct")
    lang = data.get("language", "English")
    
    if not text or not instruct:
        return JSONResponse({"error": "text and instruct required"}, status_code=400)
    
    try:
        # We use the same generation method but without ref audio, relying on instruct
        # NOTE: Qwen3-TTS Base models might need a speaker prompt if no ref audio is provided.
        # We will try to pass instruct.
        engine.load_tts()
        print(f"Designing voice: {text} | {instruct}")
        
        # NOTE: Accessing internal model directly because local_inference wrapper might be too simple
        # or we update local_inference to support design.
        # Let's try using the generate_cloned_speech but with no ref_audio, 
        # relying on the model's ability to handle just instruct if supported, 
        # or fall back to a default speaker.
        
        # Actually, let's look at local_inference.py again. It only has generate_cloned_speech.
        # We should probably update local_inference.py to support design or just use the model directly here.
        # For safety/simplicity, let's try to use the engine's model directly here since we have access.
        
        wavs, sr = engine.tts_model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct
        )
        
        audio_data = wavs[0]
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        return Response(content=buffer.getvalue(), media_type="audio/wav")
        
    except Exception as e:
        print(f"Design error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate")
async def generate_route(request: Request):
    data = await request.json()
    text = data.get("text")
    ref_audio_b64 = data.get("ref_audio")
    ref_text = data.get("ref_text")
    lang = data.get("language", "English")
    # instruct = data.get("instruct") # Not heavily used in clone mode usually, but can be passed
    
    if not text: return JSONResponse({"error": "text required"}, status_code=400)
    
    try:
        ref_audio_path = None
        temp_files = []
        
        if ref_audio_b64:
            # Handle base64 audio
            if len(ref_audio_b64) < 255 and os.path.exists(ref_audio_b64):
                ref_audio_path = ref_audio_b64
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(base64.b64decode(ref_audio_b64))
                    ref_audio_path = f.name
                    temp_files.append(ref_audio_path)
        
        # Generate
        wav_data, sr = engine.generate_cloned_speech(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=lang
        )
        
        # Cleanup
        for p in temp_files:
            if os.path.exists(p): os.remove(p)
            
        return Response(content=engine.save_to_bytes(wav_data, sr), media_type="audio/wav")
        
    except Exception as e:
        print(f"Generation error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate_fish")
async def generate_fish_route(request: Request):
    data = await request.json()
    text = data.get("text")
    ref_audio = data.get("ref_audio")
    ref_text = data.get("ref_text", "")
    modal_fish_url = data.get("modal_fish_url", "").strip().rstrip("/")

    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    if not modal_fish_url:
        return JSONResponse({"error": "modal_fish_url required"}, status_code=400)

    try:
        import requests as req
        url = modal_fish_url if modal_fish_url.endswith("/generate_fish") else f"{modal_fish_url}/generate_fish"
        resp = req.post(
            url,
            json={"text": text, "ref_audio": ref_audio, "ref_text": ref_text},
            timeout=1800
        )
        if resp.status_code != 200:
            return JSONResponse({"error": resp.text}, status_code=resp.status_code)
        return Response(content=resp.content, media_type="audio/wav")
    except Exception as e:
        print(f"Fish relay error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
