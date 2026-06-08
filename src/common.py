import modal

APP_NAME = "qwen3-tts-finetune"

# Volumes
vol_data = modal.Volume.from_name("tts-data", create_if_missing=True)
vol_model = modal.Volume.from_name("model-cache", create_if_missing=True)
vol_repo = modal.Volume.from_name("repo-cache", create_if_missing=True) # To store the cloned repo

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "htop", "wget", "sox")
    # .pip_install("packaging", "ninja", "wheel", "setuptools") # Not strictly needed if not building flash-attn
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
        "pydub"
        # "flash-attn" # Removed to avoid long build times; use sdpa instead
    )
    # Install Qwen3-TTS from official repo (assuming standard structure)
    # If this fails, we might need to clone and sys.path.append in the function.
    # But let's try direct install first as it's cleaner.
    # Note: The import name is qwen_tts
    .pip_install("git+https://github.com/QwenLM/Qwen3-TTS.git") 
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

fish_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "sox", "libsndfile1", "curl", "portaudio19-dev", "libsox-dev")
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

app = modal.App(APP_NAME)
