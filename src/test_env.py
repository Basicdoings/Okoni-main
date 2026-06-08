import modal
try:
    from src.common import app, image
except ImportError:
    from .common import app, image

@app.function(image=image)
def test_imports():
    print("Testing imports...")
    import torch
    import transformers
    import modelscope
    print(f"Torch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    try:
        from transformers import AutoConfig
        model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        print(f"Attempting to load config for {model_name}...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("Success! Model config loaded.")
    except Exception as e:
        print(f"Failed to load model config: {e}")
        print("This might be expected if the model requires authentication or is not yet public/compatible with this transformers version.")

@app.local_entrypoint()
def main():
    test_imports.remote()
