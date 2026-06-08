import base64
import sys
from pathlib import Path
import modal

# Import the app to use the Model class stub
try:
    from src.inference import Model
    from src.common import app
except ImportError:
    # Fallback if running as a module within src (though less likely with modal run file.py)
    from .inference import Model
    from .common import app

def main(
    ref_audio_path: str = None, 
    ref_text: str = None,
    text_to_speak: str = "Hello, this is a test of zero-shot voice cloning on Modal.",
    output_path: str = "output_cloned.wav"
):
    """
    Run Zero-Shot Voice Cloning using a local reference audio file.
    
    Args:
        ref_audio_path: Path to local wav file to clone voice from.
        ref_text: Transcript of what is said in the ref_audio.
        text_to_speak: Text you want the cloned voice to say.
        output_path: Where to save the result.
    """
    
    if not ref_audio_path:
        print("Please provide a reference audio path via --ref-audio-path")
        # Default for demonstration if file exists
        if Path("my_voice.wav").exists():
            ref_audio_path = "my_voice.wav"
        else:
            return

    # Check if we should treat this as a local file or a remote path
    is_local = True
    if not Path(ref_audio_path).exists():
        if ref_audio_path.startswith("/data") or ref_audio_path.startswith("/"):
            print(f"Path '{ref_audio_path}' not found locally, assuming it is a REMOTE path on the volume.")
            is_local = False
        else:
            print(f"File '{ref_audio_path}' not found locally!")
            return

    # Prepare input
    if is_local:
        print(f"Reading local file: {ref_audio_path}")
        with open(ref_audio_path, "rb") as f:
            audio_bytes = f.read()
            ref_audio_resource = base64.b64encode(audio_bytes).decode('utf-8')
            print(f"Encoded audio ({len(ref_audio_resource)} chars).")
    else:
        # Pass the path directly
        ref_audio_resource = ref_audio_path
        print(f"Using remote path: {ref_audio_resource}")
    
    print("Sending to Modal...")
    
    # Initialize the remote class
    model = Model()
    
    # Call remote function
    result_wav_bytes = model.generate.remote(
        text=text_to_speak,
        ref_audio_resource=ref_audio_resource,
        ref_text=ref_text
    )
    
    # Save output
    print(f"Received response ({len(result_wav_bytes)} bytes). Saving to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(result_wav_bytes)
    print("Done!")

if __name__ == "__main__":
    # We allow running this with `modal run src/test_zero_shot.py --ref-audio-path=...`
    # Modal's entrypoint wrapper handles the args if we define them in the function signature
    pass 

# To run with Modal, we need to wrap it in a local_entrypoint
@app.local_entrypoint()
def run_cli(ref_audio_path: str, ref_text: str = None, text: str = "Hello world"):
    main(ref_audio_path, ref_text, text)
