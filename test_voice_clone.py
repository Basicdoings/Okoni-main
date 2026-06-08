import requests
import base64
import os

# Configuration
SERVER_URL = "http://127.0.0.1:8000"
REF_AUDIO_PATH = "assets/wavs/0027.wav"  # Updated to use the wavs folder you added
REF_TEXT = "that perspective this feels like you know like some kind of Cinderella prince"
TEXT_TO_GENERATE = "Hello, this is a test of the voice cloning system."

# If 0027.wav doesn't exist, try using the existing output_cloned.wav for testing
if not os.path.exists(REF_AUDIO_PATH):
    print(f"Note: {REF_AUDIO_PATH} not found, trying output_cloned.wav instead...")
    REF_AUDIO_PATH = "output_cloned.wav"
    REF_TEXT = "This is a test audio sample"  # Generic text for the fallback file

# Check if audio file exists
if not os.path.exists(REF_AUDIO_PATH):
    print(f"Error: Could not find {REF_AUDIO_PATH}")
    print("Please update REF_AUDIO_PATH in this script to point to your 027.wav file")
    exit(1)

# Read and encode the reference audio
print(f"Reading reference audio: {REF_AUDIO_PATH}")
with open(REF_AUDIO_PATH, "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode('utf-8')

# Prepare the request
payload = {
    "text": TEXT_TO_GENERATE,
    "ref_audio": ref_audio_b64,
    "ref_text": REF_TEXT,
    "language": "English"
}

print(f"Sending request to {SERVER_URL}/generate...")
print(f"Text to generate: {TEXT_TO_GENERATE}")

try:
    response = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=300)
    
    if response.status_code == 200:
        print("[SUCCESS] Audio generated.")
        
        # Save the output
        output_file = "output_test.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"[SUCCESS] Saved to: {output_file}")
        print("\nYou can now play this file to hear the cloned voice!")
    else:
        print(f"[ERROR] Status {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"[ERROR] Request failed: {e}")
