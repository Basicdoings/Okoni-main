import requests
import time

URL = "https://elavnlabs--qwen3-tts-finetune-api.modal.run"

def check():
    print(f"Checking {URL}...")
    start = time.time()
    try:
        # Minimal payload
        payload = {"text": "Hello", "language": "English"}
        resp = requests.post(URL, json=payload, timeout=120) # 2 min timeout
        print(f"Status Code: {resp.status_code}")
        print(f"Response Time: {time.time() - start:.2f}s")
        if resp.status_code == 200:
            print("Success! Audio bytes received.")
            print(f"Content length: {len(resp.content)}")
            # Write a small file to prove it
            with open("check_output.wav", "wb") as f:
                f.write(resp.content)
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    check()
