import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import modal

try:
    from src.common import app, image, vol_data, vol_model, vol_repo
except ImportError:
    try:
        from .common import app, image, vol_data, vol_model, vol_repo
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from common import app, image, vol_data, vol_model, vol_repo

# Mount paths
DATA_DIR = "/data"
MODEL_DIR = "/model"
REPO_DIR = "/repo"
OUTPUT_DIR = "/model/finetuned"

@app.function(
    image=image,
    volumes={
        DATA_DIR: vol_data,
        MODEL_DIR: vol_model,
        REPO_DIR: vol_repo,
    },
    timeout=86400, # 24 hours
    gpu="A100",    # Use a powerful GPU
)
def finetune(
    base_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", # Use Base for fine-tuning
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    do_train: bool = True,
):
    print(f"Setting up fine-tuning environment...")
    
    # 1. Setup Repo
    # We clone into a volume so we can inspect it or reuse it
    repo_path = Path(REPO_DIR) / "Qwen3-TTS"
    if not repo_path.exists():
        print(f"Cloning Qwen3-TTS repo to {repo_path}...")
        subprocess.run(["git", "clone", "https://github.com/QwenLM/Qwen3-TTS.git", str(repo_path)], check=True)
    else:
        print("Repo already exists. Pulling latest...")
        try:
            subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)
        except:
            pass

    # Add repo to path just in case
    sys.path.append(str(repo_path))
    
    # 2. Prepare Data
    # Convert /data/out/dataset/metadata.csv -> /data/train.jsonl
    csv_path = Path(DATA_DIR) / "out" / "dataset" / "metadata.csv"
    wavs_dir = Path(DATA_DIR) / "out" / "dataset" / "wavs"
    
    if not csv_path.exists():
        # Fallback for testing if user hasn't run prep yet
        print(f"Warning: {csv_path} not found. Ensure you ran 'modal_prep_tts_dataset.py'.")
        # return # Proceed for now to show repo setup
    else:
        jsonl_path = Path(DATA_DIR) / "train.jsonl"
        print(f"Converting {csv_path} to {jsonl_path}...")
        count = 0
        with open(csv_path, "r", encoding="utf-8") as f_in, \
             open(jsonl_path, "w", encoding="utf-8") as f_out:
            reader = csv.reader(f_in, delimiter="|")
            for row in reader:
                if len(row) < 3: continue
                clip_id, text, norm_text = row
                wav_path = wavs_dir / f"{clip_id}.wav"
                if wav_path.exists():
                    # Assuming format: {"key": path, "text": text} or similar.
                    # We'll use a generic format compatible with many scripts, 
                    # but we might need to adjust after inspecting repo's example.
                    entry = {"audio": str(wav_path), "text": norm_text}
                    f_out.write(json.dumps(entry) + "\n")
                    count += 1
        print(f"Prepared {count} items.")

    if not do_train:
        print("Skipping training (do_train=False).")
        return

    # 3. Inspect and Run Training
    # We look for a training script (e.g., finetune.py, train.py) in the repo
    possible_scripts = ["finetune.py", "train.py", "scripts/finetune.py"]
    script_to_run = None
    for s in possible_scripts:
        p = repo_path / s
        if p.exists():
            script_to_run = p
            break
            
    if script_to_run:
        print(f"Found training script: {script_to_run}")
        print("Attempting to run help to see arguments...")
        subprocess.run(["python", str(script_to_run), "--help"])
        
        # Placeholder for ACTUAL run command once we know args
        # print("Running training...")
        # cmd = [
        #     "python", str(script_to_run),
        #     "--model_name_or_path", base_model_name,
        #     "--data_path", str(jsonl_path),
        #     "--output_dir", OUTPUT_DIR,
        #     "--num_train_epochs", str(num_epochs),
        #     "--learning_rate", str(learning_rate)
        # ]
        # subprocess.run(cmd, check=True)
    else:
        print("Could not find a standard 'train.py' or 'finetune.py'. Listing repo contents:")
        for p in repo_path.glob("**/*.py"):
            print(f" - {p.relative_to(repo_path)}")
        print("Please check the 'Walkthrough' or repo README for the exact command.")

    vol_model.commit()

@app.local_entrypoint()
def main():
    finetune.remote()
