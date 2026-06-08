import os
import re
import subprocess
import sys
import threading
import time
import argparse
import shutil
from pathlib import Path

from okoni_config import save_config

ROOT = Path(__file__).resolve().parent
MODAL_URL_RE = re.compile(r"https://[^\s]+\.modal\.run")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def start_process(command, title):
    return subprocess.Popen(
        command,
        cwd=ROOT,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        env=os.environ.copy(),
    )


def start_modal(wait_for_url=False):
    log_path = ROOT / "modal_backend.log"
    process = subprocess.Popen(
        [sys.executable, "-m", "modal", "serve", "server.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    save_config(modal_pid=process.pid)
    ready = threading.Event()
    state = {"modal_url": ""}

    def read_modal_output():
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()

                match = MODAL_URL_RE.search(line)
                if match and not state["modal_url"]:
                    state["modal_url"] = match.group(0).rstrip("/")
                    save_config(modal_fish_url=state["modal_url"], modal_pid=process.pid)
                    print(f"Modal URL saved: {state['modal_url']}")
                    ready.set()

    threading.Thread(target=read_modal_output, daemon=True).start()

    if wait_for_url:
        ready.wait(timeout=180)
        if not state["modal_url"]:
            print("Modal URL was not detected yet. Check modal_backend.log.")

    return process


def main():
    parser = argparse.ArgumentParser(description="Start the Okoni local app.")
    parser.add_argument(
        "--with-modal",
        action="store_true",
        help="Also start the optional remote Modal/FishAudio backend.",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

    if shutil.which("sox") is None:
        winget_packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        sox_exe = next(winget_packages.glob("ChrisBagwell.SoX_*/*/sox.exe"), None)
        if sox_exe:
            os.environ["PATH"] = f"{sox_exe.parent}{os.pathsep}{os.environ.get('PATH', '')}"

    modal_process = None
    if args.with_modal:
        print("Starting optional Modal/FishAudio backend...")
        modal_process = start_modal(wait_for_url=False)
    else:
        print("Skipping optional Modal/FishAudio backend. Starting local Qwen only.")
        save_config(modal_pid=None)

    print("Starting local Okoni server...")
    local_server = start_process([sys.executable, "local_server.py"], "Okoni Server")
    time.sleep(5)

    print("Starting Okoni GUI...")
    gui = subprocess.Popen([sys.executable, "gui_client.py"], cwd=ROOT, env=os.environ.copy())
    gui.wait()

    print("GUI closed. Local server is still running in its own window.")
    if modal_process is not None and modal_process.poll() is None:
        print("Modal backend is still running. Use the GUI Stop Modal button or close its window.")


if __name__ == "__main__":
    main()
