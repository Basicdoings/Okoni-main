import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("okoni_config.json")

DEFAULT_CONFIG = {
    "server_url": "https://bassirexy--qwen3-tts-finetune-web-dev.modal.run",
    "modal_fish_url": "",
    "modal_pid": None,
}


def load_config():
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in data.items() if k in config})
    return config


def save_config(**updates):
    config = load_config()
    config.update(updates)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config
