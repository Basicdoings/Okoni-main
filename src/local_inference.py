import os
# --- Monkeypatch for incompat. between Torch 2.1.2 and newer Transformers ---
import torch.utils._pytree
if not hasattr(torch.utils._pytree, 'register_pytree_node') and hasattr(torch.utils._pytree, '_register_pytree_node'):
    _orig_register = torch.utils._pytree._register_pytree_node
    def _wrapper(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
        return _orig_register(cls, flatten_fn, unflatten_fn)
    torch.utils._pytree.register_pytree_node = _wrapper
# --------------------------------------------------------------------------
import torch
import numpy as np
import soundfile as sf
import whisper
from pydub import AudioSegment
from pathlib import Path
from qwen_tts import Qwen3TTSModel
import io

class LocalVoiceEngine:
    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.tts_model = None
        self.whisper_model = None
        
        # 1. Determine Cache Directory
        project_root = Path(__file__).parents[1].absolute()
        self.local_model_cache = project_root / "model_cache"
        
        default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        self.cache_dir = os.getenv("HF_HOME", default_cache)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 2. Repair Global Cache from Local Backup if needed
        # This is the MOST ROBUST fix: we heal the corrupted snapshot using our good files.
        self._repair_global_cache()

    def _repair_global_cache(self):
        """Heals the corrupted HF snapshot by copying missing files from known-good sources."""
        print(f"Checking for global cache corruption...")
        
        bad_hash = "5d83992436eae1d760afd27aff78a71d676296fc"
        good_hash = "c27fe8aa05b732b1376d0f6a1e522fbccb84abbd"
        
        snap_base = Path(self.cache_dir) / "hub" / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" / "snapshots"
        local_base = self.local_model_cache / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" / "snapshots"
        
        target = snap_base / bad_hash
        if not target.exists():
            return
            
        print(f"Found corrupted snapshot: {bad_hash}. Healing...")
        
        # Sources in priority: 1. Good global snapshot, 2. Local backup
        sources = [snap_base / good_hash, local_base / good_hash]
        
        import shutil
        for source in sources:
            if not source.exists():
                continue
            
            print(f"  Using source: {source}")
            for root, dirs, files in os.walk(source):
                rel_path = os.path.relpath(root, source)
                dest_dir = target / rel_path
                os.makedirs(dest_dir, exist_ok=True)
                
                for f in files:
                    src_file = Path(root) / f
                    dest_file = dest_dir / f
                    if not dest_file.exists():
                        print(f"    Restoring: {os.path.join(rel_path, f)}")
                        shutil.copy2(src_file, dest_file)
        
        print("Cache healing process complete.")

    def load_whisper(self, model_type="base"):
        # Unload TTS model if present to free up VRAM
        if self.tts_model is not None:
            print("Unloading Qwen3-TTS to free VRAM...")
            del self.tts_model
            self.tts_model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if self.whisper_model is None:
            print(f"Loading Whisper ({model_type})...")
            self.whisper_model = whisper.load_model(model_type, device=self.device)

    def load_tts(self):
        # Unload Whisper model if present to free up VRAM
        if self.whisper_model is not None:
            print("Unloading Whisper to free VRAM...")
            del self.whisper_model
            self.whisper_model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if self.tts_model is None:
            print(f"Loading Qwen3-TTS ({self.model_name})...")
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

            # Now that we've repaired the cache, standard loading should work perfectly
            self.tts_model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                dtype=dtype,
                attn_implementation="sdpa",
                cache_dir=self.cache_dir
            )
            # Move the underlying model to device manually since wrapper doesn't have .to()
            self.tts_model.model.to(self.device)
            # Update wrapper's device property if needed (it reads from model.device)

    def transcribe(self, audio_path):
        self.load_whisper()
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()

    def generate_cloned_speech(self, text, ref_audio, ref_text=None, language="English"):
        self.load_tts()
        print(f"Generating cloned speech: '{text}'")
        
        # Qwen3TTSModel handles path or numpy
        wavs, sr = self.tts_model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text
        )
        
        # wavs[0] is the audio array
        return wavs[0], sr

    def mix_with_background(self, speech_arr, sr, bg_path, bg_volume=-20):
        """
        Mixes speech array with a background audio file.
        bg_volume: in dB (e.g. -20 for quiet background)
        """
        if not bg_path or not os.path.exists(bg_path):
            return speech_arr, sr
            
        print(f"Mixing with background: {bg_path}")
        
        # Convert speech array to pydub AudioSegment
        # speech_arr is usually float32 [-1, 1]
        speech_int16 = (speech_arr * 32767).astype(np.int16)
        speech_segment = AudioSegment(
            speech_int16.tobytes(), 
            frame_rate=sr,
            sample_width=2, 
            channels=1
        )
        
        # Load background
        bg_segment = AudioSegment.from_file(bg_path)
        bg_segment = bg_segment.set_frame_rate(sr).set_channels(1)
        bg_segment = bg_segment + bg_volume # Adjust volume
        
        # Loop background if shorter than speech, or crop if longer
        if len(bg_segment) < len(speech_segment):
            bg_segment = bg_segment * (len(speech_segment) // len(bg_segment) + 1)
        bg_segment = bg_segment[:len(speech_segment)]
        
        # Overlay
        mixed = speech_segment.overlay(bg_segment)
        
        # Convert back to numpy float32
        mixed_arr = np.array(mixed.get_array_of_samples()).astype(np.float32) / 32768.0
        return mixed_arr, sr

    def save_to_bytes(self, audio_arr, sr):
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, sr, format='WAV')
        return buffer.getvalue()
