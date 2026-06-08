import customtkinter as ctk
import requests
import sounddevice as sd
import soundfile as sf
import base64
import io
import threading
import os
import subprocess
from tkinter import filedialog, messagebox

from okoni_config import load_config, save_config

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(0.85)


# ---------------------------------------------------------------------------
# Fix: sounddevice cffi callback AttributeError on '_CallbackContext.out'
# This fires AFTER sd.wait() returns, so try/except around sd.play() can't
# catch it. Patch the cleanup method once at import time instead.
# ---------------------------------------------------------------------------
def _patch_sounddevice():
    _orig = sd._CallbackContext.finished_callback

    def _safe_finished_callback(self):
        try:
            _orig(self)
        except AttributeError:
            pass

    sd._CallbackContext.finished_callback = _safe_finished_callback

_patch_sounddevice()
# ---------------------------------------------------------------------------


class TTSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Okoni TTS")
        self.geometry("720x620")
        self.minsize(660, 560)
        
        # --- State ---
        self.ref_audio_path = None
        self.ref_audio_b64 = None
        self.ref_text = ""
        self.designed_audio = None
        self.custom_bg_path = None
        self.generated_audio = None  # kept for Save button
        self.config = load_config()
        self.modal_url = self.config.get("server_url", "http://127.0.0.1:8000")
        self.modal_fish_url = self.config.get("modal_fish_url", "")
        self.backend_var = ctk.StringVar(value="Qwen3-TTS")
        
        # --- Preload Default Audio (0027.wav) ---
        self._startup_ref_text = ""
        project_root = os.path.dirname(os.path.abspath(__file__))
        startup_file = os.path.join(project_root, "assets", "wavs", "0027.wav")
        
        if os.path.exists(startup_file):
            print(f"[INIT] Preloading absolute path: {startup_file}")
            self.ref_audio_path = startup_file
            try:
                with open(startup_file, "rb") as f:
                    self.ref_audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                print(f"[INIT] Successfully encoded {len(self.ref_audio_b64)} chars of b64")
                
                csv_paths = [
                    os.path.join(project_root, "metadata.csv"),
                    os.path.join(project_root, "assets", "wavs", "metadata.csv")
                ]
                for csv_p in csv_paths:
                    if os.path.exists(csv_p):
                        print(f"[INIT] Reading transcript from {csv_p}")
                        with open(csv_p, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split("|")
                                if len(parts) >= 2 and parts[0] == "0027":
                                    self._startup_ref_text = parts[1]
                                    break
                    if self._startup_ref_text:
                        break
            except Exception as e:
                print(f"[INIT] Error preloading: {e}")
        else:
            print(f"[INIT] Default file not found at: {startup_file}")
        
        # --- Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        
        self._build_header()
        self._build_main_content()
        
    def _build_header(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(frame, text="Server URL:").grid(row=0, column=0, sticky="w", padx=(10, 5), pady=(8, 4))
        self.url_entry = ctk.CTkEntry(frame, width=180)
        self.url_entry.insert(0, self.modal_url)
        self.url_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=(8, 4))

        ctk.CTkLabel(frame, text="Modal Fish URL:").grid(row=0, column=2, sticky="w", padx=(10, 5), pady=(8, 4))
        self.fish_url_entry = ctk.CTkEntry(frame, width=210, placeholder_text="https://xxx.modal.run")
        self.fish_url_entry.insert(0, self.modal_fish_url)
        self.fish_url_entry.grid(row=0, column=3, sticky="ew", padx=5, pady=(8, 4))

        self.btn_stop_modal = ctk.CTkButton(
            frame,
            text="Stop Modal",
            command=self._stop_modal,
            width=80,
            fg_color="#8a1f1f",
            hover_color="#a32929"
        )
        self.btn_stop_modal.grid(row=0, column=4, sticky="ew", padx=(8, 10), pady=(8, 4))
        
        ctk.CTkLabel(frame, text="Output Device:").grid(row=1, column=0, sticky="w", padx=(10, 5), pady=(4, 8))
        
        self.devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        wasapi_index = next(
            (i for i, api in enumerate(hostapis) if api["name"] == "Windows WASAPI"),
            None
        )
        output_devices = [
            (i, d) for i, d in enumerate(self.devices)
            if d["max_output_channels"] > 0
            and (wasapi_index is None or d["hostapi"] == wasapi_index)
        ]
        self.device_names = [f"{i}: {d['name']}" for i, d in output_devices]
        self.device_var = ctk.StringVar(value=self.device_names[0] if self.device_names else "Default")
        
        self.device_menu = ctk.CTkOptionMenu(frame, values=self.device_names, variable=self.device_var)
        self.device_menu.grid(row=1, column=1, columnspan=4, sticky="ew", padx=5, pady=(4, 8))
        
    def _build_main_content(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        
        self.tab_setup = self.tabview.add("1. Voice Setup (Clone)")
        self.tab_design = self.tabview.add("2. Voice Design")
        self.tab_gen = self.tabview.add("3. Generate Speech")
        
        self._build_setup_tab()
        self._build_design_tab()
        self._build_gen_tab()

    def _stop_modal(self):
        config = load_config()
        pid = config.get("modal_pid")
        stopped = False
        messages = []

        try:
            if pid:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, text=True, timeout=15
                )
                stopped = result.returncode == 0
                if result.stdout or result.stderr:
                    messages.append((result.stdout or result.stderr).strip())

            fallback = subprocess.run(
                [
                    "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                    (
                        "$root = (Resolve-Path '.').Path; "
                        "$matches = Get-CimInstance Win32_Process | "
                        "Where-Object { $_.CommandLine -match 'modal\\s+serve' -and "
                        "$_.CommandLine -match 'server\\.py' -and "
                        "$_.CommandLine -like \"*$root*\" }; "
                        "foreach ($p in $matches) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }; "
                        "$matches | ForEach-Object { $_.ProcessId }"
                    )
                ],
                capture_output=True, text=True, timeout=20
            )

            stopped = stopped or bool(fallback.stdout.strip())
            if fallback.stdout.strip():
                messages.append(f"Stopped Modal process IDs: {fallback.stdout.strip()}")
            elif fallback.stderr.strip():
                messages.append(fallback.stderr.strip())

            save_config(modal_pid=None)
            if stopped:
                messagebox.showinfo("Modal", "Modal backend stopped.")
            else:
                messagebox.showwarning(
                    "Modal",
                    "No matching Modal backend process was found. Close the Modal serve window if it is still open."
                )
        except Exception as e:
            messagebox.showerror("Modal", f"Failed to stop Modal: {e}")
        
    def _build_setup_tab(self):
        t = self.tab_setup
        
        self.var_source = ctk.StringVar(value="Local")
        fr_src = ctk.CTkFrame(t)
        fr_src.pack(fill="x", padx=8, pady=4)
        
        ctk.CTkRadioButton(fr_src, text="Local File", variable=self.var_source, value="Local", command=self._toggle_source).pack(side="left", padx=10, pady=5)
        ctk.CTkRadioButton(fr_src, text="Remote Path (Modal Volume)", variable=self.var_source, value="Remote", command=self._toggle_source).pack(side="left", padx=10, pady=5)
        
        self.frame_local = ctk.CTkFrame(t)
        self.frame_local.pack(fill="x", pady=4)
        ctk.CTkLabel(self.frame_local, text="Local File:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
        self.btn_file = ctk.CTkButton(self.frame_local, text="Select WAV", command=self._select_file, width=80)
        self.btn_file.pack(side="left", padx=10, pady=5)
        self.lbl_file = ctk.CTkLabel(
            self.frame_local,
            text="0027.wav" if self.ref_audio_path else "No file selected"
        )
        self.lbl_file.pack(side="left", padx=10)
        
        self.frame_remote = ctk.CTkFrame(t)
        ctk.CTkLabel(self.frame_remote, text="Remote Path (e.g. /data/out/dataset/wavs/0027.wav):", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
        self.entry_remote = ctk.CTkEntry(self.frame_remote, placeholder_text="/data/...", width=340)
        self.entry_remote.pack(fill="x", padx=10, pady=5)

        self.btn_transcribe = ctk.CTkButton(t, text="Transcribe Input", command=self._transcribe, fg_color="green")
        self.btn_transcribe.pack(fill="x", padx=8, pady=8)
        
        self.txt_ref = ctk.CTkTextbox(t, height=70)
        self.txt_ref.pack(fill="x", padx=8, pady=4)
        
        if hasattr(self, '_startup_ref_text') and self._startup_ref_text:
            self.txt_ref.insert("1.0", self._startup_ref_text)
        else:
            self.txt_ref.insert("1.0", "Paste or type reference text here...")

    def _build_design_tab(self):
        t = self.tab_design
        
        ctk.CTkLabel(t, text="Persona Description:", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=5)
        self.txt_design_instruct = ctk.CTkTextbox(t, height=60)
        self.txt_design_instruct.pack(fill="x", padx=10, pady=5)
        self.txt_design_instruct.insert("1.0", "Young female, gasping for air, sounding terrified and panicked")
        
        ctk.CTkLabel(t, text="Reference Script (What the persona says):").pack(anchor="w", padx=10, pady=5)
        self.txt_design_script = ctk.CTkTextbox(t, height=60)
        self.txt_design_script.pack(fill="x", padx=10, pady=5)
        self.txt_design_script.insert("1.0", "I... I don't know what to do! It's right behind me!")
        
        self.btn_design = ctk.CTkButton(t, text="Design Emotional Voice", command=self._design_voice, height=36, fg_color="purple")
        self.btn_design.pack(fill="x", padx=10, pady=10)
        
        self.btn_transfer = ctk.CTkButton(t, text="Use This Persona for Cloning", command=self._transfer_to_clone, height=36, fg_color="blue", state="disabled")
        self.btn_transfer.pack(fill="x", padx=10, pady=10)

        self.btn_play_design = ctk.CTkButton(t, text="Play Designed Clip", command=lambda: self._play_audio(self.designed_audio) if self.designed_audio else None, height=32, fg_color="green", state="disabled")
        self.btn_play_design.pack(fill="x", padx=10, pady=5)
        
    def _build_gen_tab(self):
        t = self.tab_gen

        frame_engine = ctk.CTkFrame(t)
        frame_engine.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(frame_engine, text="Engine:", font=("Arial", 12, "bold")).pack(side="left", padx=10, pady=8)
        ctk.CTkRadioButton(
            frame_engine, text="Qwen3-TTS (Clone)",
            variable=self.backend_var, value="Qwen3-TTS",
            command=self._on_engine_changed
        ).pack(side="left", padx=10, pady=8)
        ctk.CTkRadioButton(
            frame_engine, text="Fish S2-Pro (Emotional)",
            variable=self.backend_var, value="Fish S2-Pro",
            command=self._on_engine_changed
        ).pack(side="left", padx=10, pady=8)
        
        ctk.CTkLabel(t, text="Text to Speak:", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=5)
        self.txt_input = ctk.CTkTextbox(t, height=70)
        self.txt_input.pack(fill="x", padx=8, pady=4)
        if hasattr(self, '_startup_ref_text') and self._startup_ref_text:
            self.txt_input.insert("1.0", self._startup_ref_text)
        else:
            self.txt_input.insert("1.0", "Hello, this is a test of the voice cloning system.")
        
        self.lbl_instruct = ctk.CTkLabel(t, text="Style / Instruction (Optional):")
        self.lbl_instruct.pack(anchor="w", padx=8, pady=4)
        self.entry_instruct = ctk.CTkEntry(t, placeholder_text="e.g. 'Laughing', 'Sad', 'Whispering'")
        self.entry_instruct.pack(fill="x", padx=8, pady=4)
        
        self.btn_gen = ctk.CTkButton(t, text="Generate Speech", command=self._generate, height=36, font=("Arial", 14))
        self.btn_gen.pack(fill="x", padx=8, pady=8)

        # Play + Save row
        btn_row = ctk.CTkFrame(t, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=4)
        btn_row.grid_columnconfigure(0, weight=3)
        btn_row.grid_columnconfigure(1, weight=1)

        self.btn_play = ctk.CTkButton(
            btn_row, text="▶  Play Output (to Device)",
            command=self._play_current, height=36,
            font=("Arial", 14), fg_color="green", state="disabled"
        )
        self.btn_play.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_save = ctk.CTkButton(
            btn_row, text="💾  Save WAV",
            command=self._save_audio, height=36,
            font=("Arial", 12), fg_color="#1a6b8a", hover_color="#1e87b0", state="disabled"
        )
        self.btn_save.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        
        self.lbl_status = ctk.CTkLabel(t, text="Ready", text_color="gray")
        self.lbl_status.pack(pady=5)
        
        # --- Background Noise Section ---
        frame_bg = ctk.CTkFrame(t)
        frame_bg.pack(fill="x", padx=8, pady=8)
        frame_bg.grid_columnconfigure(1, weight=1)
        frame_bg.grid_columnconfigure(3, weight=1)
        
        ctk.CTkLabel(frame_bg, text="Background Noise Overlay:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 4)
        )
        
        self.noise_files = ["None"]
        noise_dir = "assets/noises"
        if os.path.exists(noise_dir):
            self.noise_files += [f for f in os.listdir(noise_dir) if f.endswith((".wav", ".mp3"))]
        
        self.noise_var = ctk.StringVar(value="None")
        self.noise_menu = ctk.CTkOptionMenu(frame_bg, values=self.noise_files, variable=self.noise_var)
        self.noise_menu.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(10, 5), pady=(4, 4))
        
        self.btn_custom_bg = ctk.CTkButton(frame_bg, text="Custom BG", command=self._select_custom_bg, width=80)
        self.btn_custom_bg.grid(row=1, column=2, sticky="ew", padx=5, pady=(4, 4))
        
        self.loop_var = ctk.BooleanVar(value=True)
        self.chk_loop = ctk.CTkCheckBox(frame_bg, text="Loop", variable=self.loop_var)
        self.chk_loop.grid(row=1, column=3, sticky="w", padx=8, pady=(4, 4))
        
        ctk.CTkLabel(frame_bg, text="Volume:").grid(row=2, column=0, sticky="w", padx=10, pady=(4, 8))
        self.bg_vol_slider = ctk.CTkSlider(frame_bg, from_=-60, to=0, number_of_steps=60)
        self.bg_vol_slider.set(-20)
        self.bg_vol_slider.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(5, 10), pady=(4, 8))
        self._on_engine_changed()

    def _on_engine_changed(self):
        if not hasattr(self, "lbl_instruct"):
            return
        if self.backend_var.get() == "Fish S2-Pro":
            self.lbl_instruct.pack_forget()
            self.entry_instruct.pack_forget()
        else:
            self.lbl_instruct.pack(anchor="w", padx=10, pady=5, after=self.txt_input)
            self.entry_instruct.pack(fill="x", padx=10, pady=5, after=self.lbl_instruct)

    def _select_custom_bg(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if path:
            self.custom_bg_path = path
            fname = os.path.basename(path)
            if fname not in self.noise_files:
                self.noise_files.append(fname)
                self.noise_menu.configure(values=self.noise_files)
            self.noise_var.set(fname)

    def _toggle_source(self):
        if self.var_source.get() == "Local":
            self.frame_remote.pack_forget()
            self.frame_local.pack(fill="x", pady=5)
        else:
            self.frame_local.pack_forget()
            self.frame_remote.pack(fill="x", pady=5)

    def _select_file(self):
        initial_dir = "assets/wavs" if os.path.exists("assets/wavs") else "."
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Audio Files", "*.wav *.mp3")]
        )
        if path:
            self.ref_audio_path = path
            self.lbl_file.configure(text=path.split("/")[-1])
            with open(path, "rb") as f:
                self.ref_audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            try:
                filename = path.split("/")[-1]
                file_id = filename.split(".")[0]
                csv_paths = [
                    os.path.join(os.path.dirname(path), "metadata.csv"),
                    os.path.join(os.path.dirname(path), "..", "metadata.csv"),
                    "metadata.csv"
                ]
                found_text = None
                for csv_p in csv_paths:
                    if os.path.exists(csv_p):
                        with open(csv_p, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split("|")
                                if len(parts) >= 2 and parts[0] == file_id:
                                    found_text = parts[1]
                                    break
                    if found_text:
                        break
                if found_text:
                    print(f"Loaded transcript for {file_id}: {found_text[:30]}...")
                    self.txt_ref.delete("1.0", "end")
                    self.txt_ref.insert("1.0", found_text)
            except Exception as e:
                print(f"Error loading metadata: {e}")
    
    def _transcribe(self):
        self.lbl_status.configure(text="Transcribing...", text_color="orange")
        self.update()
        
        def run():
            try:
                base_url = self.url_entry.get().strip().rstrip("/")
                url = f"{base_url}/transcribe"
                if self.var_source.get() == "Local":
                    if not self.ref_audio_b64:
                        messagebox.showerror("Error", "Select a local file.")
                        self.lbl_status.configure(text="Error", text_color="red")
                        return
                    payload = {"audio": self.ref_audio_b64}
                else:
                    messagebox.showinfo("Info", "Remote transcription not yet supported directly. Please enter text manually or use Local file.")
                    self.lbl_status.configure(text="Remote Transcribe N/A", text_color="gray")
                    return

                resp = requests.post(url, json=payload, timeout=300)
                if resp.status_code == 200:
                    text = resp.json().get("text", "")
                    self.txt_ref.delete("1.0", "end")
                    self.txt_ref.insert("1.0", text)
                    self.lbl_status.configure(text="Transcription Complete", text_color="green")
                else:
                    self.lbl_status.configure(text=f"Error: {resp.text}", text_color="red")
            except Exception as e:
                self.lbl_status.configure(text=f"Error: {e}", text_color="red")
        
        threading.Thread(target=run).start()

    def _design_voice(self):
        script = self.txt_design_script.get("1.0", "end").strip()
        instruct = self.txt_design_instruct.get("1.0", "end").strip()
        if not script or not instruct:
            messagebox.showwarning("Warning", "Please provide both description and script.")
            return
            
        self.lbl_status.configure(text="Designing voice... (cold start ~30s)", text_color="purple")
        self.update()
        
        def run():
            try:
                base_url = self.url_entry.get().strip().rstrip("/")
                url = f"{base_url}/design"
                payload = {"text": script, "instruct": instruct, "language": "English"}
                resp = requests.post(url, json=payload, timeout=300)
                if resp.status_code == 200:
                    self.designed_audio = resp.content
                    self.lbl_status.configure(text="Design Complete! Play or Transfer now.", text_color="green")
                    self.btn_transfer.configure(state="normal")
                    self.btn_play_design.configure(state="normal")
                else:
                    self.lbl_status.configure(text=f"Design Error: {resp.text}", text_color="red")
            except Exception as e:
                self.lbl_status.configure(text=f"Error: {e}", text_color="red")
        
        threading.Thread(target=run).start()

    def _transfer_to_clone(self):
        if not self.designed_audio:
            return
        self.ref_audio_b64 = base64.b64encode(self.designed_audio).decode('utf-8')
        self.ref_text = self.txt_design_script.get("1.0", "end").strip()
        self.var_source.set("Local")
        self._toggle_source()
        self.lbl_file.configure(text="[Design Tab Persona]")
        self.txt_ref.delete("1.0", "end")
        self.txt_ref.insert("1.0", self.ref_text)
        messagebox.showinfo("Success", "Persona transferred to Clone! Go to tab 3 to generate.")
        self.tabview.set("3. Generate Speech")

    def _generate(self):
        text = self.txt_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to speak.")
            return
            
        self.lbl_status.configure(text="Generating audio... please wait (cold start ~60s)...", text_color="orange")
        self.update()
        
        def run():
            try:
                base_url = self.url_entry.get().strip().rstrip("/")
                is_fish = self.backend_var.get() == "Fish S2-Pro"
                url = f"{base_url}/generate_fish" if is_fish else f"{base_url}/generate"
                ref_text = self.txt_ref.get("1.0", "end").strip() or None

                if is_fish:
                    modal_fish_url = self.fish_url_entry.get().strip().rstrip("/")
                    if not modal_fish_url:
                        self.lbl_status.configure(text="Error: Modal Fish URL required", text_color="red")
                        return
                    save_config(modal_fish_url=modal_fish_url)
                    payload = {"text": text, "ref_text": ref_text, "modal_fish_url": modal_fish_url}
                else:
                    instruct = self.entry_instruct.get().strip()
                    payload = {
                        "text": text, "language": "English",
                        "instruct": instruct or None, "ref_text": ref_text
                    }
                
                source = self.var_source.get()
                print(f"[DEBUG] Source Mode: {source}")
                print(f"[DEBUG] self.ref_audio_path: {getattr(self, 'ref_audio_path', 'N/A')}")
                print(f"[DEBUG] self.ref_audio_b64 is set: {bool(getattr(self, 'ref_audio_b64', None))}")
                
                if source == "Local":
                    if hasattr(self, 'ref_audio_b64') and self.ref_audio_b64:
                        print(f"[DEBUG] Using cached ref_audio_b64 (len={len(self.ref_audio_b64)})")
                        payload["ref_audio"] = self.ref_audio_b64
                    elif hasattr(self, 'ref_audio_path') and self.ref_audio_path:
                        print(f"[DEBUG] b64 missing, checking path: {self.ref_audio_path}")
                        if os.path.exists(self.ref_audio_path):
                            try:
                                with open(self.ref_audio_path, "rb") as f:
                                    ref_audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                                payload["ref_audio"] = ref_audio_b64
                                print("[DEBUG] Re-read success")
                            except Exception as e:
                                print(f"[ERROR] Failed to read audio file: {e}")
                        else:
                            print(f"[ERROR] Path does not exist: {self.ref_audio_path}")
                    else:
                        print("[ERROR] No local audio source variables found!")
                        self.lbl_status.configure(text="Error: No Audio Selected", text_color="red")
                        return
                else:
                    rem_path = self.entry_remote.get().strip()
                    if is_fish:
                        self.lbl_status.configure(text="Fish S2-Pro requires local reference audio or no reference.", text_color="red")
                        return
                    if rem_path:
                        print(f"[DEBUG] Using Remote Path: {rem_path}")
                        payload["ref_audio"] = rem_path
                    else:
                        print("[ERROR] No remote path provided")
                
                print(f"Calling {url} with keys: {list(payload.keys())}...")
                resp = requests.post(url, json=payload, timeout=1800 if is_fish else 600)
                
                if resp.status_code == 200:
                    self.generated_audio = resp.content
                    self.lbl_status.configure(text="Generation Done! Local mixing will apply on Play.", text_color="green")
                    self.btn_play.configure(state="normal")
                    self.btn_save.configure(state="normal")
                else:
                    self.lbl_status.configure(text=f"Gen Error: {resp.text}", text_color="red")
            except requests.exceptions.Timeout:
                self.lbl_status.configure(text="Error: Request Timed Out", text_color="red")
            except Exception as e:
                print(f"Generate Exception: {e}")
                self.lbl_status.configure(text=f"Error: {e}", text_color="red")
        
        threading.Thread(target=run).start()
        
    def _play_current(self):
        if not self.generated_audio:
            return
        self.lbl_status.configure(text="Playing audio...", text_color="blue")
        self.update()
        self._play_audio(self.generated_audio)
        self.lbl_status.configure(text="Playback Finished", text_color="green")

    def _save_audio(self):
        """Save the last generated (raw, pre-mix) audio to a WAV file."""
        if not self.generated_audio:
            messagebox.showwarning("Nothing to Save", "Generate audio first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile="output.wav",
            title="Save Generated Audio"
        )
        if not path:
            return  # user cancelled

        try:
            # generated_audio is raw bytes from the server (WAV-encoded)
            with open(path, "wb") as f:
                f.write(self.generated_audio)
            self.lbl_status.configure(text=f"Saved: {os.path.basename(path)}", text_color="green")
            print(f"[SAVE] Audio saved to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file:\n{e}")

    def _play_audio(self, audio_bytes):
        try:
            import numpy as np
            from scipy.signal import resample

            self.lbl_status.configure(text="Mixing and preparing audio...", text_color="blue")
            self.update()

            speech_data, speech_fs = sf.read(io.BytesIO(audio_bytes))
            if len(speech_data.shape) > 1:
                speech_data = speech_data.mean(axis=1)
            
            noise_choice = self.noise_var.get()
            final_data = speech_data
            final_fs = speech_fs

            if noise_choice != "None":
                print(f"Local Mixing: {noise_choice}")
                if self.custom_bg_path and noise_choice == os.path.basename(self.custom_bg_path):
                    bg_path = self.custom_bg_path
                else:
                    bg_path = os.path.join("assets", "noises", noise_choice)
                
                if os.path.exists(bg_path):
                    try:
                        bg_data, bg_fs = sf.read(bg_path)
                        if len(bg_data.shape) > 1:
                            bg_data = bg_data.mean(axis=1)
                        
                        print(f"DEBUG: Loaded noise '{noise_choice}'. Peak Amp: {np.max(np.abs(bg_data)):.4f}, SR: {bg_fs}")
                        
                        if bg_fs != speech_fs:
                            print(f"Resampling noise from {bg_fs} to {speech_fs}")
                            num_samples = int(len(bg_data) * speech_fs / bg_fs)
                            bg_data = resample(bg_data, num_samples)
                        
                        db_adj = self.bg_vol_slider.get()
                        gain = 10 ** (db_adj / 20)
                        bg_data = bg_data * gain
                        
                        speech_len = len(speech_data)
                        if self.loop_var.get():
                            reps = int(np.ceil(speech_len / len(bg_data)))
                            bg_data = np.tile(bg_data, reps)[:speech_len]
                        else:
                            if len(bg_data) > speech_len:
                                bg_data = bg_data[:speech_len]
                            else:
                                bg_data = np.pad(bg_data, (0, speech_len - len(bg_data)))
                        
                        final_data = speech_data + bg_data
                        max_val = np.max(np.abs(final_data))
                        if max_val > 1.0:
                            final_data = final_data / max_val
                            
                        print("Local mixing complete.")
                    except Exception as e:
                        print(f"Mixing error (is it a valid audio file?): {e}")
                        if bg_path.lower().endswith(".mp3"):
                            print("TIP: If MP3 fails, try converting it to .wav")

            self._current_data = final_data.astype(np.float32)
            self._current_fs = final_fs
            
            dev_str = self.device_var.get()
            dev_idx = int(dev_str.split(":")[0]) if ":" in dev_str else None
            
            try:
                device_info = sd.query_devices(dev_idx, 'output')
                target_fs = int(device_info['default_samplerate'])
            except Exception:
                target_fs = self._current_fs

            print(f"Playing to device {dev_idx}. Source SR: {self._current_fs}, Target SR: {target_fs}")
            
            if target_fs != self._current_fs:
                print(f"Final resample from {self._current_fs} to {target_fs}...")
                num_samples = int(len(self._current_data) * target_fs / self._current_fs)
                self._current_data = resample(self._current_data, num_samples).astype(np.float32)
                self._current_fs = target_fs

            sd.play(self._current_data, self._current_fs, device=dev_idx)
            sd.wait()
            
            print("Playback finished.")
            
        except Exception as e:
            print(f"Playback error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Playback Error", f"{e}")


if __name__ == "__main__":
    app = TTSApp()
    app.mainloop()
