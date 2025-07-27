import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import cv2
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np

from time_trial_tracker import config, camera, stats, utils

class LapTimerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Time Trial Tracker")
        self.root.geometry("800x830")
        self.root.configure(bg='#1e1e1e')

        self.setup_styles()

        # State
        self.selected_camera = 1
        self.cap = None
        self.camera_thread = None
        self.running = False
        self.current_frame = None
        self.current_roi = None
        self.current_thresh = None
        self.track_name = ""
        self.lap_times = []
        self.attempt_id = 1
        self.race_active = False
        self.race_ready = False
        self.last_time_seen = ""
        self.stable_frame_count = 0
        self.last_time_change = time.time()
        self.last_lap_logged_time = 0
        self.flash_counter = {}
        self.prev_flash = False
        self.current_time_text = ""
        self.flash_detected = False
        self.debug_panel_visible = True
        self.global_stats = config.load_global_stats()
        self.crop_config = config.load_crop_config()

        # Thresholds
        self.STABLE_FRAME_THRESHOLD = 3
        self.FINAL_STABLE_SECONDS = 2.0
        self.LAP_FLASH_COUNT_REQUIRED = 3
        self.LAP_COOLDOWN_SECONDS = 5

        # GUI
        self.create_widgets()

        # Start with status
        self.status_label.config(text="Ready to start")

    def setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabelFrame', background='#1e1e1e', foreground='#ffffff')
        style.configure('TLabelFrame.Label', background='#1e1e1e', foreground='#ffffff')

        style.configure('Title.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Arial', 24, 'bold'))
        style.configure('Heading.TLabel', background='#1e1e1e', foreground='#4CAF50', font=('Arial', 14, 'bold'))
        style.configure('Data.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Arial', 14))
        style.configure('Time.TLabel', background='#1e1e1e', foreground='#FFD700', font=('Arial', 20, 'bold'))
        style.configure('Best.TLabel', background='#1e1e1e', foreground='#4CAF50', font=('Arial', 18, 'bold'))
        style.configure('Current.TLabel', background='#1e1e1e', foreground='#2196F3', font=('Arial', 18, 'bold'))
        style.configure('LapHeading.TLabel', background='#1e1e1e', foreground='#4CAF50', font=('Arial', 16, 'bold'))

        style.configure('Modern.TButton', background='#4CAF50', foreground='white', borderwidth=0, focuscolor='none')
        style.map('Modern.TButton', background=[('active', '#45a049')])

        style.configure('DebugSmall.TLabel', background='#1e1e1e', foreground='#ffffff', font=('Arial', 10))

        style.configure('TCombobox',
                        fieldbackground='#2d2d2d',
                        background='#2d2d2d',
                        foreground='white',
                        arrowcolor='white',
                        borderwidth=1,
                        insertcolor='white')

        style.map('TCombobox',
                fieldbackground=[('readonly', '#2d2d2d')],
                background=[('readonly', '#2d2d2d')],
                foreground=[('readonly', 'white')])

        style.configure('Large.TCombobox',
                        fieldbackground='#2d2d2d',
                        background='#2d2d2d',
                        foreground='white',
                        arrowcolor='white',
                        borderwidth=1,
                        insertcolor='white',
                        font=('Arial', 14))

        style.map('Large.TCombobox',
                fieldbackground=[('readonly', '#2d2d2d')],
                background=[('readonly', '#2d2d2d')],
                foreground=[('readonly', 'white')])

        style.configure('Dark.TLabelframe',
                        background='#1e1e1e',
                        foreground='#ffffff',
                        borderwidth=2,
                        relief='groove')

        style.configure('Dark.TLabelframe.Label',
                        background='#1e1e1e',
                        foreground='#ffffff',
                        font=('Arial', 12, 'bold'))

        self.root.option_add('*TCombobox*Listbox.font', 'Arial 10')
        self.root.option_add('*TCombobox*Listbox.background', '#2d2d2d')
        self.root.option_add('*TCombobox*Listbox.foreground', 'white')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#4CAF50')
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        self.root.option_add('*TCombobox*Listbox.borderWidth', '0')

    def create_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.main_frame = main_frame  # Store for later use

        self.create_title_section()
        self.create_lap_display_section()
        self.create_debug_panel()
        self.create_camera_panel()

    def create_title_section(self) -> None:
        title_label = ttk.Label(self.main_frame, text="Time Trial Tracker", style='Title.TLabel')
        title_label.pack(pady=(0, 10))

        track_select_frame = ttk.Frame(self.main_frame)
        track_select_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(track_select_frame, text="Track:", style='Data.TLabel').pack(side='left', padx=(0, 10))

        self.track_var = tk.StringVar()
        self.track_combo = ttk.Combobox(track_select_frame, textvariable=self.track_var, width=30, state='readonly', style='Large.TCombobox')
        self.track_combo['values'] = [
            "Mario Bros. Circuit", "Crown City", "Whistlestop Summit", "DK Spaceport",
            "Desert Hills", "Shy Guy Bazaar", "Wario Stadium", "Airship Fortress",
            "DK Pass", "Starview Peak", "Sky-High Sundae", "Wario Shipyard",
            "Koopa Troopa Beach", "Faraway Oasis", "Peach Stadium", "Peach Beach",
            "Salty Salty Speedway", "Dino Dino Jungle", "Great ? Block Ruins",
            "Cheep Cheep Falls", "Dandelion Depths", "Boo Cinema", "Dry Bones Burnout",
            "Moo Moo Meadows", "Choco Mountain", "Toad's Factory", "Bowser's Castle",
            "Acorn Heights", "Mario Circuit", "Rainbow Road"
        ]
        self.track_combo.pack(side='left')
        self.track_combo.bind('<<ComboboxSelected>>', self.on_track_selected)

    def create_lap_display_section(self) -> None:
        race_frame = ttk.LabelFrame(self.main_frame, padding=15, style='Dark.TLabelframe')
        race_frame.pack(fill='both', expand=True, pady=(0, 10))

        # Current time display
        time_frame = ttk.Frame(race_frame)
        time_frame.pack(fill='x', pady=(0, 20))
        ttk.Label(time_frame, text="Current Time:", style='LapHeading.TLabel').pack(side='left', padx=(0, 10))
        self.current_time_label = ttk.Label(time_frame, text="--:--.---", style='Time.TLabel')
        self.current_time_label.pack(side='left')

        # Lap time grid
        lap_times_frame = ttk.Frame(race_frame)
        lap_times_frame.pack(fill='x', pady=(0, 20))

        lap_times_frame.grid_columnconfigure(1, weight=1)  # Current
        lap_times_frame.grid_columnconfigure(2, weight=1)  # PR
        lap_times_frame.grid_columnconfigure(3, weight=1)  # Best Run

        self.lap_labels = {}
        self.delta_labels = {}
        self.pr_labels = {}
        self.best_run_labels = {}

        headers = ["Current Lap Times", "Personal Records", "Best Overall Run"]
        for i, header in enumerate(headers, start=1):
            header_frame = ttk.Frame(lap_times_frame)
            header_frame.grid(row=0, column=i, sticky='ew', padx=1, pady=(0, 10))
            ttk.Label(header_frame, text=header, style='LapHeading.TLabel').pack()

        for lap_index in range(1, 4):
            row = lap_index
            lap_label = f"Lap {lap_index}"
            ttk.Label(lap_times_frame, text=f"{lap_label}:", style='Data.TLabel', width=8).grid(row=row, column=0, sticky='w', pady=2)

            # Current
            current_col = ttk.Frame(lap_times_frame)
            current_col.grid(row=row, column=1, sticky='ew', padx=1, pady=2)
            current_content = ttk.Frame(current_col)
            current_content.pack()
            self.lap_labels[lap_label] = ttk.Label(current_content, text="--:--.---", style='Current.TLabel')
            self.lap_labels[lap_label].pack(side='left')
            self.delta_labels[lap_label] = ttk.Label(current_content, text="", style='Data.TLabel')
            self.delta_labels[lap_label].pack(side='left', padx=(5, 0))

            # PR
            pr_col = ttk.Frame(lap_times_frame)
            pr_col.grid(row=row, column=2, sticky='ew', padx=1, pady=2)
            self.pr_labels[lap_label] = ttk.Label(pr_col, text="--:--.---", style='Best.TLabel')
            self.pr_labels[lap_label].pack()

            # Best run
            best_run_col = ttk.Frame(lap_times_frame)
            best_run_col.grid(row=row, column=3, sticky='ew', padx=1, pady=2)
            self.best_run_labels[lap_label] = ttk.Label(best_run_col, text="--:--.---", style='Current.TLabel')
            self.best_run_labels[lap_label].pack()

        # Final time
        ttk.Label(lap_times_frame, text="Final:", style='Data.TLabel', width=8).grid(row=4, column=0, sticky='w', pady=2)

        # Current Final
        final_col = ttk.Frame(lap_times_frame)
        final_col.grid(row=4, column=1, sticky='ew', padx=1, pady=2)
        final_content = ttk.Frame(final_col)
        final_content.pack()
        self.lap_labels["Final"] = ttk.Label(final_content, text="--:--.---", style='Current.TLabel')
        self.lap_labels["Final"].pack(side='left')
        self.delta_labels["Final"] = ttk.Label(final_content, text="", style='Data.TLabel')
        self.delta_labels["Final"].pack(side='left', padx=(5, 0))

        # PR Final
        pr_final_col = ttk.Frame(lap_times_frame)
        pr_final_col.grid(row=4, column=2, sticky='ew', padx=1, pady=2)
        self.pr_labels["Final"] = ttk.Label(pr_final_col, text="--:--.---", style='Best.TLabel')
        self.pr_labels["Final"].pack()

        # Best Final
        best_final_col = ttk.Frame(lap_times_frame)
        best_final_col.grid(row=4, column=3, sticky='ew', padx=1, pady=2)
        self.best_run_labels["Final"] = ttk.Label(best_final_col, text="--:--.---", style='Current.TLabel')
        self.best_run_labels["Final"].pack()

    def create_debug_panel(self) -> None:
        self.toggle_debug_button = ttk.Button(
            self.main_frame,
            text="Hide Debug/Controls ▲",
            command=self.toggle_debug_panel,
            style='Modern.TButton'
        )
        self.toggle_debug_button.pack(pady=(10, 5))

        self.debug_panel = ttk.LabelFrame(
            self.main_frame,
            text="Debug & Controls",
            padding=10,
            style='Dark.TLabelframe'
        )
        self.debug_panel.pack(fill='x', expand=False, pady=(5, 0))

        top_row = ttk.Frame(self.debug_panel)
        top_row.pack(fill='x', pady=(0, 15))

        # --- Left Side Controls ---
        controls_left = ttk.Frame(top_row)
        controls_left.pack(side='left', fill='y', padx=(0, 20))

        # Buttons
        button_frame = ttk.Frame(controls_left)
        button_frame.pack(fill='x', pady=(0, 10))

        self.start_button = ttk.Button(
            button_frame,
            text="Start Camera",
            command=self.toggle_camera,
            style='Modern.TButton'
        )
        self.start_button.pack(side='left', padx=(0, 5))

        self.crop_button = ttk.Button(
            button_frame,
            text="Set Crop Region",
            command=self.set_crop_region,
            style='Modern.TButton'
        )
        self.crop_button.pack(side='left', padx=(0, 5))

        # Camera dropdown
        camera_frame = ttk.Frame(controls_left)
        camera_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(camera_frame, text="Camera:", style='Heading.TLabel').pack(side='left', padx=(0, 10))
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, width=15, state='readonly')
        self.camera_combo['values'] = ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
        self.camera_combo.set("Camera 1")
        self.camera_combo.pack(side='left')

        # Status + Attempt
        status_frame = ttk.Frame(controls_left)
        status_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(status_frame, text="Status:", style='Heading.TLabel').pack(anchor='w')
        self.status_label = ttk.Label(status_frame, text="Ready", style='Data.TLabel')
        self.status_label.pack(anchor='w')

        ttk.Label(status_frame, text="Current Attempt:", style='Heading.TLabel').pack(anchor='w', pady=(10, 0))
        self.attempt_label = ttk.Label(status_frame, text="No active attempt", style='Data.TLabel')
        self.attempt_label.pack(anchor='w')

        # Debug labels
        ttk.Label(status_frame, text="Debug Info:", style='Heading.TLabel').pack(anchor='w', pady=(10, 0))
        self.ocr_label = ttk.Label(status_frame, text="OCR: --", style='DebugSmall.TLabel')
        self.ocr_label.pack(anchor='w')
        self.stable_label = ttk.Label(status_frame, text="Stable: 0", style='DebugSmall.TLabel')
        self.stable_label.pack(anchor='w')
        self.flash_label = ttk.Label(status_frame, text="Flash: No", style='DebugSmall.TLabel')
        self.flash_label.pack(anchor='w')
        self.color_label = ttk.Label(status_frame, text="Text RGB: --", style='DebugSmall.TLabel')
        self.color_label.pack(anchor='w')
        self.cyan_label = ttk.Label(status_frame, text="Cyan: --", style='DebugSmall.TLabel')
        self.cyan_label.pack(anchor='w')

        # --- Right Side Canvases ---
        camera_view_frame = ttk.Frame(top_row)
        camera_view_frame.pack(side='right', fill='both', expand=True)

        ttk.Label(camera_view_frame, text="Timer Region:", style='Heading.TLabel').pack(anchor='w')
        self.camera_canvas = tk.Canvas(camera_view_frame, width=400, height=100, bg='black')
        self.camera_canvas.pack(anchor='w', pady=(5, 10))

        ttk.Label(camera_view_frame, text="Processed Image:", style='Heading.TLabel').pack(anchor='w')
        self.threshold_canvas = tk.Canvas(camera_view_frame, width=400, height=100, bg='black')
        self.threshold_canvas.pack(anchor='w', pady=(5, 0))
    
    def toggle_debug_panel(self) -> None:
        if self.debug_panel_visible:
            self.debug_panel.pack_forget()
            self.toggle_debug_button.config(text="Show Debug/Controls ▼")
            self.root.geometry("800x480")
        else:
            self.debug_panel.pack(fill='x', expand=False, pady=(5, 0))
            self.toggle_debug_button.config(text="Hide Debug/Controls ▲")
            self.root.geometry("800x830")

        self.debug_panel_visible = not self.debug_panel_visible
        self.root.update_idletasks()

    def create_camera_panel(self) -> None:
        camera_frame = ttk.Frame(self.main_frame)
        camera_frame.pack(fill='both', expand=True)

        ttk.Label(camera_frame, text="Timer Region:", style='Heading.TLabel').pack(anchor='w')
        self.camera_canvas = tk.Canvas(camera_frame, width=400, height=100, bg='black')
        self.camera_canvas.pack(anchor='w', pady=(5, 10))

        ttk.Label(camera_frame, text="Processed Image:", style='Heading.TLabel').pack(anchor='w')
        self.threshold_canvas = tk.Canvas(camera_frame, width=400, height=100, bg='black')
        self.threshold_canvas.pack(anchor='w', pady=(5, 0))

    def toggle_camera(self) -> None:
        if not self.running:
            if not self.track_var.get():
                messagebox.showwarning("Warning", "Please select a track first!")
                return
            self.track_name = self.track_var.get()
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self) -> None:
        try:
            cam_text = self.camera_var.get()
            if cam_text:
                self.selected_camera = int(cam_text.split()[-1])
            self.cap = cv2.VideoCapture(self.selected_camera)

            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open Camera {self.selected_camera}")
                return

            self.running = True
            self.start_button.config(text="Stop Camera")
            self.status_label.config(text=f"Camera {self.selected_camera} active - Waiting for race start")

            self.reset_timing_state()

            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_button.config(text="Start Camera")
        self.status_label.config(text="Camera stopped")
        self.camera_canvas.delete("all")
        self.threshold_canvas.delete("all")

    def camera_loop(self) -> None:
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.current_frame = frame
            if self.crop_config:
                result = camera.process_frame(frame, self.crop_config)
                self.handle_ocr_result(*result)
            self.root.after(0, self.update_gui)
            time.sleep(0.033)  # ~30 FPS

    def on_track_selected(self, event=None) -> None:
        self.track_name = self.track_var.get()
        self.update_best_times_display()

    def update_gui(self) -> None:
        # Display cropped ROI
        if self.current_roi is not None:
            roi_rgb = cv2.cvtColor(self.current_roi, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(roi_rgb).resize((400, 100), Image.Resampling.LANCZOS)
            self.camera_image = ImageTk.PhotoImage(img) 
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(200, 50, image=self.camera_image)

        # Display thresholded image
        if self.current_thresh is not None:
            img = Image.fromarray(self.current_thresh).resize((400, 100), Image.Resampling.LANCZOS)
            self.threshold_image = ImageTk.PhotoImage(img)
            self.threshold_canvas.delete("all")
            self.threshold_canvas.create_image(200, 50, image=self.threshold_image)

        # Update text debug
        self.current_time_label.config(text=self.current_time_text or "--:--.---")
        self.ocr_label.config(text=f"OCR: {self.current_time_text}")
        self.flash_label.config(text=f"Flash: {'Yes' if self.flash_detected else 'No'}")
        self.stable_label.config(text=f"Stable: {self.stable_frame_count}")

    def handle_ocr_result(
        self,
        time_text: str,
        roi: np.ndarray,
        thresh: np.ndarray,
        method: str,
        debug_info: dict[str, float]
    ) -> None:
        now = time.time()
        flash_started = debug_info.get("is_flash", 0.0) and not self.prev_flash
        self.prev_flash = bool(debug_info.get("is_flash", 0.0))

        # Save for GUI
        self.current_time_text = time_text
        self.flash_detected = bool(debug_info.get("is_flash", 0.0))
        self.current_roi = roi
        self.current_thresh = thresh

        # Save debug color info
        self.color_label.config(text=f"Text RGB: {int(debug_info.get('r', 0))}, {int(debug_info.get('g', 0))}, {int(debug_info.get('b', 0))}")
        self.cyan_label.config(text=f"Cyan Δ: {debug_info.get('cyan_balance', 0):.1f}")

        # Stability check
        if time_text == self.last_time_seen:
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 1
            self.last_time_seen = time_text
            self.last_time_change = now

        stable_duration = now - self.last_time_change

        if not utils.is_valid_time(time_text):
            return

        # Detect race start
        if time_text == "0:00.000":
            self.race_ready = True
            self.race_active = False
            self.status_label.config(text="Race ready - waiting for start")
            return

        if self.race_ready and not self.race_active:
            if self.lap_times:
                stats.update_global_stats_with_lap_times(self.global_stats, self.track_name, self.attempt_id, self.lap_times)
            self.race_active = True
            self.race_ready = False
            self.lap_times = []
            self.flash_counter = {}
            self.last_lap_logged_time = 0
            self.attempt_id, _ = stats.start_new_attempt(self.global_stats, self.track_name)
            self.clear_current_lap_displays()
            self.status_label.config(text="Race active!")
            self.attempt_label.config(text=f"Attempt {self.attempt_id}")
            return

        if not self.race_active:
            return

        # Count flashes
        if flash_started:
            self.flash_counter[time_text] = self.flash_counter.get(time_text, 0) + 1

        # Log lap
        if self.flash_counter.get(time_text, 0) == self.LAP_FLASH_COUNT_REQUIRED:
            if len(self.lap_times) == 0 or time_text != self.lap_times[-1]:
                if now - self.last_lap_logged_time >= self.LAP_COOLDOWN_SECONDS:
                    self.log_lap_time(time_text)
                    self.last_lap_logged_time = now

        # Final time detection
        if not self.flash_detected and stable_duration > self.FINAL_STABLE_SECONDS:
            if len(self.lap_times) > 0 and time_text != self.lap_times[-1]:
                self.log_final_time(time_text)
                self.race_active = False
                self.race_ready = False
                self.status_label.config(text="Race finished!")

    def reset_timing_state(self) -> None:
        self.race_active = False
        self.race_ready = False
        self.last_time_seen = ""
        self.stable_frame_count = 0
        self.last_time_change = time.time()
        self.last_lap_logged_time = 0
        self.flash_counter = {}
        self.prev_flash = False

    def log_lap_time(self, time_text: str) -> None:
        lap_number = len(self.lap_times) + 1
        if lap_number > 3:
            return

        lap_label = f"Lap {lap_number}"
        self.lap_times.append(time_text)

        # Delta from best lap
        best_laps = self.global_stats["track_stats"].get(self.track_name, {}).get("best_laps", {})
        best = best_laps.get(lap_label)
        delta = utils.calculate_delta(best, time_text) if best else ""

        # Color-code delta
        if delta.startswith("+"):
            color = "#FF5722"
        elif delta.startswith("-"):
            color = "#4CAF50"
        else:
            color = "#ffffff"

        self.lap_labels[lap_label].config(text=time_text)
        self.delta_labels[lap_label].config(text=delta, foreground=color)

        stats.update_best_lap(self.global_stats, self.track_name, lap_label, time_text)
        self.update_best_times_display()

    def log_final_time(self, time_text: str) -> None:
        self.lap_times.append(time_text)
        self.lap_labels["Final"].config(text=time_text)

        # Lap 3 back-calc
        if len(self.lap_times) >= 3:
            lap3 = stats.calculate_lap3_from_final(self.lap_times[0], self.lap_times[1], time_text)
            if lap3:
                self.lap_labels["Lap 3"].config(text=lap3)
                best_lap3 = self.global_stats["track_stats"].get(self.track_name, {}).get("best_laps", {}).get("Lap 3")
                delta3 = utils.calculate_delta(best_lap3, lap3) if best_lap3 else ""
                color = "#FF5722" if delta3.startswith("+") else "#4CAF50" if delta3.startswith("-") else "#ffffff"
                self.delta_labels["Lap 3"].config(text=delta3, foreground=color)
                stats.update_best_lap(self.global_stats, self.track_name, "Lap 3", lap3)

        # Final delta
        best_final = self.global_stats["track_stats"].get(self.track_name, {}).get("best_laps", {}).get("Final")
        delta = utils.calculate_delta(best_final, time_text) if best_final else ""
        color = "#FF5722" if delta.startswith("+") else "#4CAF50" if delta.startswith("-") else "#ffffff"
        self.delta_labels["Final"].config(text=delta, foreground=color)

        stats.update_best_lap(self.global_stats, self.track_name, "Final", time_text)
        stats.update_global_stats_with_lap_times(self.global_stats, self.track_name, self.attempt_id, self.lap_times)
        self.update_best_times_display()

    def clear_current_lap_displays(self) -> None:
        for label in self.lap_labels.values():
            label.config(text="--:--.---")
        for label in self.delta_labels.values():
            label.config(text="")

    def update_best_times_display(self) -> None:
        for lap in ["Lap 1", "Lap 2", "Lap 3", "Final"]:
            self.pr_labels[lap].config(text="--:--.---")
            self.best_run_labels[lap].config(text="--:--.---")

        if not self.track_name or self.track_name not in self.global_stats.get("track_stats", {}):
            return

        track_data = self.global_stats["track_stats"][self.track_name]
        best_laps = track_data.get("best_laps", {})
        attempts = track_data.get("attempts", [])

        for lap in ["Lap 1", "Lap 2", "Lap 3", "Final"]:
            best = best_laps.get(lap)
            if best:
                self.pr_labels[lap].config(text=best)

        best_final = None
        best_attempt = None
        for attempt in attempts:
            laps = attempt.get("laps", [])
            if len(laps) >= 3 and utils.is_valid_time(laps[2]):
                if best_final is None or laps[2] < best_final:
                    best_final = laps[2]
                    best_attempt = laps

        if best_attempt:
            self.best_run_labels["Lap 1"].config(text=best_attempt[0])
            self.best_run_labels["Lap 2"].config(text=best_attempt[1])
            self.best_run_labels["Final"].config(text=best_attempt[2])
            lap3 = stats.calculate_lap3_from_final(best_attempt[0], best_attempt[1], best_attempt[2])
            if lap3:
                self.best_run_labels["Lap 3"].config(text=lap3)

    def set_crop_region(self) -> None:
        if not self.running or self.current_frame is None:
            messagebox.showwarning("Warning", "Start the camera first.")
            return

        zoom = 2.5
        zoomed = cv2.resize(self.current_frame, None, fx=zoom, fy=zoom)
        r = cv2.selectROI("Select Timer Region", zoomed, False, False)
        cv2.destroyAllWindows()

        if r[2] > 0 and r[3] > 0:
            self.crop_config = {
                "x1": int(r[0] / zoom),
                "y1": int(r[1] / zoom),
                "x2": int((r[0] + r[2]) / zoom),
                "y2": int((r[1] + r[3]) / zoom)
            }
            config.save_crop_config(self.crop_config)
            messagebox.showinfo("Success", "Crop region saved.")

    def reset_session(self) -> None:
        self.lap_times = []
        self.reset_timing_state()
        self.clear_current_lap_displays()
        self.attempt_label.config(text="Session reset")
        self.status_label.config(text="Session reset - ready")
        self.update_best_times_display()

    def export_data(self) -> None:
        if not self.track_name:
            messagebox.showwarning("Warning", "No track selected.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"{self.track_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if filename:
            export = {
                "track": self.track_name,
                "export_time": datetime.now().isoformat(),
                "current_session_laps": self.lap_times,
                "track_statistics": self.global_stats.get("track_stats", {}).get(self.track_name, {}),
                "global_stats": self.global_stats
            }
            try:
                with open(filename, "w") as f:
                    import json
                    json.dump(export, f, indent=2)
                messagebox.showinfo("Exported", f"Data saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def on_closing(self) -> None:
        if self.running:
            self.stop_camera()
        if self.race_active and self.lap_times:
            stats.update_global_stats_with_lap_times(self.global_stats, self.track_name, self.attempt_id, self.lap_times)
        self.root.destroy()









