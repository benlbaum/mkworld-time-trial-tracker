import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import json
import os
import time
import threading
from datetime import datetime, timedelta
from PIL import Image, ImageTk
import numpy as np
import pytesseract
import re

class LapTimerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mario Kart Lap Timer")
        self.root.geometry("800x830")
        self.root.configure(bg='#1e1e1e')
        
        # Style configuration
        self.setup_styles()
        
        # Camera and processing variables
        self.selected_camera = 1  # Default camera
        self.cap = None
        self.camera_thread = None
        self.running = False
        self.current_frame = None
        self.current_roi = None
        self.current_thresh = None
        self.crop_config = None
        
        # Lap timing variables (from your original code)
        self.track_name = ""
        self.lap_times = []
        self.attempt_id = 1
        self.current_attempt_id = 1
        self.race_active = False
        self.race_ready = False
        self.last_time_seen = ""
        self.stable_frame_count = 0
        self.last_time_change = time.time()
        self.last_lap_logged_time = 0
        self.flash_counter = {}
        self.prev_flash = False
        
        # Configuration constants
        self.STABLE_FRAME_THRESHOLD = 3
        self.FINAL_STABLE_SECONDS = 2.0
        self.LAP_FLASH_COUNT_REQUIRED = 3
        self.LAP_COOLDOWN_SECONDS = 5
        
        # GUI state
        self.current_time_text = ""
        self.flash_detected = False
        self.global_stats = {}
        self.debug_panel_visible = True
        
        # Create GUI
        self.create_widgets()
        
        # Load configuration
        self.load_configs()
        
    def setup_styles(self):
        """Configure modern dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for dark theme
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabelFrame', background='#1e1e1e', foreground='#ffffff')
        style.configure('TLabelFrame.Label', background='#1e1e1e', foreground='#ffffff')
        
        style.configure('Title.TLabel', 
                       background='#1e1e1e', 
                       foreground='#ffffff', 
                       font=('Arial', 24, 'bold'))
        
        style.configure('Heading.TLabel', 
                       background='#1e1e1e', 
                       foreground='#4CAF50', 
                       font=('Arial', 14, 'bold'))
        
        style.configure('Data.TLabel', 
                       background='#1e1e1e', 
                       foreground='#ffffff', 
                       font=('Arial', 14))
        
        style.configure('Time.TLabel', 
                       background='#1e1e1e', 
                       foreground='#FFD700', 
                       font=('Arial', 20, 'bold'))
        
        style.configure('Best.TLabel', 
                       background='#1e1e1e', 
                       foreground='#4CAF50', 
                       font=('Arial', 18, 'bold'))
        
        style.configure('Current.TLabel', 
                       background='#1e1e1e', 
                       foreground='#2196F3', 
                       font=('Arial', 18, 'bold'))
        
        style.configure('LapHeading.TLabel', 
               background='#1e1e1e', 
               foreground='#4CAF50', 
               font=('Arial', 16, 'bold'))
        
        style.configure('Modern.TButton', 
                       background='#4CAF50', 
                       foreground='white', 
                       borderwidth=0, 
                       focuscolor='none')
        
        style.map('Modern.TButton',
                 background=[('active', '#45a049')])
        
        style.configure('DebugSmall.TLabel', 
                    background='#1e1e1e', 
                    foreground='#ffffff', 
                    font=('Arial', 10))  # Smaller font than the regular 12
        
        # Better dark combobox styling - replace the existing TCombobox configuration
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

        # Also add styling for the dropdown list
        style.configure('TCombobox.Listbox',
                    background='#2d2d2d',
                    foreground='white',
                    selectbackground='#4CAF50',
                    selectforeground='white')
        
        # Large combobox style - add this to setup_styles()
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
        
        # Custom dark LabelFrame style - add this to setup_styles()
        style.configure('Dark.TLabelframe', 
                    background='#1e1e1e', 
                    foreground='#ffffff',
                    borderwidth=2,
                    relief='groove')
        style.configure('Dark.TLabelframe.Label', 
                    background='#1e1e1e', 
                    foreground='#ffffff',
                    font=('Arial', 12, 'bold'))

        # Better dropdown styling for Large combobox
        style.configure('Large.TCombobox.Listbox',
                    background='#2d2d2d',          # Dark background
                    foreground='white',             # White text
                    selectbackground='#4CAF50',     # Green selection
                    selectforeground='white',       # White selected text
                    font=('Arial', 16),             # Larger font
                    borderwidth=0,                  # Remove borders
                    relief='flat')                  # Flat appearance

        # Also configure the popdown window
        self.root.option_add('*TCombobox*Listbox.font', 'Arial 10')
        self.root.option_add('*TCombobox*Listbox.background', '#2d2d2d')
        self.root.option_add('*TCombobox*Listbox.foreground', 'white')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#4CAF50')
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        self.root.option_add('*TCombobox*Listbox.borderWidth', '0')
        
    def toggle_debug_panel(self):
        """Toggle the debug panel visibility and resize window"""
        if self.debug_panel_visible:
            self.debug_panel.pack_forget()
            self.toggle_debug_button.config(text="Show Debug/Controls ▼")
            self.debug_panel_visible = False
            # Resize to compact size
            self.root.geometry("800x480")
        else:
            self.debug_panel.pack(fill='both', expand=True, pady=(5, 0))
            self.toggle_debug_button.config(text="Hide Debug/Controls ▲")
            self.debug_panel_visible = True
            # Resize to full size
            self.root.geometry("800x830")
        
        # Force the window to update its layout
        self.root.update_idletasks()
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container with dark background
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        main_frame.configure(style='TFrame')
        
        # Title
        title_label = ttk.Label(main_frame, text="Mario Kart Lap Timer", style='Title.TLabel')
        title_label.pack(pady=(0, 10))

        # Main content area - Combined Race Info
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='x', pady=(0, 5))

        # Track selector above the race frame
        track_select_frame = ttk.Frame(content_frame)
        track_select_frame.pack(fill='x')

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

        # Single combined panel - back to simple title or no title
        race_frame = ttk.LabelFrame(content_frame, padding=15, style='Dark.TLabelframe')
        race_frame.pack(fill='both', expand=True)

        # Current time display
        time_frame = ttk.Frame(race_frame)
        time_frame.pack(fill='x', pady=(0, 20))

        ttk.Label(time_frame, text="Current Time:", style='LapHeading.TLabel').pack(side='left', padx=(0, 10))
        self.current_time_label = ttk.Label(time_frame, text="--:--.---", style='Time.TLabel')
        self.current_time_label.pack(side='left')

        # Three-column layout for lap times - single grid container
        lap_times_frame = ttk.Frame(race_frame)
        lap_times_frame.pack(fill='x', pady=(0, 20))

        # Configure column weights ONCE for the entire grid
        lap_times_frame.grid_columnconfigure(1, weight=1)  # Current column
        lap_times_frame.grid_columnconfigure(2, weight=1)  # PR column  
        lap_times_frame.grid_columnconfigure(3, weight=1)  # Best run column

        # Create lap time displays dictionaries
        self.lap_labels = {}
        self.delta_labels = {}
        self.pr_labels = {}
        self.best_run_labels = {}

        # Row 0: Headers - with centered text
        # Current header - centered
        current_header_frame = ttk.Frame(lap_times_frame)
        current_header_frame.grid(row=0, column=1, sticky='ew', padx=1, pady=(0, 10))
        ttk.Label(current_header_frame, text="Current Lap Times", style='LapHeading.TLabel').pack()

        # PR header - centered  
        pr_header_frame = ttk.Frame(lap_times_frame)
        pr_header_frame.grid(row=0, column=2, sticky='ew', padx=1, pady=(0, 10))
        ttk.Label(pr_header_frame, text="Personal Records", style='LapHeading.TLabel').pack()

        # Best run header - centered
        best_header_frame = ttk.Frame(lap_times_frame)
        best_header_frame.grid(row=0, column=3, sticky='ew', padx=1, pady=(0, 10))
        ttk.Label(best_header_frame, text="Best Overall Run", style='LapHeading.TLabel').pack()

        # Rows 1-3: Lap times
        for i in range(1, 4):
            row = i  # Row 1, 2, 3
            
            # Lap label
            ttk.Label(lap_times_frame, text=f"Lap {i}:", style='Data.TLabel', width=8).grid(row=row, column=0, sticky='w', pady=2)
            
            # Current lap time column
            current_col = ttk.Frame(lap_times_frame, style='Debug.TFrame')
            current_col.grid(row=row, column=1, sticky='ew', padx=1, pady=2)
            
            current_content = ttk.Frame(current_col)
            current_content.pack()
            self.lap_labels[f"Lap {i}"] = ttk.Label(current_content, text="--:--.---", style='Current.TLabel')
            self.lap_labels[f"Lap {i}"].pack(side='left')
            self.delta_labels[f"Lap {i}"] = ttk.Label(current_content, text="", style='Data.TLabel')
            self.delta_labels[f"Lap {i}"].pack(side='left', padx=(5, 0))
            
            # Personal record column
            pr_col = ttk.Frame(lap_times_frame, style='Debug.TFrame')
            pr_col.grid(row=row, column=2, sticky='ew', padx=1, pady=2)
            
            self.pr_labels[f"Lap {i}"] = ttk.Label(pr_col, text="--:--.---", style='Best.TLabel')
            self.pr_labels[f"Lap {i}"].pack()
            
            # Best run column
            best_run_col = ttk.Frame(lap_times_frame, style='Debug.TFrame')
            best_run_col.grid(row=row, column=3, sticky='ew', padx=1, pady=2)
            
            self.best_run_labels[f"Lap {i}"] = ttk.Label(best_run_col, text="--:--.---", style='Current.TLabel')
            self.best_run_labels[f"Lap {i}"].pack()

        # Row 4: Final time
        ttk.Label(lap_times_frame, text="Final:", style='Data.TLabel', width=8).grid(row=4, column=0, sticky='w', pady=2)

        # Current final column
        current_final_col = ttk.Frame(lap_times_frame, style='Debug.TFrame')
        current_final_col.grid(row=4, column=1, sticky='ew', padx=1, pady=2)

        current_final_content = ttk.Frame(current_final_col)
        current_final_content.pack()
        self.lap_labels["Final"] = ttk.Label(current_final_content, text="--:--.---", style='Current.TLabel')
        self.lap_labels["Final"].pack(side='left')
        self.delta_labels["Final"] = ttk.Label(current_final_content, text="", style='Data.TLabel')
        self.delta_labels["Final"].pack(side='left', padx=(5, 0))

        # PR final column
        pr_final_col = ttk.Frame(lap_times_frame, style='Debug.TFrame')
        pr_final_col.grid(row=4, column=2, sticky='ew', padx=1, pady=2)

        self.pr_labels["Final"] = ttk.Label(pr_final_col, text="--:--.---", style='Best.TLabel')
        self.pr_labels["Final"].pack()

        # Best run final column
        best_run_final_col = ttk.Frame(lap_times_frame, style='Debug.TFrame')
        best_run_final_col.grid(row=4, column=3, sticky='ew', padx=1, pady=2)

        self.best_run_labels["Final"] = ttk.Label(best_run_final_col, text="--:--.---", style='Current.TLabel')
        self.best_run_labels["Final"].pack()

        # Toggle button for debug panel
        self.toggle_debug_button = ttk.Button(main_frame, text="Hide Debug/Controls ▲", 
                                            command=self.toggle_debug_panel, style='Modern.TButton')
        self.toggle_debug_button.pack(pady=(10, 5))

        # Collapsible debug/controls panel
        self.debug_panel = ttk.LabelFrame(main_frame, text="Debug & Controls", padding=10, style='Dark.TLabelframe')
        self.debug_panel.pack(fill='x', expand=False, pady=(5, 0))

        # Top row - Controls
        top_row = ttk.Frame(self.debug_panel)
        top_row.pack(fill='x', pady=(0, 15))

        # Left side - Track selection and buttons
        controls_left = ttk.Frame(top_row)
        controls_left.pack(side='left', fill='y', padx=(0, 20))

        # Control buttons
        button_frame = ttk.Frame(controls_left)
        button_frame.pack(fill='x', pady=(0, 10))

        self.start_button = ttk.Button(button_frame, text="Start Camera", 
                                    command=self.toggle_camera, style='Modern.TButton')
        self.start_button.pack(side='left', padx=(0, 5))

        self.crop_button = ttk.Button(button_frame, text="Set Crop Region", 
                                    command=self.set_crop_region, style='Modern.TButton')
        self.crop_button.pack(side='left', padx=(0, 5))

        # Camera selection
        camera_frame = ttk.Frame(controls_left)
        camera_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(camera_frame, text="Camera:", style='Heading.TLabel').pack(side='left', padx=(0, 10))
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, width=15, state='readonly')
        self.camera_combo['values'] = ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
        self.camera_combo.set("Camera 1")  # Default to camera 1
        self.camera_combo.pack(side='left')

        # Status and debug info
        status_debug_frame = ttk.Frame(controls_left)
        status_debug_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(status_debug_frame, text="Status:", style='Heading.TLabel').pack(anchor='w')
        self.status_label = ttk.Label(status_debug_frame, text="Ready to start", style='Data.TLabel')
        self.status_label.pack(anchor='w')

        ttk.Label(status_debug_frame, text="Current Attempt:", style='Heading.TLabel').pack(anchor='w', pady=(10, 0))
        self.attempt_label = ttk.Label(status_debug_frame, text="No active attempt", style='Data.TLabel')
        self.attempt_label.pack(anchor='w')

        ttk.Label(status_debug_frame, text="Debug Info:", style='Heading.TLabel').pack(anchor='w', pady=(10, 0))
        self.ocr_label = ttk.Label(status_debug_frame, text="OCR: --", style='DebugSmall.TLabel')
        self.ocr_label.pack(anchor='w')
        self.stable_label = ttk.Label(status_debug_frame, text="Stable: 0", style='DebugSmall.TLabel')
        self.stable_label.pack(anchor='w')
        self.flash_label = ttk.Label(status_debug_frame, text="Flash: No", style='DebugSmall.TLabel')
        self.flash_label.pack(anchor='w')
        self.method_label = ttk.Label(status_debug_frame, text="Method: --", style='DebugSmall.TLabel')
        self.method_label.pack(anchor='w')
        self.color_label = ttk.Label(status_debug_frame, text="Text RGB: --", style='DebugSmall.TLabel')
        self.color_label.pack(anchor='w')
        self.cyan_label = ttk.Label(status_debug_frame, text="Cyan: --", style='DebugSmall.TLabel')
        self.cyan_label.pack(anchor='w')

        # Right side - Camera feeds
        camera_frame = ttk.Frame(top_row)
        camera_frame.pack(side='right', fill='both', expand=True)

        # Camera feed
        ttk.Label(camera_frame, text="Timer Region:", style='Heading.TLabel').pack(anchor='w')
        self.camera_canvas = tk.Canvas(camera_frame, width=400, height=100, bg='black')
        self.camera_canvas.pack(anchor='w', pady=(5, 10))

        # Threshold view
        ttk.Label(camera_frame, text="Processed Image:", style='Heading.TLabel').pack(anchor='w')
        self.threshold_canvas = tk.Canvas(camera_frame, width=400, height=100, bg='black')
        self.threshold_canvas.pack(anchor='w', pady=(5, 0))

    def load_configs(self):
        """Load saved configurations"""
        if os.path.exists("crop_config.json"):
            with open("crop_config.json", "r") as f:
                self.crop_config = json.load(f)
        
        self.load_global_stats()
    
    def load_global_stats(self):
        """Load global statistics"""
        if os.path.exists("global_stats.json"):
            with open("global_stats.json", "r") as f:
                self.global_stats = json.load(f)
        else:
            self.global_stats = {"global_attempt_id": 1, "track_stats": {}}
    
    def save_global_stats(self):
        """Save global statistics"""
        with open("global_stats.json", "w") as f:
            json.dump(self.global_stats, f, indent=2)
    
    def on_track_selected(self, event=None):
        """Handle track selection"""
        self.track_name = self.track_var.get()
        self.update_best_times_display()
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.running:
            if not self.track_var.get():
                messagebox.showwarning("Warning", "Please select a track first!")
                return
            
            self.track_name = self.track_var.get()
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture"""
        try:
            # Get selected camera index
            camera_text = self.camera_var.get()
            if camera_text:
                self.selected_camera = int(camera_text.split()[-1])  # Extract number from "Camera X"
            
            self.cap = cv2.VideoCapture(self.selected_camera)  # Use selected camera
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open Camera {self.selected_camera}")
                return
            
            self.running = True
            self.start_button.config(text="Stop Camera")
            self.status_label.config(text=f"Camera {self.selected_camera} active - Waiting for race start")
            
            # Initialize timing variables
            self.reset_timing_state()
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(text="Start Camera")
        self.status_label.config(text="Camera stopped")
        
        # Clear canvas
        self.camera_canvas.delete("all")
        self.threshold_canvas.delete("all")
    
    def reset_timing_state(self):
        """Reset all timing-related state"""
        self.race_active = False
        self.race_ready = False
        self.last_time_seen = ""
        self.stable_frame_count = 0
        self.last_time_change = time.time()
        self.last_lap_logged_time = 0
        self.flash_counter = {}
        self.prev_flash = False
        
    def camera_loop(self):
        """Main camera processing loop"""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.current_frame = frame
            
            # Process frame if crop region is set
            if self.crop_config:
                self.process_frame(frame)
            
            # Update GUI (must be done in main thread)
            self.root.after(0, self.update_gui)
            
            time.sleep(0.033)  # ~30 FPS
    
    def process_frame(self, frame):
        """Process the current frame for lap timing - integrated from original code"""
        crop = self.crop_config
        roi = frame[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]
        
        if roi.size == 0:
            return
        
        # Resize and process (from original code)
        roi_resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholding approaches for better text detection
        # Method 1: Original fixed threshold
        _, thresh1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Method 2: Adaptive threshold (better for varying lighting)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: OTSU threshold (automatically finds best threshold)
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 4: Inverted threshold for dark text on light background
        _, thresh4 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Try OCR on all threshold methods and pick the best result
        config = '--psm 7 -c tessedit_char_whitelist=0123456789.:'
        
        time_results = []
        thresholds = [thresh1, thresh2, thresh3, thresh4]
        threshold_names = ["Fixed", "Adaptive", "OTSU", "Inverted"]
        
        best_time_text = ""
        best_thresh = thresh1
        best_method = "Fixed"
        
        for i, thresh in enumerate(thresholds):
            try:
                time_text = pytesseract.image_to_string(thresh, config=config).strip()
                if self.is_valid_time(time_text):
                    # Found valid time, use this one
                    best_time_text = time_text
                    best_thresh = thresh
                    best_method = threshold_names[i]
                    break
                elif time_text:  # Non-empty but invalid format
                    time_results.append((time_text, thresh, threshold_names[i]))
            except:
                continue
        
        # If no valid time found, use the first non-empty result or fallback to original
        if not best_time_text and time_results:
            best_time_text, best_thresh, best_method = time_results[0]
        elif not best_time_text:
            # Fallback to original method
            best_time_text = pytesseract.image_to_string(thresh1, config=config).strip()
            best_thresh = thresh1
            best_method = "Fixed (fallback)"
        
        self.current_roi = roi_resized
        self.current_thresh = best_thresh
        
        # Update debug info to show which method worked
        self.current_thresh_method = best_method
        
        # Advanced flash detection with debug info
        mask = best_thresh.astype(bool)
        
        # Flash detection - look at only the text pixels, not background
        # Use the threshold mask to identify actual text pixels
        text_mask = best_thresh > 127  # White pixels in threshold = text pixels
        
        if np.any(text_mask):
            # Get color values of ONLY the text pixels (not background)
            text_pixels = roi_resized[text_mask]
            text_avg_color = np.mean(text_pixels, axis=0)
            text_r, text_g, text_b = text_avg_color
            text_brightness = (text_r + text_g + text_b) / 3
            
            # Calculate text-specific metrics
            text_blue_dominance = text_b - text_r
            text_cyan_effect = (text_g + text_b) / 2 - text_r
            
            # Flash detection based on pure text color (no background contamination)
            # Normal text: RGB(236,205,150) = Yellowish-white, Brightness 198
            # Flash text: RGB(100,205,205) = Cyan, Brightness 170
            
            # Flash characteristics:
            # 1. Green and Blue are equal or very close (cyan effect)
            # 2. Red is much lower than Green/Blue
            # 3. Green and Blue values are high (>180)
            
            cyan_balance = abs(text_g - text_b)  # How close G and B are (low = more cyan)
            red_suppression = (text_g + text_b) / 2 - text_r  # How much G+B exceed R
            
            # Flash detection: cyan color (G≈B and both >> R)
            is_flash = (cyan_balance < 20 and red_suppression > 80 and text_g > 180 and text_b > 180)
            
            # Store additional debug values
            self.debug_cyan_balance = cyan_balance
            self.debug_red_suppression = red_suppression
            
            # Store debug values for display - now showing pure text color
            self.debug_r = text_r
            self.debug_g = text_g  
            self.debug_b = text_b
            self.debug_avg_brightness = text_brightness
            self.debug_blue_dominance = text_blue_dominance
            self.debug_cyan_effect = text_cyan_effect
        else:
            is_flash = False
            # Initialize debug values
            self.debug_cyan_balance = 0
            self.debug_red_suppression = 0
            self.debug_r = 0
            self.debug_g = 0
            self.debug_b = 0
            self.debug_avg_brightness = 0
            self.debug_blue_dominance = 0
            self.debug_cyan_effect = 0
        
        flash_started = is_flash and not self.prev_flash
        self.prev_flash = is_flash
        
        # Update GUI state
        self.current_time_text = best_time_text
        self.flash_detected = is_flash
        
        # Timing logic (from original code)
        now = time.time()
        self.stable_frame_count = self.stable_frame_count + 1 if time_text == self.last_time_seen else 1
        if time_text != self.last_time_seen:
            self.last_time_seen = time_text
            self.last_time_change = now
        
        stable_time_duration = now - self.last_time_change
        
        if self.is_valid_time(time_text):
            if time_text == "0:00.000":
                self.race_ready = True
                self.race_active = False
                self.root.after(0, lambda: self.status_label.config(text="Race ready - waiting for start"))
            elif self.race_ready and not self.race_active:
                # Save previous attempt data before starting new one
                if self.lap_times:
                    self.update_global_stats_with_lap_times(self.current_attempt_id, self.lap_times)
                
                # Start new race
                self.race_active = True
                self.race_ready = False
                self.lap_times = []
                self.flash_counter = {}
                self.last_lap_logged_time = 0
                self.current_attempt_id, _ = self.start_new_attempt()
                
                # Clear GUI lap displays for new attempt
                self.root.after(0, self.clear_current_lap_displays)
                
                self.root.after(0, lambda: self.status_label.config(text="Race active!"))
                self.root.after(0, lambda: self.attempt_label.config(text=f"Attempt {self.current_attempt_id}"))
                
            if self.race_active:
                if flash_started:
                    self.flash_counter[time_text] = self.flash_counter.get(time_text, 0) + 1
                
                if self.flash_counter.get(time_text, 0) == self.LAP_FLASH_COUNT_REQUIRED:
                    if len(self.lap_times) == 0 or time_text != self.lap_times[-1]:
                        if now - self.last_lap_logged_time >= self.LAP_COOLDOWN_SECONDS:
                            self.log_lap_time(time_text)
                            self.last_lap_logged_time = now
                
                if not is_flash and stable_time_duration > self.FINAL_STABLE_SECONDS:
                    if len(self.lap_times) > 0 and time_text != self.lap_times[-1]:
                        self.log_final_time(time_text)
                        self.race_active = False
                        self.race_ready = False
                        
                        selfroot.after(0, lambda: self.status_label.config(text="Race finished!"))
    
    def is_valid_time(self, text):
        """Check if text is a valid time format"""
        return bool(re.match(r"^\d+:\d{2}\.\d{3}$", text))
    
    def start_new_attempt(self):
        """Start a new attempt and return attempt ID"""
        global_id = self.global_stats["global_attempt_id"]
        
        if self.track_name not in self.global_stats["track_stats"]:
            self.global_stats["track_stats"][self.track_name] = {
                "best_laps": {},
                "attempts": []
            }
        
        # Track new attempt
        self.global_stats["track_stats"][self.track_name]["attempts"].append({
            "global_id": global_id,
            "datetime": datetime.now().isoformat()
        })
        
        self.global_stats["global_attempt_id"] += 1
        self.save_global_stats()
        
        return global_id, self.global_stats["track_stats"][self.track_name]["best_laps"]
    
    def clear_current_lap_displays(self):
        """Clear current lap time displays for new attempt"""
        for label in self.lap_labels.values():
            label.config(text="--:--.---")
        
        for label in self.delta_labels.values():
            label.config(text="")
        
        # Note: We don't clear PR labels or best run labels as those persist across attempts
    
    def log_lap_time(self, time_text):
        """Log a lap time"""
        lap_number = len(self.lap_times) + 1
        lap_label = f"Lap {lap_number}"
        
        # Don't log if we already have 3 laps (should be handled by final time logic)
        if lap_number > 3:
            return
        
        # Get previous best for delta calculation
        prev_best_laps = self.global_stats["track_stats"].get(self.track_name, {}).get("best_laps", {})
        prev_best = prev_best_laps.get(lap_label)
        
        self.lap_times.append(time_text)
        
        # Calculate and display delta
        delta_text = ""
        if prev_best:
            delta_text = self.calculate_delta(prev_best, time_text)
            # Set color based on delta
            if delta_text.startswith("+"):
                delta_color = "#FF5722"  # Red for slower
            elif delta_text.startswith("-"):
                delta_color = "#4CAF50"  # Green for faster
            else:
                delta_color = "#ffffff"  # White for equal
                
            self.root.after(0, lambda: self.delta_labels[lap_label].config(text=delta_text, foreground=delta_color))
        
        # Update best lap
        self.update_best_lap(lap_label, time_text)
        
        # Update GUI
        self.root.after(0, lambda: self.lap_labels[lap_label].config(text=time_text))
        self.root.after(0, self.update_best_times_display)
    
    def log_final_time(self, time_text):
        """Log the final time"""
        final_time_text = time_text
        self.lap_times.append(final_time_text)
        self.update_best_lap("Final", final_time_text)
        
        # Calculate Lap 3 if we have enough data
        if len(self.lap_times) == 3:
            try:
                lap1_td = self.parse_time(self.lap_times[0])
                lap2_td = self.parse_time(self.lap_times[1])
                final_td = self.parse_time(final_time_text)
                
                if lap1_td and lap2_td and final_td:
                    lap3_td = final_td - lap1_td - lap2_td
                    lap3_str = self.format_time(lap3_td)
                    self.update_best_lap("Lap 3", lap3_str)
                    
                    # Calculate delta for Lap 3
                    prev_best_laps = self.global_stats["track_stats"].get(self.track_name, {}).get("best_laps", {})
                    prev_best_lap3 = prev_best_laps.get("Lap 3")
                    if prev_best_lap3:
                        delta_text = self.calculate_delta(prev_best_lap3, lap3_str)
                        if delta_text.startswith("+"):
                            delta_color = "#FF5722"
                        elif delta_text.startswith("-"):
                            delta_color = "#4CAF50"
                        else:
                            delta_color = "#ffffff"
                        self.root.after(0, lambda: self.delta_labels["Lap 3"].config(text=delta_text, foreground=delta_color))
                    
                    self.root.after(0, lambda: self.lap_labels["Lap 3"].config(text=lap3_str))
            except Exception as e:
                print(f"Lap 3 calculation failed: {e}")
        
        # Calculate delta for final time
        prev_best_laps = self.global_stats["track_stats"].get(self.track_name, {}).get("best_laps", {})
        prev_best_final = prev_best_laps.get("Final")
        if prev_best_final:
            delta_text = self.calculate_delta(prev_best_final, final_time_text)
            if delta_text.startswith("+"):
                delta_color = "#FF5722"
            elif delta_text.startswith("-"):
                delta_color = "#4CAF50"
            else:
                delta_color = "#ffffff"
            self.root.after(0, lambda: self.delta_labels["Final"].config(text=delta_text, foreground=delta_color))
        
        # Update global stats
        self.update_global_stats_with_lap_times(self.current_attempt_id, self.lap_times)
        
        # Update GUI
        self.root.after(0, lambda: self.lap_labels["Final"].config(text=final_time_text))
        self.root.after(0, self.update_best_times_display)
    
    def parse_time(self, time_str):
        """Parse time string to timedelta"""
        try:
            minutes, rest = time_str.split(":")
            seconds, millis = rest.split(".")
            return timedelta(minutes=int(minutes), seconds=int(seconds), milliseconds=int(millis))
        except:
            return None
    
    def format_time(self, td):
        """Format timedelta to time string"""
        return f"{td.seconds // 60}:{td.seconds % 60:02d}.{int(td.microseconds / 1000):03d}"
    
    def calculate_delta(self, best_time, current_time):
        """Calculate delta between best and current time"""
        best_td = self.parse_time(best_time)
        curr_td = self.parse_time(current_time)
        
    def calculate_delta(self, best_time, current_time):
        """Calculate delta between best and current time"""
        best_td = self.parse_time(best_time)
        curr_td = self.parse_time(current_time)
        
        if best_td and curr_td:
            diff = curr_td - best_td
            seconds = round(diff.total_seconds(), 3)
            sign = "+" if seconds >= 0 else ""
            return f"{sign}{seconds:.3f}s"
        return ""
    
    def update_best_lap(self, lap_label, lap_time):
        """Update best lap time"""
        track_data = self.global_stats["track_stats"][self.track_name]
        best = track_data["best_laps"].get(lap_label)
        
        if best is None or lap_time < best:
            track_data["best_laps"][lap_label] = lap_time
        
        self.save_global_stats()
    
    def update_global_stats_with_lap_times(self, attempt_id, lap_times):
        """Update global stats with lap times"""
        track_data = self.global_stats["track_stats"].setdefault(self.track_name, {"best_laps": {}, "attempts": []})
        
        for attempt in track_data["attempts"]:
            if attempt["global_id"] == attempt_id:
                attempt["laps"] = lap_times
                break
        
        self.save_global_stats()
    
    def update_gui(self):
        """Update GUI elements with current data"""
        # Update camera displays
        if self.current_roi is not None:
            # Update camera canvas
            roi_rgb = cv2.cvtColor(self.current_roi, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)
            roi_resized = roi_pil.resize((400, 100), Image.LANCZOS)
            roi_photo = ImageTk.PhotoImage(roi_resized)
            
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(200, 50, image=roi_photo)
            self.camera_canvas.image = roi_photo  # Keep reference
        
        if self.current_thresh is not None:
            # Update threshold canvas
            thresh_pil = Image.fromarray(self.current_thresh)
            thresh_resized = thresh_pil.resize((400, 100), Image.LANCZOS)
            thresh_photo = ImageTk.PhotoImage(thresh_resized)
            
            self.threshold_canvas.delete("all")
            self.threshold_canvas.create_image(200, 50, image=thresh_photo)
            self.threshold_canvas.image = thresh_photo  # Keep reference
        
        # Update text displays
        self.current_time_label.config(text=self.current_time_text or "--:--.---")
        self.flash_label.config(text=f"Flash: {'Yes' if self.flash_detected else 'No'}")
        self.ocr_label.config(text=f"OCR: {self.current_time_text}")
        self.stable_label.config(text=f"Stable: {self.stable_frame_count}")
    
    def set_crop_region(self):
        """Set the crop region for timer detection"""
        if not self.running or self.current_frame is None:
            messagebox.showwarning("Warning", "Please start the camera first!")
            return
        
        # Use OpenCV's selectROI
        zoom_factor = 2.5
        zoomed = cv2.resize(self.current_frame, None, fx=zoom_factor, fy=zoom_factor)
        r = cv2.selectROI("Select Timer Region (Press SPACE or ENTER to confirm, ESC to cancel)", zoomed, False, False)
        cv2.destroyAllWindows()
        
        if r[2] > 0 and r[3] > 0:  # Valid selection
            self.crop_config = {
                "x1": int(r[0] / zoom_factor),
                "y1": int(r[1] / zoom_factor),
                "x2": int((r[0] + r[2]) / zoom_factor),
                "y2": int((r[1] + r[3]) / zoom_factor)
            }
            
            # Save config
            with open("crop_config.json", "w") as f:
                json.dump(self.crop_config, f)
            
            messagebox.showinfo("Success", "Crop region set successfully!")
    
    def reset_session(self):
        """Reset the current session"""
        self.lap_times = []
        self.reset_timing_state()
        
        # Clear lap displays
        for label in self.lap_labels.values():
            label.config(text="--:--.---")
        
        for label in self.delta_labels.values():
            label.config(text="")
        
        self.attempt_label.config(text="Session reset")
        self.status_label.config(text="Session reset - ready to start")
        self.update_best_times_display()
    
    def update_best_times_display(self):
        """Update the best times display with PR times and best overall run"""
        # Clear all displays first
        for lap in ["Lap 1", "Lap 2", "Lap 3", "Final"]:
            self.pr_labels[lap].config(text="--:--.---")
            self.best_run_labels[lap].config(text="--:--.---")
        
        if not self.track_name or self.track_name not in self.global_stats.get("track_stats", {}):
            return
        
        track_data = self.global_stats["track_stats"][self.track_name]
        best_laps = track_data.get("best_laps", {})
        attempts = track_data.get("attempts", [])
        
        # Update Personal Records (best individual lap times)
        for lap in ["Lap 1", "Lap 2", "Lap 3", "Final"]:
            best_time = best_laps.get(lap, "--:--.---")
            self.pr_labels[lap].config(text=best_time)
        
        # Find the attempt with the best overall final time
        best_overall_attempt = None
        best_final_time = None
        
        for attempt in attempts:
            # Only consider attempts with complete lap data (3 times: lap1, lap2, final)
            if "laps" in attempt and len(attempt["laps"]) == 3:
                final_time = attempt["laps"][2]
                
                if self.is_valid_time(final_time):
                    if best_final_time is None or final_time < best_final_time:
                        best_final_time = final_time
                        best_overall_attempt = attempt
        
        # Update Best Overall Run column
        if best_overall_attempt and "laps" in best_overall_attempt:
            laps = best_overall_attempt["laps"]
            
            # Show the lap times from the best overall run
            self.best_run_labels["Lap 1"].config(text=laps[0])
            self.best_run_labels["Lap 2"].config(text=laps[1])
            
            # Calculate Lap 3 from the complete attempt
            try:
                lap1_td = self.parse_time(laps[0])
                lap2_td = self.parse_time(laps[1])
                final_td = self.parse_time(laps[2])
                
                if lap1_td and lap2_td and final_td:
                    lap3_td = final_td - lap1_td - lap2_td
                    lap3_str = self.format_time(lap3_td)
                    self.best_run_labels["Lap 3"].config(text=lap3_str)
            except Exception as e:
                print(f"Lap 3 calculation failed for best run: {e}")
            
            # Show final time
            self.best_run_labels["Final"].config(text=laps[2])
    
    def export_data(self):
        """Export session data to file"""
        if not self.track_name:
            messagebox.showwarning("Warning", "No track selected!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialname=f"mario_kart_{self.track_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            export_data = {
                "track": self.track_name,
                "export_time": datetime.now().isoformat(),
                "current_session_laps": self.lap_times,
                "track_statistics": self.global_stats.get("track_stats", {}).get(self.track_name, {}),
                "global_stats": self.global_stats
            }
            
            try:
                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def on_closing(self):
        """Handle application closing"""
        if self.running:
            self.stop_camera()
        
        # Save any pending lap times
        if self.race_active and self.lap_times:
            self.update_global_stats_with_lap_times(self.current_attempt_id, self.lap_times)
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = LapTimerGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Bind keyboard shortcuts
    def on_key_press(event):
        if event.char == 'q' or event.char == 'Q':
            app.on_closing()
        elif event.char == 'r' or event.char == 'R':
            if app.running:
                app.reset_session()
        elif event.char == 's' or event.char == 'S':
            if app.track_name:
                app.export_data()
        elif event.char == 'c' or event.char == 'C':
            if app.running:
                app.set_crop_region()
    
    root.bind('<KeyPress>', on_key_press)
    root.focus_set()  # Allow keyboard events
    
    root.mainloop()

if __name__ == "__main__":
    main()

                        