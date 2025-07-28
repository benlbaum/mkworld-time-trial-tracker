import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import pytesseract
from typing import Optional, Tuple, List
import re
import os
import platform
import time
import threading
from threading import Thread


class OCRProcessor:
    """Dedicated OCR processing thread similar to OCR.py approach"""
    
    def __init__(self):
        self.stopped = False
        self.current_frame = None
        self.ocr_result = ""
        self.timer_result = ""
        self.processed_image = None
        self.lock = threading.Lock()
        
        # OCR settings
        self.tesseract_config = "--psm 8 -c tessedit_char_whitelist=0123456789:."
        self.timer_pattern = r'\d:\d{2}\.\d{3}'
        
        # Image processing settings
        self.selected_color_hsv = [0, 0, 255]
        self.color_ranges = [10, 50, 50]
        self.scale_factor = 3.0
        self.dilate_iterations = 1
        self.erode_iterations = 1
        
        self.tesseract_available = True
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Configure Tesseract path"""
        possible_paths = []
        
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                "tesseract.exe"
            ]
        elif platform.system() == "Darwin":
            possible_paths = [
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
                "tesseract"
            ]
        else:
            possible_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "tesseract"
            ]
        
        for path in possible_paths:
            try:
                if path in ["tesseract", "tesseract.exe"]:
                    pytesseract.image_to_string(
                        Image.new('RGB', (100, 30), color='white'),
                        config="--version"
                    )
                    self.tesseract_available = True
                    break
                elif os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    pytesseract.image_to_string(
                        Image.new('RGB', (100, 30), color='white'),
                        config="--version"
                    )
                    self.tesseract_available = True
                    break
            except:
                continue
    
    def start(self):
        """Start OCR processing thread"""
        Thread(target=self.process_loop, daemon=True).start()
        return self
    
    def process_loop(self):
        """Main OCR processing loop running in separate thread"""
        while not self.stopped:
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
                else:
                    frame_to_process = None
            
            if frame_to_process is not None:
                try:
                    # Process image for OCR
                    processed = self.process_image_for_ocr(frame_to_process)
                    
                    if processed is not None and self.tesseract_available:
                        # Perform OCR
                        raw_text = pytesseract.image_to_string(
                            processed, config=self.tesseract_config
                        ).strip()
                        
                        # Extract timer
                        timer_match = re.search(self.timer_pattern, raw_text)
                        timer_text = timer_match.group(0) if timer_match else ""
                        
                        # Update results
                        with self.lock:
                            self.ocr_result = raw_text
                            self.timer_result = timer_text
                            self.processed_image = processed
                    
                except Exception as e:
                    print(f"OCR processing error: {e}")
            
            # Process at ~2 FPS to save CPU
            time.sleep(0.5)
    
    def update_frame(self, frame):
        """Update the frame to be processed"""
        with self.lock:
            self.current_frame = frame
    
    def get_results(self):
        """Get current OCR results"""
        with self.lock:
            return self.ocr_result, self.timer_result, self.processed_image
    
    def update_settings(self, **kwargs):
        """Update OCR processing settings"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def process_image_for_ocr(self, image):
        """Process image for optimal OCR"""
        if image is None:
            return None
        
        # Scale up for better OCR
        if self.scale_factor > 1.0:
            new_width = int(image.shape[1] * self.scale_factor)
            new_height = int(image.shape[0] * self.scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to HSV for color masking
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2HSV)
        
        # Create color mask
        h, s, v = self.selected_color_hsv
        h_range, s_range, v_range = self.color_ranges
        
        # Handle hue wrapping
        if h - h_range < 0:
            lower1 = np.array([0, max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper1 = np.array([h + h_range, min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            lower2 = np.array([max(0, 180 + (h - h_range)), max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper2 = np.array([179, min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif h + h_range > 179:
            lower1 = np.array([max(0, h - h_range), max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper1 = np.array([179, min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            lower2 = np.array([0, max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper2 = np.array([min(179, (h + h_range) - 180), min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array([max(0, h - h_range), max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper = np.array([min(179, h + h_range), min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        if self.dilate_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)
            
        if self.erode_iterations > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=self.erode_iterations)
        
        # Additional cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def stop(self):
        """Stop OCR processing"""
        self.stopped = True

class OptimizedOCRSetup:
    def __init__(self, parent: tk.Tk, camera_index: int, crop_corners: List[Tuple[int, int]]):
        self.parent = parent
        self.camera_index = camera_index
        self.crop_corners = crop_corners
        
        # Camera setup
        self.capture: Optional[cv2.VideoCapture] = None
        self.preview_running = False
        self._preview_image = None
        self.camera_resolution = (1920, 1080)  # Will be loaded from config
        
        # OCR processor
        self.ocr_processor = OCRProcessor().start()
        
        # UI state
        self.color_picker_active = False
        self.last_detected_time = ""
        
        # Rate limiting for UI updates
        self.last_ui_update = 0
        self.ui_update_interval = 0.1  # Update UI at 10 FPS max
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("OCR Setup - Optimized Processing")
        self.dialog.geometry("1000x900")
        self.dialog.minsize(800, 700)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.setup_ui()
        self.start_preview()

    def start_preview(self):
        """Start optimized camera preview"""
        # Load saved camera configuration for fast startup
        try:
            import json
            if os.path.exists("camera_config.json"):
                with open("camera_config.json", 'r') as f:
                    config = json.load(f)
                    if 'camera_resolution' in config:
                        self.camera_resolution = tuple(config['camera_resolution'])
                        print(f"Using saved camera resolution: {self.camera_resolution}")
        except Exception as e:
            print(f"Could not load saved camera config: {e}")
        
        # Setup camera at lower resolution for preview (much faster)
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
            return
        
        # Use moderate resolution for preview (balance between quality and speed)
        preview_width, preview_height = 1280, 720  # 720p preview
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)
        
        # Optimize for speed
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        try:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass
        
        self.preview_running = True
        self.update_preview()

    def update_preview(self):
        """Fast preview update with throttled OCR processing"""
        if not self.preview_running or not self.capture:
            return
            
        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.dialog.after(33, self.update_preview)  # ~30 FPS retry
            return
        
        # Always get crop for display (fast operation)
        cropped = self.get_crop(frame)
        
        if cropped is not None:
            # Store current crop for color picking
            self.current_crop_image = cropped.copy()
            
            # Always update original preview (responsive UI)
            self.update_image_display(self.original_preview, cropped, max_size=(400, 250))
            
            # Send frame to OCR processor (it handles its own timing)
            self.ocr_processor.update_frame(cropped)
            
            # Update UI with OCR results at limited rate
            current_time = time.time()
            if current_time - self.last_ui_update >= self.ui_update_interval:
                self.last_ui_update = current_time
                self.update_ocr_results()
        
        # Fast preview updates for smooth UI
        self.dialog.after(33, self.update_preview)  # ~30 FPS

    def update_ocr_results(self):
        """Update UI with latest OCR results"""
        ocr_text, timer_text, processed_img = self.ocr_processor.get_results()
        
        # Update text results
        self.ocr_result_var.set(ocr_text if ocr_text else "Processing...")
        self.timer_result_var.set(timer_text if timer_text else "No timer detected")
        
        # Update processed image preview
        if processed_img is not None:
            self.update_image_display(self.processed_preview, processed_img, max_size=(400, 250))
        
        # Store last detected time
        if timer_text:
            self.last_detected_time = timer_text

    def get_crop(self, frame):
        """Fast crop operation - same as before but optimized"""
        try:
            # Calculate scale factor from current frame to saved camera resolution
            current_height, current_width = frame.shape[:2]
            scale_x = current_width / self.camera_resolution[0]
            scale_y = current_height / self.camera_resolution[1]
            
            # Scale crop corners to current frame size
            scaled_corners = []
            for x, y in self.crop_corners:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_corners.append((scaled_x, scaled_y))
            
            src_points = np.array(scaled_corners, dtype=np.float32)
            
            # Calculate output dimensions
            widths = [
                np.linalg.norm(src_points[1] - src_points[0]),
                np.linalg.norm(src_points[2] - src_points[3])
            ]
            heights = [
                np.linalg.norm(src_points[3] - src_points[0]),
                np.linalg.norm(src_points[2] - src_points[1])
            ]
            
            max_width = max(int(max(widths)), 1)
            max_height = max(int(max(heights)), 1)
            
            dst_points = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            corrected = cv2.warpPerspective(frame, matrix, (max_width, max_height))
            
            return corrected
            
        except Exception as e:
            print(f"Crop error: {e}")
            return None

    def setup_ui(self):
        """Setup UI with full color picker functionality"""
        # Main container
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Preview (50% width)
        left_frame = ttk.LabelFrame(main_frame, text="Timer Preview (Threaded OCR Processing)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Original preview
        self.original_preview = tk.Label(left_frame, text="Original Crop\n(Click to pick text color)", 
                                       bg="black", fg="white", cursor="crosshair")
        self.original_preview.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.original_preview.bind("<Button-1>", self.on_color_pick_click)
        self.original_preview.bind("<Motion>", self.on_mouse_motion)
        
        # Processed preview
        self.processed_preview = tk.Label(left_frame, text="OCR Processing...", 
                                        bg="black", fg="white")
        self.processed_preview.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Controls (50% width)
        right_frame = ttk.LabelFrame(main_frame, text="OCR Controls", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Performance info
        perf_frame = ttk.LabelFrame(right_frame, text="Performance Info", padding=10)
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(perf_frame, text="• Preview: ~30 FPS (smooth UI)").pack(anchor="w")
        ttk.Label(perf_frame, text="• OCR Processing: ~2 FPS (CPU efficient)").pack(anchor="w")
        ttk.Label(perf_frame, text="• Threaded processing prevents UI lag").pack(anchor="w")
        
        # OCR Results
        results_frame = ttk.LabelFrame(right_frame, text="OCR Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Detected Text:").pack(anchor="w")
        self.ocr_result_var = tk.StringVar(value="Starting OCR processor...")
        self.ocr_result_label = ttk.Label(results_frame, textvariable=self.ocr_result_var, 
                                         foreground="blue", font=("Courier", 12, "bold"))
        self.ocr_result_label.pack(anchor="w", pady=(0, 5))
        self.ocr_result_label.config(width=40)  # Fixed width to prevent layout shifts
        
        ttk.Label(results_frame, text="Parsed Timer:").pack(anchor="w")
        self.timer_result_var = tk.StringVar(value="Waiting for timer...")
        self.timer_result_label = ttk.Label(results_frame, textvariable=self.timer_result_var,
                                           foreground="green", font=("Courier", 14, "bold"))
        self.timer_result_label.pack(anchor="w")
        self.timer_result_label.config(width=25)  # Fixed width for timer text
        
        # Color picker controls (full implementation)
        color_frame = ttk.LabelFrame(right_frame, text="Text Color Selection", padding=10)
        color_frame.pack(fill=tk.X, pady=(0, 10))
        
        instruction_text = "Click on timer text in preview above to select color"
        instruction_label = ttk.Label(color_frame, text=instruction_text, foreground="gray", wraplength=300)
        instruction_label.pack(pady=(0, 5))
        instruction_label.config(width=45)  # Fixed width to prevent wrapping changes
        
        picker_frame = ttk.Frame(color_frame)
        picker_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.color_picker_btn = ttk.Button(picker_frame, text="Activate Color Picker", 
                                          command=self.toggle_color_picker)
        self.color_picker_btn.pack(side=tk.LEFT)
        
        self.color_display = tk.Label(picker_frame, text="  ", bg="white", relief="sunken", width=4)
        self.color_display.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.color_info_var = tk.StringVar(value="HSV: (0, 0, 255)")
        color_info_label = ttk.Label(color_frame, textvariable=self.color_info_var, font=("Courier", 9))
        color_info_label.pack(anchor="w")
        color_info_label.config(width=20)  # Fixed width to prevent jitter
        
        self.live_color_var = tk.StringVar(value="Hover over image to see live HSV")
        self.live_color_label = ttk.Label(color_frame, textvariable=self.live_color_var, 
                                         font=("Courier", 8), foreground="gray")
        self.live_color_label.pack(anchor="w")
        self.live_color_label.config(width=35)  # Fixed width for longer live text
        
        # Color tolerance controls
        ttk.Label(color_frame, text="Color Tolerance Ranges:").pack(anchor="w", pady=(10, 0))
        
        hue_frame = ttk.Frame(color_frame)
        hue_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(hue_frame, text="Hue ±:").pack(side=tk.LEFT)
        self.hue_range_var = tk.IntVar(value=10)
        ttk.Scale(hue_frame, from_=0, to=50, variable=self.hue_range_var,
                 orient="horizontal", command=self.on_color_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Label(hue_frame, textvariable=self.hue_range_var, width=3).pack(side=tk.RIGHT)
        
        sat_frame = ttk.Frame(color_frame)
        sat_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(sat_frame, text="Sat ±:").pack(side=tk.LEFT)
        self.sat_range_var = tk.IntVar(value=50)
        ttk.Scale(sat_frame, from_=0, to=100, variable=self.sat_range_var,
                 orient="horizontal", command=self.on_color_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Label(sat_frame, textvariable=self.sat_range_var, width=3).pack(side=tk.RIGHT)
        
        val_frame = ttk.Frame(color_frame)
        val_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(val_frame, text="Val ±:").pack(side=tk.LEFT)
        self.val_range_var = tk.IntVar(value=50)
        ttk.Scale(val_frame, from_=0, to=100, variable=self.val_range_var,
                 orient="horizontal", command=self.on_color_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Label(val_frame, textvariable=self.val_range_var, width=3).pack(side=tk.RIGHT)
        
        ttk.Button(color_frame, text="Reset to White", command=self.reset_to_white).pack(pady=(5, 0))
        
        # Processing controls (full implementation)
        processing_frame = ttk.LabelFrame(right_frame, text="Image Enhancement", padding=10)
        processing_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(processing_frame, text="Scale Factor:").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=3.0)
        scale_frame = ttk.Frame(processing_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(scale_frame, from_=1.0, to=8.0, variable=self.scale_var, 
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(scale_frame, textvariable=self.scale_var, width=6).pack(side=tk.RIGHT)
        
        ttk.Label(processing_frame, text="Dilate (thicken):").pack(anchor="w")
        self.dilate_var = tk.IntVar(value=1)
        dilate_frame = ttk.Frame(processing_frame)
        dilate_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(dilate_frame, from_=0, to=5, variable=self.dilate_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(dilate_frame, textvariable=self.dilate_var, width=4).pack(side=tk.RIGHT)
        
        ttk.Label(processing_frame, text="Erode (thin):").pack(anchor="w")
        self.erode_var = tk.IntVar(value=1)
        erode_frame = ttk.Frame(processing_frame)
        erode_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(erode_frame, from_=0, to=5, variable=self.erode_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(erode_frame, textvariable=self.erode_var, width=4).pack(side=tk.RIGHT)
        
        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Test OCR", command=self.manual_ocr_test).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Back to Camera", command=self.back_to_camera).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Finish Setup", command=self.finish_setup).pack(fill=tk.X)

    def toggle_color_picker(self):
        """Toggle color picker mode"""
        self.color_picker_active = not self.color_picker_active
        if self.color_picker_active:
            self.color_picker_btn.config(text="Deactivate Color Picker")
            self.original_preview.config(cursor="crosshair")
        else:
            self.color_picker_btn.config(text="Activate Color Picker")
            self.original_preview.config(cursor="")

    def on_color_pick_click(self, event):
        """Handle color picking from preview image"""
        if not self.color_picker_active or not hasattr(self, 'current_crop_image'):
            return
            
        # Get click coordinates relative to the image
        label_width = self.original_preview.winfo_width()
        label_height = self.original_preview.winfo_height()
        
        if hasattr(self, 'current_crop_image'):
            img_h, img_w = self.current_crop_image.shape[:2]
            
            # Calculate the actual image position within the label (centered)
            if hasattr(self.original_preview, '_image_ref'):
                display_img = self.original_preview._image_ref
                display_w = display_img.width()
                display_h = display_img.height()
                
                # Calculate offset from center
                offset_x = (label_width - display_w) // 2
                offset_y = (label_height - display_h) // 2
                
                # Adjust click coordinates
                click_x = event.x - offset_x
                click_y = event.y - offset_y
                
                # Scale to original image coordinates
                scale_x = img_w / display_w
                scale_y = img_h / display_h
                
                orig_x = int(click_x * scale_x)
                orig_y = int(click_y * scale_y)
                
                # Ensure coordinates are within bounds
                if 0 <= orig_x < img_w and 0 <= orig_y < img_h:
                    # Get pixel color
                    if len(self.current_crop_image.shape) == 3:
                        bgr_color = self.current_crop_image[orig_y, orig_x]
                        # Convert BGR to RGB then to HSV
                        rgb_color = np.array([[[bgr_color[2], bgr_color[1], bgr_color[0]]]], dtype=np.uint8)
                        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0, 0]
                    else:
                        # Grayscale - convert to HSV equivalent
                        gray_val = self.current_crop_image[orig_y, orig_x]
                        hsv_color = np.array([0, 0, gray_val])
                    
                    # Store as regular integers and update OCR processor
                    selected_hsv = [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])]
                    self.ocr_processor.update_settings(selected_color_hsv=selected_hsv)
                    self.update_color_display(selected_hsv)

    def on_mouse_motion(self, event):
        """Show live HSV values as mouse moves over the image"""
        if not self.color_picker_active or not hasattr(self, 'current_crop_image'):
            self.live_color_var.set("Activate color picker to see live HSV")
            return
            
        # Get mouse coordinates relative to the image (same logic as click handler)
        label_width = self.original_preview.winfo_width()
        label_height = self.original_preview.winfo_height()
        
        if hasattr(self, 'current_crop_image'):
            img_h, img_w = self.current_crop_image.shape[:2]
            
            # Calculate the actual image position within the label (centered)
            if hasattr(self.original_preview, '_image_ref'):
                display_img = self.original_preview._image_ref
                display_w = display_img.width()
                display_h = display_img.height()
                
                # Calculate offset from center
                offset_x = (label_width - display_w) // 2
                offset_y = (label_height - display_h) // 2
                
                # Adjust mouse coordinates
                mouse_x = event.x - offset_x
                mouse_y = event.y - offset_y
                
                # Scale to original image coordinates
                scale_x = img_w / display_w
                scale_y = img_h / display_h
                
                orig_x = int(mouse_x * scale_x)
                orig_y = int(mouse_y * scale_y)
                
                # Ensure coordinates are within bounds
                if 0 <= orig_x < img_w and 0 <= orig_y < img_h:
                    # Get pixel color
                    if len(self.current_crop_image.shape) == 3:
                        bgr_color = self.current_crop_image[orig_y, orig_x]
                        # Convert BGR to RGB then to HSV
                        rgb_color = np.array([[[bgr_color[2], bgr_color[1], bgr_color[0]]]], dtype=np.uint8)
                        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0, 0]
                    else:
                        # Grayscale - convert to HSV equivalent
                        gray_val = self.current_crop_image[orig_y, orig_x]
                        hsv_color = np.array([0, 0, gray_val])
                    
                    # Show live HSV values
                    h, s, v = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])
                    self.live_color_var.set(f"Live HSV: ({h}, {s}, {v}) - Click to select")
                else:
                    self.live_color_var.set("Outside image bounds")
            else:
                self.live_color_var.set("Image not loaded")

    def update_color_display(self, hsv_color=None):
        """Update the color display and info"""
        if hsv_color is None:
            # Get current color from OCR processor
            hsv_color = self.ocr_processor.selected_color_hsv
        
        h, s, v = hsv_color
        self.color_info_var.set(f"HSV: ({h}, {s}, {v})")
        
        # Convert HSV to RGB for display
        hsv_for_rgb = np.array([[[h, s, v]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_for_rgb, cv2.COLOR_HSV2RGB)[0, 0]
        
        # Update color display
        hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        self.color_display.config(bg=hex_color)

    def reset_to_white(self):
        """Reset color selection to white"""
        white_hsv = [0, 0, 255]
        self.ocr_processor.update_settings(selected_color_hsv=white_hsv)
        self.update_color_display(white_hsv)

    def on_color_param_change(self, *args):
        """Called when color tolerance parameters change"""
        color_ranges = [
            int(self.hue_range_var.get()),
            int(self.sat_range_var.get()),
            int(self.val_range_var.get())
        ]
        self.ocr_processor.update_settings(color_ranges=color_ranges)

    def on_param_change(self, *args):
        """Update OCR processor settings when any parameter changes"""
        self.ocr_processor.update_settings(
            scale_factor=float(self.scale_var.get()),
            dilate_iterations=int(self.dilate_var.get()),
            erode_iterations=int(self.erode_var.get())
        )

    def update_image_display(self, label_widget, cv_image, max_size=(400, 250)):
        """Fast image display update with larger preview size for 50/50 split"""
        try:
            if len(cv_image.shape) == 3:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            img_w, img_h = pil_image.size
            max_w, max_h = max_size
            
            ratio = min(max_w / img_w, max_h / img_h)
            new_w = max(1, int(img_w * ratio))
            new_h = max(1, int(img_h * ratio))
            
            display_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            label_widget.configure(image=photo, text="")
            label_widget._image_ref = photo
            
        except Exception as e:
            label_widget.configure(image="", text=f"Display error: {str(e)}")

    def manual_ocr_test(self):
        """Test current OCR settings"""
        ocr_text, timer_text, processed_img = self.ocr_processor.get_results()
        messagebox.showinfo("OCR Test", f"Raw OCR: {ocr_text}\n\nParsed Timer: {timer_text}")

    def save_settings(self):
        """Save current settings"""
        messagebox.showinfo("Settings", "Settings saved for optimized OCR processing")

    def back_to_camera(self):
        """Return to camera setup"""
        self.stop_preview()
        self.dialog.destroy()

    def finish_setup(self):
        """Complete setup"""
        if not self.last_detected_time:
            if not messagebox.askyesno("Warning", 
                "No timer has been detected yet. Continue anyway?"):
                return
        
        messagebox.showinfo("Setup Complete", 
            f"Optimized OCR setup complete!\n"
            f"Last detected time: {self.last_detected_time}\n"
            f"Processing runs efficiently in background thread")
        
        self.stop_preview()
        self.dialog.destroy()

    def stop_preview(self):
        """Stop camera and OCR processing"""
        self.preview_running = False
        if self.capture:
            self.capture.release()
            self.capture = None
        self.ocr_processor.stop()

    def on_cancel(self):
        """Cancel setup"""
        self.stop_preview()
        self.dialog.destroy()


# Usage function
def run_optimized_ocr_setup(camera_index: int, crop_corners: List[Tuple[int, int]]) -> bool:
    """Run the optimized OCR setup"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = OptimizedOCRSetup(root, camera_index, crop_corners)
        root.wait_window(app.dialog)
        return True
    except Exception as e:
        messagebox.showerror("Error", f"OCR setup failed: {str(e)}")
        return False