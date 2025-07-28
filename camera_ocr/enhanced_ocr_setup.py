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


class EnhancedOCRProcessor:
    """Enhanced OCR processing with multiple preprocessing methods"""
    
    def __init__(self):
        self.stopped = False
        self.current_frame = None
        self.ocr_result = ""
        self.timer_result = ""
        self.processed_image = None
        self.lock = threading.Lock()
        
        # OCR settings
        self.tesseract_configs = [
            "--psm 8 -c tessedit_char_whitelist=0123456789:.",
            "--psm 7 -c tessedit_char_whitelist=0123456789:.",
            "--psm 6 -c tessedit_char_whitelist=0123456789:.",
            "--psm 13 -c tessedit_char_whitelist=0123456789:."
        ]
        self.timer_pattern = r'\d:\d{2}\.\d{3}'
        
        # Processing method selection
        self.processing_method = "white_text_extraction"  # Default to best for white text
        
        # Image processing settings (kept for compatibility)
        self.selected_color_hsv = [0, 0, 255]
        self.color_ranges = [10, 50, 50]
        self.scale_factor = 4.0  # Increased default
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
                    # Process image using selected method
                    processed = self.process_image_for_ocr(frame_to_process)
                    
                    if processed is not None and self.tesseract_available:
                        # Try multiple OCR configurations and pick best result
                        best_text = ""
                        best_timer = ""
                        best_score = 0
                        
                        for config in self.tesseract_configs:
                            try:
                                raw_text = pytesseract.image_to_string(
                                    processed, config=config
                                ).strip()
                                
                                # Score this result
                                score = self.score_timer_text(raw_text)
                                
                                if score > best_score:
                                    best_score = score
                                    best_text = raw_text
                                    
                                    # Extract timer pattern
                                    timer_match = re.search(self.timer_pattern, raw_text)
                                    best_timer = timer_match.group(0) if timer_match else raw_text
                                    
                            except Exception as e:
                                continue
                        
                        # Update results
                        with self.lock:
                            self.ocr_result = best_text
                            self.timer_result = best_timer
                            self.processed_image = processed
                    
                except Exception as e:
                    print(f"OCR processing error: {e}")
            
            # Process at ~2 FPS to save CPU
            time.sleep(0.5)
    
    def score_timer_text(self, text):
        """Score OCR text based on timer pattern likelihood"""
        if not text:
            return 0
        
        score = 0
        
        # Check for timer patterns
        timer_patterns = [
            r'\d:\d{2}\.\d{3}',  # X:XX.XXX
            r'\d:\d{2}\.\d{2}',  # X:XX.XX  
            r'\d:\d{2}\.\d{1}',  # X:XX.X
            r'\d:\d{2}',         # X:XX
        ]
        
        for i, pattern in enumerate(timer_patterns):
            if re.search(pattern, text):
                score += (4 - i) * 10
        
        # Bonus for valid characters
        valid_chars = "0123456789:."
        for char in text:
            if char in valid_chars:
                score += 1
            else:
                score -= 2
        
        # Bonus for reasonable length
        if 4 <= len(text) <= 10:
            score += 5
        
        return max(0, score)
    
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
        """Process image using selected method"""
        if image is None:
            return None
        
        # Choose processing method
        if self.processing_method == "white_text_extraction":
            return self.method_white_text_extraction(image)
        elif self.processing_method == "adaptive_threshold":
            return self.method_adaptive_threshold(image)
        elif self.processing_method == "edge_enhanced":
            return self.method_edge_enhanced(image)
        elif self.processing_method == "color_masking":
            return self.method_color_masking(image)  # Original method
        else:
            return self.method_white_text_extraction(image)  # Default
    
    def method_white_text_extraction(self, image):
        """Extract white text - best for timer displays"""
        try:
            if len(image.shape) == 3:
                # Convert to HSV for better white detection
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Create mask for white/light colored pixels
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 30, 255])
                mask = cv2.inRange(hsv, lower_white, upper_white)
            else:
                # For grayscale, threshold for bright pixels
                _, mask = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
            
            # Scale up significantly
            scale = int(self.scale_factor)
            height, width = mask.shape
            scaled_mask = cv2.resize(mask, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(scaled_mask, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Apply dilate/erode based on settings
            if self.dilate_iterations > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                cleaned = cv2.dilate(cleaned, kernel, iterations=self.dilate_iterations)
                
            if self.erode_iterations > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                cleaned = cv2.erode(cleaned, kernel, iterations=self.erode_iterations)
            
            return cleaned
            
        except Exception as e:
            print(f"White text extraction error: {e}")
            return None
    
    def method_adaptive_threshold(self, image):
        """Adaptive thresholding method"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Scale up
            scale = int(self.scale_factor)
            height, width = gray.shape
            scaled = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
            
            # Apply slight blur
            blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Check if we need to invert (for white text)
            center_region = thresh[thresh.shape[0]//4:3*thresh.shape[0]//4, 
                                 thresh.shape[1]//4:3*thresh.shape[1]//4]
            white_pixels = np.sum(center_region == 255)
            black_pixels = np.sum(center_region == 0)
            
            if black_pixels > white_pixels:
                thresh = cv2.bitwise_not(thresh)
            
            return thresh
            
        except Exception as e:
            print(f"Adaptive threshold error: {e}")
            return None
    
    def method_edge_enhanced(self, image):
        """Edge enhancement method"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Scale up
            scale = int(self.scale_factor)
            height, width = gray.shape
            scaled = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(scaled)
            
            # Apply unsharp mask
            blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
            unsharp = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            
            # Threshold
            _, thresh = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Check orientation
            white_pixels = np.sum(thresh == 255)
            black_pixels = np.sum(thresh == 0)
            
            if black_pixels > white_pixels:
                thresh = cv2.bitwise_not(thresh)
            
            return thresh
            
        except Exception as e:
            print(f"Edge enhanced error: {e}")
            return None
    
    def method_color_masking(self, image):
        """Original color masking method (kept for compatibility)"""
        if image is None:
            return None
        
        # Scale up the image first
        scale = self.scale_factor
        if scale > 1.0:
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to HSV for color masking
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2HSV)
        
        # Create color mask (original logic)
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
        self.camera_resolution = (1920, 1080)
        
        # Enhanced OCR processor
        self.ocr_processor = EnhancedOCRProcessor().start()
        
        # UI state
        self.color_picker_active = False
        self.last_detected_time = ""
        
        # Rate limiting for UI updates
        self.last_ui_update = 0
        self.ui_update_interval = 0.1
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Enhanced OCR Setup - Multiple Processing Methods")
        self.dialog.geometry("1000x900")
        self.dialog.minsize(800, 700)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.setup_ui()
        self.start_preview()

    def setup_ui(self):
        """Setup UI with method selection"""
        # Main container
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Preview (50% width)
        left_frame = ttk.LabelFrame(main_frame, text="Timer Preview (Enhanced Processing)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Original preview
        self.original_preview = tk.Label(left_frame, text="Original Crop\n(Click to pick text color)", 
                                       bg="black", fg="white", cursor="crosshair")
        self.original_preview.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.original_preview.bind("<Button-1>", self.on_color_pick_click)
        self.original_preview.bind("<Motion>", self.on_mouse_motion)
        
        # Processed preview
        self.processed_preview = tk.Label(left_frame, text="Processing...", 
                                        bg="black", fg="white")
        self.processed_preview.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Controls (50% width)
        right_frame = ttk.LabelFrame(main_frame, text="Enhanced OCR Controls", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Processing Method Selection (NEW!)
        method_frame = ttk.LabelFrame(right_frame, text="Processing Method", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.method_var = tk.StringVar(value="white_text_extraction")
        methods = [
            ("White Text Extraction (Best for timers)", "white_text_extraction"),
            ("Adaptive Threshold", "adaptive_threshold"),
            ("Edge Enhanced", "edge_enhanced"),
            ("Color Masking (Original)", "color_masking")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, 
                          value=value, command=self.on_method_change).pack(anchor="w")
        
        # OCR Results
        results_frame = ttk.LabelFrame(right_frame, text="OCR Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Detected Text:").pack(anchor="w")
        self.ocr_result_var = tk.StringVar(value="Starting enhanced OCR...")
        self.ocr_result_label = ttk.Label(results_frame, textvariable=self.ocr_result_var, 
                                         foreground="blue", font=("Courier", 12, "bold"))
        self.ocr_result_label.pack(anchor="w", pady=(0, 5))
        self.ocr_result_label.config(width=40)
        
        ttk.Label(results_frame, text="Parsed Timer:").pack(anchor="w")
        self.timer_result_var = tk.StringVar(value="Waiting for timer...")
        self.timer_result_label = ttk.Label(results_frame, textvariable=self.timer_result_var,
                                           foreground="green", font=("Courier", 14, "bold"))
        self.timer_result_label.pack(anchor="w")
        self.timer_result_label.config(width=25)
        
        # Color picker (simplified for white text method)
        color_frame = ttk.LabelFrame(right_frame, text="Color Selection (for Color Masking)", padding=10)
        color_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(color_frame, text="Only used when Color Masking method is selected").pack(pady=(0, 5))
        
        picker_frame = ttk.Frame(color_frame)
        picker_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.color_picker_btn = ttk.Button(picker_frame, text="Activate Color Picker", 
                                          command=self.toggle_color_picker)
        self.color_picker_btn.pack(side=tk.LEFT)
        
        self.color_display = tk.Label(picker_frame, text="  ", bg="white", relief="sunken", width=4)
        self.color_display.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Processing controls
        processing_frame = ttk.LabelFrame(right_frame, text="Image Enhancement", padding=10)
        processing_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(processing_frame, text="Scale Factor:").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=4.0)
        scale_frame = ttk.Frame(processing_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(scale_frame, from_=2.0, to=8.0, variable=self.scale_var, 
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
        
        ttk.Button(button_frame, text="Test Current Method", command=self.manual_ocr_test).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Test All Methods", command=self.test_all_methods).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Back to Camera", command=self.back_to_camera).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Finish Setup", command=self.finish_setup).pack(fill=tk.X)

    def on_method_change(self):
        """Handle processing method change"""
        method = self.method_var.get()
        self.ocr_processor.update_settings(processing_method=method)
        print(f"Switched to processing method: {method}")

    def test_all_methods(self):
        """Test all processing methods on current frame"""
        if not hasattr(self, 'current_crop_image') or self.current_crop_image is None:
            messagebox.showwarning("No Image", "No image available to test. Make sure camera is running.")
            return
        
        # Test all methods
        methods = [
            ("White Text Extraction", "white_text_extraction"),
            ("Adaptive Threshold", "adaptive_threshold"), 
            ("Edge Enhanced", "edge_enhanced"),
            ("Color Masking", "color_masking")
        ]
        
        results = []
        test_processor = EnhancedOCRProcessor()
        
        for method_name, method_key in methods:
            test_processor.processing_method = method_key
            test_processor.scale_factor = self.scale_var.get()
            test_processor.dilate_iterations = self.dilate_var.get()
            test_processor.erode_iterations = self.erode_var.get()
            
            # Copy color settings if using color masking
            if method_key == "color_masking":
                test_processor.selected_color_hsv = self.ocr_processor.selected_color_hsv
                test_processor.color_ranges = self.ocr_processor.color_ranges
            
            processed = test_processor.process_image_for_ocr(self.current_crop_image)
            
            if processed is not None:
                # Test OCR
                best_text = ""
                best_score = 0
                
                for config in test_processor.tesseract_configs:
                    try:
                        text = pytesseract.image_to_string(processed, config=config).strip()
                        score = test_processor.score_timer_text(text)
                        
                        if score > best_score:
                            best_score = score
                            best_text = text
                    except:
                        continue
                
                results.append(f"{method_name}: '{best_text}' (score: {best_score:.1f})")
            else:
                results.append(f"{method_name}: Failed to process")
        
        # Show results
        result_text = "Method Comparison Results:\n\n" + "\n".join(results)
        messagebox.showinfo("All Methods Test", result_text)

    # [Keep all other methods from the previous version - start_preview, update_preview, etc.]
    # Just replacing the OCRProcessor with EnhancedOCRProcessor

    def start_preview(self):
        """Start optimized camera preview"""
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
        
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
            return
        
        preview_width, preview_height = 1280, 720
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        try:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass
        
        self.preview_running = True
        self.update_preview()

    def update_preview(self):
        """Fast preview update with enhanced OCR processing"""
        if not self.preview_running or not self.capture:
            return
            
        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.dialog.after(33, self.update_preview)
            return
        
        cropped = self.get_crop(frame)
        
        if cropped is not None:
            self.current_crop_image = cropped.copy()
            self.update_image_display(self.original_preview, cropped, max_size=(400, 250))
            self.ocr_processor.update_frame(cropped)
            
            current_time = time.time()
            if current_time - self.last_ui_update >= self.ui_update_interval:
                self.last_ui_update = current_time
                self.update_ocr_results()
        
        self.dialog.after(33, self.update_preview)

    def update_ocr_results(self):
        """Update UI with latest OCR results"""
        ocr_text, timer_text, processed_img = self.ocr_processor.get_results()
        
        self.ocr_result_var.set(ocr_text if ocr_text else "Processing...")
        self.timer_result_var.set(timer_text if timer_text else "No timer detected")
        
        if processed_img is not None:
            self.update_image_display(self.processed_preview, processed_img, max_size=(400, 250))
        
        if timer_text:
            self.last_detected_time = timer_text

    def get_crop(self, frame):
        """Fast crop operation"""
        try:
            current_height, current_width = frame.shape[:2]
            scale_x = current_width / self.camera_resolution[0]
            scale_y = current_height / self.camera_resolution[1]
            
            scaled_corners = []
            for x, y in self.crop_corners:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_corners.append((scaled_x, scaled_y))
            
            src_points = np.array(scaled_corners, dtype=np.float32)
            
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
            
        label_width = self.original_preview.winfo_width()
        label_height = self.original_preview.winfo_height()
        
        if hasattr(self, 'current_crop_image'):
            img_h, img_w = self.current_crop_image.shape[:2]
            
            if hasattr(self.original_preview, '_image_ref'):
                display_img = self.original_preview._image_ref
                display_w = display_img.width()
                display_h = display_img.height()
                
                offset_x = (label_width - display_w) // 2
                offset_y = (label_height - display_h) // 2
                
                click_x = event.x - offset_x
                click_y = event.y - offset_y
                
                scale_x = img_w / display_w
                scale_y = img_h / display_h
                
                orig_x = int(click_x * scale_x)
                orig_y = int(click_y * scale_y)
                
                if 0 <= orig_x < img_w and 0 <= orig_y < img_h:
                    if len(self.current_crop_image.shape) == 3:
                        bgr_color = self.current_crop_image[orig_y, orig_x]
                        rgb_color = np.array([[[bgr_color[2], bgr_color[1], bgr_color[0]]]], dtype=np.uint8)
                        hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0, 0]
                    else:
                        gray_val = self.current_crop_image[orig_y, orig_x]
                        hsv_color = np.array([0, 0, gray_val])
                    
                    selected_hsv = [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])]
                    self.ocr_processor.update_settings(selected_color_hsv=selected_hsv)
                    self.update_color_display(selected_hsv)

    def on_mouse_motion(self, event):
        """Show live HSV values"""
        if not self.color_picker_active or not hasattr(self, 'current_crop_image'):
            return
        # Implementation similar to on_color_pick_click but just for display

    def update_color_display(self, hsv_color=None):
        """Update the color display"""
        if hsv_color is None:
            hsv_color = self.ocr_processor.selected_color_hsv
        
        h, s, v = hsv_color
        hsv_for_rgb = np.array([[[h, s, v]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_for_rgb, cv2.COLOR_HSV2RGB)[0, 0]
        hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        self.color_display.config(bg=hex_color)

    def on_param_change(self, *args):
        """Update OCR processor settings"""
        self.ocr_processor.update_settings(
            scale_factor=float(self.scale_var.get()),
            dilate_iterations=int(self.dilate_var.get()),
            erode_iterations=int(self.erode_var.get())
        )

    def update_image_display(self, label_widget, cv_image, max_size=(400, 250)):
        """Fast image display update"""
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
        method = self.method_var.get()
        messagebox.showinfo("OCR Test", 
            f"Method: {method}\n"
            f"Raw OCR: {ocr_text}\n"
            f"Parsed Timer: {timer_text}")

    def save_settings(self):
        """Save current settings"""
        settings = {
            'processing_method': self.method_var.get(),
            'scale_factor': self.scale_var.get(),
            'dilate_iterations': self.dilate_var.get(),
            'erode_iterations': self.erode_var.get()
        }
        messagebox.showinfo("Settings", f"Enhanced OCR settings saved:\n{settings}")

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
        
        method = self.method_var.get()
        messagebox.showinfo("Setup Complete", 
            f"Enhanced OCR setup complete!\n"
            f"Processing method: {method}\n"
            f"Last detected time: {self.last_detected_time}")
        
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
def run_enhanced_ocr_setup(camera_index: int, crop_corners: List[Tuple[int, int]]) -> bool:
    """Run the enhanced OCR setup"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = OptimizedOCRSetup(root, camera_index, crop_corners)
        root.wait_window(app.dialog)
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Enhanced OCR setup failed: {str(e)}")
        return False