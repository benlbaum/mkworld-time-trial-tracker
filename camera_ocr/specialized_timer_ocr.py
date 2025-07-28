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


class SpecializedTimerOCRProcessor:
    """Highly specialized OCR processor for fixed-format timer text (X:XX.XXX)"""
    
    def __init__(self):
        self.stopped = False
        self.current_frame = None
        self.ocr_result = ""
        self.timer_result = ""
        self.processed_image = None
        self.lock = threading.Lock()
        
        # Specialized Tesseract configs for fixed timer format
        self.tesseract_configs = [
            # Most restrictive first - only timer characters, treat as single word
            "--psm 8 -c tessedit_char_whitelist=0123456789:. -c tessedit_enable_dict_correction=0",
            # Single line, no dictionary correction
            "--psm 7 -c tessedit_char_whitelist=0123456789:. -c tessedit_enable_dict_correction=0",
            # Raw line - minimal processing
            "--psm 13 -c tessedit_char_whitelist=0123456789:.",
            # Single uniform block
            "--psm 6 -c tessedit_char_whitelist=0123456789:.",
            # Fallback - digits only then add punctuation
            "--psm 8 -c tessedit_char_whitelist=0123456789"
        ]
        
        # Fixed timer pattern - exactly X:XX.XXX
        self.timer_pattern = re.compile(r'^(\d):(\d{2})\.(\d{3})$')
        
        # Processing settings optimized for this specific use case
        self.scale_factor = 6.0  # High scaling for small timer text
        self.white_threshold_low = 180   # Lower bound for white/yellow text
        self.white_threshold_high = 255  # Upper bound
        self.morphology_strength = 2     # Text cleanup strength
        
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
        """Main OCR processing loop"""
        while not self.stopped:
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
                else:
                    frame_to_process = None
            
            if frame_to_process is not None:
                try:
                    # Apply specialized timer processing
                    processed = self.process_timer_text(frame_to_process)
                    
                    if processed is not None and self.tesseract_available:
                        # Try OCR with multiple configs
                        best_timer = ""
                        best_confidence = 0
                        best_raw = ""
                        
                        for config in self.tesseract_configs:
                            try:
                                raw_text = pytesseract.image_to_string(
                                    processed, config=config
                                ).strip()
                                
                                # Clean and validate the result
                                cleaned = self.clean_timer_text(raw_text)
                                confidence = self.validate_timer_format(cleaned)
                                
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_timer = cleaned
                                    best_raw = raw_text
                                    
                            except Exception as e:
                                continue
                        
                        # Update results
                        with self.lock:
                            self.ocr_result = best_raw
                            self.timer_result = best_timer
                            self.processed_image = processed
                    
                except Exception as e:
                    print(f"Timer OCR processing error: {e}")
            
            # Process at ~3 FPS for responsive timer updates
            time.sleep(0.33)
    
    def process_timer_text(self, image):
        """Specialized processing for timer text on semi-transparent black background"""
        try:
            # Convert to different color spaces for robust white/yellow detection
            if len(image.shape) == 3:
                # Method 1: HSV for white/yellow detection
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # White text: low saturation, high value
                # Yellow text: hue around 60, high saturation and value
                white_mask = cv2.inRange(hsv, 
                    np.array([0, 0, self.white_threshold_low]), 
                    np.array([180, 30, 255]))
                
                yellow_mask = cv2.inRange(hsv, 
                    np.array([20, 100, self.white_threshold_low]), 
                    np.array([40, 255, 255]))
                
                # Combine white and yellow masks
                color_mask = cv2.bitwise_or(white_mask, yellow_mask)
                
                # Method 2: LAB color space (often better for white detection)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]  # Lightness channel
                
                # Threshold on lightness for bright text
                _, light_mask = cv2.threshold(l_channel, self.white_threshold_low, 255, cv2.THRESH_BINARY)
                
                # Combine both methods
                combined_mask = cv2.bitwise_or(color_mask, light_mask)
                
            else:
                # Grayscale - simple threshold for bright pixels
                _, combined_mask = cv2.threshold(image, self.white_threshold_low, 255, cv2.THRESH_BINARY)
            
            # Scale up significantly for better OCR (timer text is usually small)
            scale = int(self.scale_factor)
            height, width = combined_mask.shape
            scaled_mask = cv2.resize(combined_mask, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
            
            # Specialized cleanup for timer text
            # 1. Close small gaps in characters (common with ":" and ".")
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(scaled_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            # 2. Remove small noise that's not text
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            # 3. Slightly dilate to make characters more solid (helps with OCR)
            if self.morphology_strength > 0:
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=self.morphology_strength)
            
            # 4. Final cleanup - connect broken characters
            kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_final, iterations=1)
            
            return final
            
        except Exception as e:
            print(f"Timer text processing error: {e}")
            return None
    
    def clean_timer_text(self, raw_text):
        """Clean OCR text specifically for timer format (X:XX.XXX)"""
        if not raw_text:
            return ""
        
        # Remove all whitespace and unwanted characters
        cleaned = re.sub(r'[^\d:.]', '', raw_text)
        
        # Common OCR corrections for this specific font/use case
        corrections = {
            'O': '0', 'o': '0', 'Q': '0',  # Common misreads for 0
            'I': '1', 'l': '1', '|': '1', 'i': '1',  # Common misreads for 1
            'S': '5', 's': '5',  # Common misreads for 5
            'G': '6', 'g': '6',  # Common misreads for 6
            'T': '7', 't': '7',  # Common misreads for 7
            'B': '8', 'b': '8',  # Common misreads for 8
            'g': '9', 'q': '9',  # Common misreads for 9
        }
        
        for wrong, right in corrections.items():
            cleaned = cleaned.replace(wrong, right)
        
        # Try to reconstruct proper timer format
        # Extract all digits
        digits = re.findall(r'\d', cleaned)
        colons = cleaned.count(':')
        periods = cleaned.count('.')
        
        # If we have the right number of digits (7) and punctuation, try to format
        if len(digits) >= 6:  # At least 6 digits for X:XX.XXX
            if len(digits) == 6:
                # Format as X:XX.XXX with leading zero for milliseconds if needed
                result = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
            elif len(digits) == 7:
                # Most likely format: X:XX.XXX
                result = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
            else:
                # Try to use first 7 digits
                result = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
            
            return result
        
        # Fallback: return cleaned text as-is
        return cleaned
    
    def validate_timer_format(self, text):
        """Validate and score timer text against exact format X:XX.XXX"""
        if not text:
            return 0
        
        # Perfect match for timer format
        match = self.timer_pattern.match(text)
        if match:
            # Perfect format match
            score = 100
            
            # Additional validation - check if values make sense
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = int(match.group(3))
            
            # Reasonable timer values
            if 0 <= minutes <= 9 and 0 <= seconds <= 59 and 0 <= milliseconds <= 999:
                score += 20  # Bonus for reasonable values
            
            return score
        
        # Partial matches - try to be helpful
        score = 0
        
        # Count correct elements
        if re.search(r'\d:', text):  # Has minute:
            score += 25
        if re.search(r':\d{2}', text):  # Has :XX seconds
            score += 25
        if re.search(r'\.\d{1,3}', text):  # Has .XXX milliseconds
            score += 25
        
        # Length check
        if 7 <= len(text) <= 9:  # Expected length for X:XX.XXX
            score += 15
        
        # Only valid characters
        if re.match(r'^[\d:.]+$', text):
            score += 10
        
        return score
    
    def update_frame(self, frame):
        """Update the frame to be processed"""
        with self.lock:
            self.current_frame = frame
    
    def get_results(self):
        """Get current OCR results"""
        with self.lock:
            return self.ocr_result, self.timer_result, self.processed_image
    
    def update_settings(self, **kwargs):
        """Update processing settings"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def stop(self):
        """Stop OCR processing"""
        self.stopped = True


class SpecializedTimerOCRSetup:
    def __init__(self, parent: tk.Tk, camera_index: int, crop_corners: List[Tuple[int, int]]):
        self.parent = parent
        self.camera_index = camera_index
        self.crop_corners = crop_corners
        
        # Camera setup
        self.capture: Optional[cv2.VideoCapture] = None
        self.preview_running = False
        self._preview_image = None
        self.camera_resolution = (1920, 1080)
        
        # Specialized timer OCR processor
        self.ocr_processor = SpecializedTimerOCRProcessor().start()
        
        # UI state
        self.last_detected_time = ""
        
        # Rate limiting for UI updates
        self.last_ui_update = 0
        self.ui_update_interval = 0.1
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Specialized Timer OCR - Fixed Format X:XX.XXX")
        self.dialog.geometry("1000x800")
        self.dialog.minsize(800, 600)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.setup_ui()
        self.start_preview()

    def setup_ui(self):
        """Setup UI for specialized timer OCR"""
        # Main container
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Preview (50% width)
        left_frame = ttk.LabelFrame(main_frame, text="Timer Preview (Specialized for X:XX.XXX)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Original preview
        self.original_preview = tk.Label(left_frame, text="Original Timer Crop", 
                                       bg="black", fg="white")
        self.original_preview.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Processed preview
        self.processed_preview = tk.Label(left_frame, text="Processed for OCR\n(White/Yellow Text Extraction)", 
                                        bg="black", fg="white")
        self.processed_preview.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Controls (50% width)
        right_frame = ttk.LabelFrame(main_frame, text="Timer-Specific OCR Controls", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Timer-specific info
        info_frame = ttk.LabelFrame(right_frame, text="Timer Format Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        info_text = (
            "• Fixed Format: X:XX.XXX (e.g., 0:50.192)\n"
            "• White or flashing white/yellow text\n"
            "• Semi-transparent black background\n"
            "• Same font always\n"
            "• Optimized for this exact use case"
        )
        ttk.Label(info_frame, text=info_text, justify="left").pack(anchor="w")
        
        # OCR Results with confidence
        results_frame = ttk.LabelFrame(right_frame, text="Timer Recognition Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Raw OCR Output:").pack(anchor="w")
        self.ocr_result_var = tk.StringVar(value="Starting specialized timer OCR...")
        self.ocr_result_label = ttk.Label(results_frame, textvariable=self.ocr_result_var, 
                                         foreground="blue", font=("Courier", 10))
        self.ocr_result_label.pack(anchor="w", pady=(0, 8))
        
        ttk.Label(results_frame, text="Validated Timer:").pack(anchor="w")
        self.timer_result_var = tk.StringVar(value="Waiting for timer...")
        self.timer_result_label = ttk.Label(results_frame, textvariable=self.timer_result_var,
                                           foreground="green", font=("Courier", 18, "bold"))
        self.timer_result_label.pack(anchor="w", pady=(0, 8))
        
        # Validation status
        ttk.Label(results_frame, text="Format Validation:").pack(anchor="w")
        self.validation_var = tk.StringVar(value="No timer detected")
        self.validation_label = ttk.Label(results_frame, textvariable=self.validation_var,
                                         foreground="orange", font=("Arial", 10))
        self.validation_label.pack(anchor="w")
        
        # Processing Settings - Simplified for timer-specific use
        settings_frame = ttk.LabelFrame(right_frame, text="Processing Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scale factor
        ttk.Label(settings_frame, text="Image Scale (higher = more detail):").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=6.0)
        scale_frame = ttk.Frame(settings_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(scale_frame, from_=4.0, to=10.0, variable=self.scale_var, 
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(scale_frame, textvariable=self.scale_var, width=6).pack(side=tk.RIGHT)
        
        # White text threshold
        ttk.Label(settings_frame, text="White Text Threshold:").pack(anchor="w")
        self.threshold_var = tk.IntVar(value=180)
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(threshold_frame, from_=150, to=200, variable=self.threshold_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(threshold_frame, textvariable=self.threshold_var, width=6).pack(side=tk.RIGHT)
        
        # Text cleanup strength
        ttk.Label(settings_frame, text="Text Cleanup Strength:").pack(anchor="w")
        self.cleanup_var = tk.IntVar(value=2)
        cleanup_frame = ttk.Frame(settings_frame)
        cleanup_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(cleanup_frame, from_=0, to=5, variable=self.cleanup_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(cleanup_frame, textvariable=self.cleanup_var, width=6).pack(side=tk.RIGHT)
        
        # Quick presets for different conditions
        preset_frame = ttk.LabelFrame(right_frame, text="Quick Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.pack(fill=tk.X)
        
        ttk.Button(preset_btn_frame, text="Bright Timer", 
                  command=lambda: self.apply_preset("bright")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_btn_frame, text="Dim Timer", 
                  command=lambda: self.apply_preset("dim")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_btn_frame, text="Small Text", 
                  command=lambda: self.apply_preset("small")).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Test OCR", command=self.manual_ocr_test).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Back to Camera", command=self.back_to_camera).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Finish Setup", command=self.finish_setup).pack(fill=tk.X)

    def apply_preset(self, preset_name):
        """Apply presets optimized for different timer conditions"""
        presets = {
            "bright": {  # Bright, clear timer
                "scale_factor": 6.0,
                "white_threshold_low": 180,
                "morphology_strength": 2
            },
            "dim": {  # Dimmer timer or poor lighting
                "scale_factor": 7.0,
                "white_threshold_low": 160,
                "morphology_strength": 3
            },
            "small": {  # Very small timer text
                "scale_factor": 8.0,
                "white_threshold_low": 170,
                "morphology_strength": 2
            }
        }
        
        if preset_name in presets:
            settings = presets[preset_name]
            self.scale_var.set(settings["scale_factor"])
            self.threshold_var.set(settings["white_threshold_low"])
            self.cleanup_var.set(settings["morphology_strength"])
            self.on_param_change()
            print(f"Applied {preset_name} timer preset")

    def on_param_change(self, *args):
        """Update OCR processor settings"""
        self.ocr_processor.update_settings(
            scale_factor=float(self.scale_var.get()),
            white_threshold_low=int(self.threshold_var.get()),
            morphology_strength=int(self.cleanup_var.get())
        )

    def start_preview(self):
        """Start camera preview"""
        try:
            import json
            if os.path.exists("camera_config.json"):
                with open("camera_config.json", 'r') as f:
                    config = json.load(f)
                    if 'camera_resolution' in config:
                        self.camera_resolution = tuple(config['camera_resolution'])
        except Exception as e:
            print(f"Could not load camera config: {e}")
        
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
            return
        
        preview_width, preview_height = 1280, 720
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        self.preview_running = True
        self.update_preview()

    def update_preview(self):
        """Update preview with specialized timer processing"""
        if not self.preview_running or not self.capture:
            return
            
        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.dialog.after(33, self.update_preview)
            return
        
        cropped = self.get_crop(frame)
        
        if cropped is not None:
            self.current_crop_image = cropped.copy()
            self.update_image_display(self.original_preview, cropped, max_size=(400, 200))
            self.ocr_processor.update_frame(cropped)
            
            current_time = time.time()
            if current_time - self.last_ui_update >= self.ui_update_interval:
                self.last_ui_update = current_time
                self.update_ocr_results()
        
        self.dialog.after(33, self.update_preview)

    def update_ocr_results(self):
        """Update UI with OCR results and validation"""
        ocr_text, timer_text, processed_img = self.ocr_processor.get_results()
        
        self.ocr_result_var.set(ocr_text if ocr_text else "Processing...")
        self.timer_result_var.set(timer_text if timer_text else "No valid timer")
        
        # Update validation status
        if timer_text:
            confidence = self.ocr_processor.validate_timer_format(timer_text)
            if confidence >= 100:
                self.validation_var.set("✓ Perfect timer format")
                self.validation_label.config(foreground="green")
            elif confidence >= 50:
                self.validation_var.set("⚠ Partial timer format")
                self.validation_label.config(foreground="orange")
            else:
                self.validation_var.set("✗ Invalid format")
                self.validation_label.config(foreground="red")
        else:
            self.validation_var.set("No timer detected")
            self.validation_label.config(foreground="gray")
        
        if processed_img is not None:
            self.update_image_display(self.processed_preview, processed_img, max_size=(400, 200))
        
        if timer_text and self.ocr_processor.validate_timer_format(timer_text) >= 100:
            self.last_detected_time = timer_text

    def get_crop(self, frame):
        """Get cropped timer area"""
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

    def update_image_display(self, label_widget, cv_image, max_size=(400, 200)):
        """Update image display"""
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
            label_widget.configure(image="", text=f"Error: {str(e)}")

    def manual_ocr_test(self):
        """Test current OCR settings"""
        ocr_text, timer_text, processed_img = self.ocr_processor.get_results()
        confidence = self.ocr_processor.validate_timer_format(timer_text) if timer_text else 0
        
        messagebox.showinfo("Timer OCR Test", 
            f"Raw OCR: '{ocr_text}'\n"
            f"Cleaned Timer: '{timer_text}'\n"
            f"Format Confidence: {confidence}/120\n"
            f"Valid Format: {'Yes' if confidence >= 100 else 'No'}\n\n"
            f"Settings:\n"
            f"Scale: {self.scale_var.get():.1f}x\n"
            f"Threshold: {self.threshold_var.get()}\n"
            f"Cleanup: {self.cleanup_var.get()}")

    def save_settings(self):
        """Save specialized timer OCR settings"""
        settings = {
            'processing_type': 'specialized_timer',
            'scale_factor': self.scale_var.get(),
            'white_threshold_low': self.threshold_var.get(),
            'morphology_strength': self.cleanup_var.get(),
            'timer_format': 'X:XX.XXX',
            'text_colors': 'white_yellow',
            'background': 'semi_transparent_black'
        }
        
        try:
            import json
            with open('timer_ocr_config.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo("Settings Saved", 
                f"Specialized timer OCR settings saved!\n\n"
                f"Configuration:\n"
                f"• Format: X:XX.XXX only\n"
                f"• Colors: White/Yellow text detection\n"
                f"• Scale: {self.scale_var.get():.1f}x magnification\n"
                f"• Threshold: {self.threshold_var.get()}\n"
                f"• Cleanup: Level {self.cleanup_var.get()}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save settings: {str(e)}")

    def back_to_camera(self):
        """Return to camera setup"""
        self.stop_preview()
        self.dialog.destroy()
        
        # Import camera setup
        try:
            try:
                from .camera_setup import CameraSetup
            except ImportError:
                from camera_setup import CameraSetup
            
            camera_app = CameraSetup(self.parent)
            self.parent.wait_window(camera_app.dialog)
            
        except ImportError as e:
            messagebox.showerror("Error", f"camera_setup.py not found: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera setup: {str(e)}")

    def finish_setup(self):
        """Complete setup"""
        if not self.last_detected_time:
            if not messagebox.askyesno("Warning", 
                "No valid timer format detected yet. Continue anyway?\n\n"
                "The system is optimized for X:XX.XXX format only."):
                return
        
        # Auto-save settings on finish
        self.save_settings()
        
        messagebox.showinfo("Setup Complete", 
            f"Specialized Timer OCR setup complete!\n\n"
            f"✓ Optimized for format: X:XX.XXX\n"
            f"✓ White/Yellow text detection\n"
            f"✓ Semi-transparent background handling\n"
            f"✓ Fixed font recognition\n\n"
            f"Last detected: {self.last_detected_time if self.last_detected_time else 'None'}\n\n"
            f"The system will now provide highly accurate timer readings\n"
            f"for your specific use case!")
        
        self.stop_preview()
        self.dialog.destroy()

    def stop_preview(self):
        """Stop camera and processing"""
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
def run_specialized_timer_ocr_setup(camera_index: int, crop_corners: List[Tuple[int, int]]) -> bool:
    """Run the specialized timer OCR setup"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = SpecializedTimerOCRSetup(root, camera_index, crop_corners)
        root.wait_window(app.dialog)
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Specialized Timer OCR setup failed: {str(e)}")
        return False


# Additional utility function for integration
def create_timer_ocr_processor():
    """Create a standalone timer OCR processor for use in other applications"""
    return SpecializedTimerOCRProcessor().start()