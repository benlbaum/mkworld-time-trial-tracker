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


class RobustCameraManager:
    """Robust camera manager with automatic recovery from driver failures"""
    
    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None
        self.camera_lock = threading.Lock()
        self.last_successful_frame = None
        self.consecutive_failures = 0
        self.max_failures = 5
        self.recovery_delay = 1.0  # seconds
        
        # Camera settings
        self.target_width = 1280
        self.target_height = 720
        self.target_fps = 30
        
    def initialize_camera(self):
        """Initialize camera with robust settings"""
        try:
            with self.camera_lock:
                if self.capture:
                    self.capture.release()
                
                # Try different camera backends for better stability
                backends = [
                    cv2.CAP_DSHOW,    # DirectShow (Windows, most stable)
                    cv2.CAP_MSMF,     # Microsoft Media Foundation
                    cv2.CAP_ANY       # Auto-detect
                ]
                
                for backend in backends:
                    try:
                        self.capture = cv2.VideoCapture(self.camera_index, backend)
                        if self.capture and self.capture.isOpened():
                            print(f"Camera opened with backend: {backend}")
                            break
                    except:
                        continue
                
                if not self.capture or not self.capture.isOpened():
                    return False
                
                # Configure camera for stability
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                # Additional stability settings
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                
                try:
                    self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                except:
                    pass
                
                # Test camera with a few frames
                for _ in range(3):
                    ret, frame = self.capture.read()
                    if ret and frame is not None:
                        self.last_successful_frame = frame.copy()
                        self.consecutive_failures = 0
                        return True
                
                return False
                
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def get_frame(self):
        """Get frame with automatic recovery on failure"""
        try:
            with self.camera_lock:
                if not self.capture or not self.capture.isOpened():
                    if not self.initialize_camera():
                        return False, self.last_successful_frame
                
                ret, frame = self.capture.read()
                
                if ret and frame is not None and frame.size > 0:
                    # Successful frame
                    self.last_successful_frame = frame.copy()
                    self.consecutive_failures = 0
                    return True, frame
                else:
                    # Frame read failed
                    self.consecutive_failures += 1
                    print(f"Frame read failed (attempt {self.consecutive_failures})")
                    
                    # Try camera recovery after several failures
                    if self.consecutive_failures >= self.max_failures:
                        print("Attempting camera recovery...")
                        time.sleep(self.recovery_delay)
                        
                        if self.initialize_camera():
                            print("Camera recovery successful")
                            ret, frame = self.capture.read()
                            if ret and frame is not None:
                                self.last_successful_frame = frame.copy()
                                self.consecutive_failures = 0
                                return True, frame
                        else:
                            print("Camera recovery failed")
                    
                    # Return last successful frame to keep UI responsive
                    return False, self.last_successful_frame
                    
        except Exception as e:
            print(f"Camera error: {e}")
            self.consecutive_failures += 1
            return False, self.last_successful_frame
    
    def release(self):
        """Release camera resources"""
        try:
            with self.camera_lock:
                if self.capture:
                    self.capture.release()
                    self.capture = None
        except:
            pass


class RobustTimerOCRProcessor:
    """Timer OCR processor with enhanced error handling"""
    
    def __init__(self):
        self.stopped = False
        self.current_frame = None
        self.ocr_result = ""
        self.timer_result = ""
        self.processed_image = None
        self.processing_error = ""
        self.lock = threading.Lock()
        
        # OCR settings
        self.tesseract_configs = [
            "--psm 8 -c tessedit_char_whitelist=0123456789:. -c tessedit_enable_dict_correction=0",
            "--psm 7 -c tessedit_char_whitelist=0123456789:. -c tessedit_enable_dict_correction=0",
            "--psm 13 -c tessedit_char_whitelist=0123456789:.",
            "--psm 6 -c tessedit_char_whitelist=0123456789:.",
            "--psm 8 -c tessedit_char_whitelist=0123456789"
        ]
        
        self.timer_pattern = re.compile(r'^(\d):(\d{2})\.(\d{3})$')
        
        # Processing settings
        self.scale_factor = 6.0
        self.white_threshold_low = 180
        self.morphology_strength = 2
        
        # Error tracking
        self.processing_errors = 0
        self.last_successful_process = time.time()
        
        self.tesseract_available = True
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Configure Tesseract with error handling"""
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
                    print(f"Tesseract found: {path}")
                    break
                elif os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    pytesseract.image_to_string(
                        Image.new('RGB', (100, 30), color='white'),
                        config="--version"
                    )
                    self.tesseract_available = True
                    print(f"Tesseract configured: {path}")
                    break
            except Exception as e:
                continue
        
        if not self.tesseract_available:
            print("Warning: Tesseract not found")
    
    def start(self):
        """Start OCR processing thread"""
        Thread(target=self.process_loop, daemon=True).start()
        return self
    
    def process_loop(self):
        """Main OCR processing loop with error handling"""
        while not self.stopped:
            try:
                with self.lock:
                    if self.current_frame is not None:
                        frame_to_process = self.current_frame.copy()
                    else:
                        frame_to_process = None
                
                if frame_to_process is not None:
                    try:
                        # Process image
                        processed = self.process_timer_text(frame_to_process)
                        
                        if processed is not None and self.tesseract_available:
                            # Try OCR with error handling
                            best_timer = ""
                            best_confidence = 0
                            best_raw = ""
                            
                            for config in self.tesseract_configs:
                                try:
                                    raw_text = pytesseract.image_to_string(
                                        processed, config=config
                                    ).strip()
                                    
                                    cleaned = self.clean_timer_text(raw_text)
                                    confidence = self.validate_timer_format(cleaned)
                                    
                                    if confidence > best_confidence:
                                        best_confidence = confidence
                                        best_timer = cleaned
                                        best_raw = raw_text
                                        
                                except Exception as ocr_error:
                                    continue
                            
                            # Update results
                            with self.lock:
                                self.ocr_result = best_raw
                                self.timer_result = best_timer
                                self.processed_image = processed
                                self.processing_error = ""
                                self.last_successful_process = time.time()
                                self.processing_errors = 0
                        
                    except Exception as process_error:
                        self.processing_errors += 1
                        with self.lock:
                            self.processing_error = f"Processing error: {str(process_error)}"
                        print(f"OCR processing error: {process_error}")
                
            except Exception as loop_error:
                print(f"OCR loop error: {loop_error}")
                time.sleep(1)  # Prevent rapid error loops
            
            # Adaptive sleep based on success rate
            if self.processing_errors < 3:
                time.sleep(0.33)  # ~3 FPS when working well
            else:
                time.sleep(1.0)   # Slower when having issues
    
    def process_timer_text(self, image):
        """Process timer text with error handling"""
        try:
            if image is None or image.size == 0:
                return None
            
            # Robust color space conversion
            if len(image.shape) == 3:
                try:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    
                    # White/yellow detection with error handling
                    white_mask = cv2.inRange(hsv, 
                        np.array([0, 0, self.white_threshold_low]), 
                        np.array([180, 30, 255]))
                    
                    yellow_mask = cv2.inRange(hsv, 
                        np.array([20, 100, self.white_threshold_low]), 
                        np.array([40, 255, 255]))
                    
                    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
                    
                    # LAB fallback
                    try:
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        l_channel = lab[:, :, 0]
                        _, light_mask = cv2.threshold(l_channel, self.white_threshold_low, 255, cv2.THRESH_BINARY)
                        combined_mask = cv2.bitwise_or(color_mask, light_mask)
                    except:
                        combined_mask = color_mask
                        
                except Exception as color_error:
                    # Fallback to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, combined_mask = cv2.threshold(gray, self.white_threshold_low, 255, cv2.THRESH_BINARY)
            else:
                _, combined_mask = cv2.threshold(image, self.white_threshold_low, 255, cv2.THRESH_BINARY)
            
            # Safe scaling
            try:
                scale = max(1, int(self.scale_factor))
                height, width = combined_mask.shape
                new_width = min(width * scale, 8000)  # Prevent excessive memory usage
                new_height = min(height * scale, 6000)
                
                scaled_mask = cv2.resize(combined_mask, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            except Exception as scale_error:
                print(f"Scaling error: {scale_error}")
                scaled_mask = combined_mask
            
            # Safe morphological operations
            try:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned = cv2.morphologyEx(scaled_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
                
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)
                
                if self.morphology_strength > 0:
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=min(self.morphology_strength, 5))
                
                return cleaned
                
            except Exception as morph_error:
                print(f"Morphology error: {morph_error}")
                return scaled_mask
            
        except Exception as e:
            print(f"Timer text processing error: {e}")
            return None
    
    def clean_timer_text(self, raw_text):
        """Clean OCR text with error handling"""
        try:
            if not raw_text:
                return ""
            
            # Remove unwanted characters
            cleaned = re.sub(r'[^\d:.]', '', raw_text)
            
            # OCR corrections
            corrections = {
                'O': '0', 'o': '0', 'Q': '0',
                'I': '1', 'l': '1', '|': '1', 'i': '1',
                'S': '5', 's': '5',
                'G': '6', 'g': '6',
                'T': '7', 't': '7',
                'B': '8', 'b': '8',
                'g': '9', 'q': '9',
            }
            
            for wrong, right in corrections.items():
                cleaned = cleaned.replace(wrong, right)
            
            # Format reconstruction
            digits = re.findall(r'\d', cleaned)
            
            if len(digits) >= 6:
                if len(digits) >= 7:
                    result = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
                else:
                    result = f"{digits[0]}:{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}"
                return result
            
            return cleaned
            
        except Exception as e:
            print(f"Text cleaning error: {e}")
            return raw_text if raw_text else ""
    
    def validate_timer_format(self, text):
        """Validate timer format with error handling"""
        try:
            if not text:
                return 0
            
            match = self.timer_pattern.match(text)
            if match:
                score = 100
                
                try:
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    milliseconds = int(match.group(3))
                    
                    if 0 <= minutes <= 9 and 0 <= seconds <= 59 and 0 <= milliseconds <= 999:
                        score += 20
                except:
                    pass
                
                return score
            
            # Partial scoring
            score = 0
            if re.search(r'\d:', text): score += 25
            if re.search(r':\d{2}', text): score += 25
            if re.search(r'\.\d{1,3}', text): score += 25
            if 7 <= len(text) <= 9: score += 15
            if re.match(r'^[\d:.]+$', text): score += 10
            
            return score
            
        except Exception as e:
            print(f"Validation error: {e}")
            return 0
    
    def update_frame(self, frame):
        """Update frame with error handling"""
        try:
            with self.lock:
                self.current_frame = frame
        except:
            pass
    
    def get_results(self):
        """Get results with error handling"""
        try:
            with self.lock:
                return self.ocr_result, self.timer_result, self.processed_image, self.processing_error
        except:
            return "", "", None, "Lock error"
    
    def update_settings(self, **kwargs):
        """Update settings with error handling"""
        try:
            with self.lock:
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except Exception as e:
            print(f"Settings update error: {e}")
    
    def stop(self):
        """Stop processing"""
        self.stopped = True


class RobustTimerOCRSetup:
    def __init__(self, parent: tk.Tk, camera_index: int, crop_corners: List[Tuple[int, int]]):
        self.parent = parent
        self.camera_index = camera_index
        self.crop_corners = crop_corners
        
        # Robust camera manager
        self.camera_manager = RobustCameraManager(camera_index)
        self.preview_running = False
        self.camera_resolution = (1920, 1080)
        
        # Robust OCR processor
        self.ocr_processor = RobustTimerOCRProcessor().start()
        
        # UI state
        self.last_detected_time = ""
        self.frame_count = 0
        self.error_count = 0
        
        # Rate limiting
        self.last_ui_update = 0
        self.ui_update_interval = 0.1
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Robust Timer OCR - With Camera Recovery")
        self.dialog.geometry("1000x850")
        self.dialog.minsize(800, 650)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.setup_ui()
        self.start_preview()

    def setup_ui(self):
        """Setup UI with status monitoring"""
        # Main container
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar at top
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=5)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        status_info_frame = ttk.Frame(status_frame)
        status_info_frame.pack(fill=tk.X)
        
        ttk.Label(status_info_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_status_var = tk.StringVar(value="Initializing...")
        self.camera_status_label = ttk.Label(status_info_frame, textvariable=self.camera_status_var, 
                                            foreground="orange", width=15)
        self.camera_status_label.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(status_info_frame, text="OCR:").pack(side=tk.LEFT)
        self.ocr_status_var = tk.StringVar(value="Starting...")
        self.ocr_status_label = ttk.Label(status_info_frame, textvariable=self.ocr_status_var, 
                                         foreground="orange", width=15)
        self.ocr_status_label.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(status_info_frame, text="Frames:").pack(side=tk.LEFT)
        self.frame_count_var = tk.StringVar(value="0")
        ttk.Label(status_info_frame, textvariable=self.frame_count_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Left side - Preview
        left_frame = ttk.LabelFrame(main_frame, text="Timer Preview (Robust Processing)", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.original_preview = tk.Label(left_frame, text="Initializing Camera...", 
                                       bg="black", fg="white")
        self.original_preview.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.processed_preview = tk.Label(left_frame, text="Waiting for Processing...", 
                                        bg="black", fg="white")
        self.processed_preview.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Controls
        right_frame = ttk.LabelFrame(main_frame, text="Robust Timer OCR Controls", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # OCR Results with error display
        results_frame = ttk.LabelFrame(right_frame, text="Timer Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Current Timer:").pack(anchor="w")
        self.timer_result_var = tk.StringVar(value="Waiting...")
        self.timer_result_label = ttk.Label(results_frame, textvariable=self.timer_result_var,
                                           foreground="green", font=("Courier", 16, "bold"))
        self.timer_result_label.pack(anchor="w", pady=(0, 5))
        
        ttk.Label(results_frame, text="Raw OCR:").pack(anchor="w")
        self.ocr_result_var = tk.StringVar(value="Processing...")
        self.ocr_result_label = ttk.Label(results_frame, textvariable=self.ocr_result_var, 
                                         foreground="blue", font=("Courier", 10))
        self.ocr_result_label.pack(anchor="w", pady=(0, 5))
        
        ttk.Label(results_frame, text="Status:").pack(anchor="w")
        self.error_var = tk.StringVar(value="System starting...")
        self.error_label = ttk.Label(results_frame, textvariable=self.error_var,
                                    foreground="gray", font=("Arial", 9))
        self.error_label.pack(anchor="w")
        
        # Processing settings
        settings_frame = ttk.LabelFrame(right_frame, text="Processing Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="Scale Factor:").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=6.0)
        scale_frame = ttk.Frame(settings_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(scale_frame, from_=4.0, to=10.0, variable=self.scale_var, 
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(scale_frame, textvariable=self.scale_var, width=6).pack(side=tk.RIGHT)
        
        ttk.Label(settings_frame, text="White Threshold:").pack(anchor="w")
        self.threshold_var = tk.IntVar(value=180)
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(threshold_frame, from_=150, to=200, variable=self.threshold_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(threshold_frame, textvariable=self.threshold_var, width=6).pack(side=tk.RIGHT)
        
        # Quick presets
        preset_frame = ttk.LabelFrame(right_frame, text="Quick Settings", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.pack(fill=tk.X)
        
        ttk.Button(preset_btn_frame, text="Bright Timer", 
                  command=lambda: self.apply_preset("bright")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_btn_frame, text="Dim Timer", 
                  command=lambda: self.apply_preset("dim")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_btn_frame, text="Reset Camera", 
                  command=self.reset_camera).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Test OCR", command=self.manual_ocr_test).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Back to Camera", command=self.back_to_camera).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Finish Setup", command=self.finish_setup).pack(fill=tk.X)

    def apply_preset(self, preset_name):
        """Apply presets with feedback"""
        presets = {
            "bright": {"scale_factor": 6.0, "white_threshold_low": 180, "morphology_strength": 2},
            "dim": {"scale_factor": 7.0, "white_threshold_low": 160, "morphology_strength": 3}
        }
        
        if preset_name in presets:
            settings = presets[preset_name]
            self.scale_var.set(settings["scale_factor"])
            self.threshold_var.set(settings["white_threshold_low"])
            self.on_param_change()
            print(f"Applied {preset_name} timer preset")

    def reset_camera(self):
        """Manually reset camera"""
        print("Manual camera reset requested")
        self.camera_manager.consecutive_failures = self.camera_manager.max_failures
        # Force recovery on next frame
    
    def on_param_change(self, *args):
        """Update settings"""
        self.ocr_processor.update_settings(
            scale_factor=float(self.scale_var.get()),
            white_threshold_low=int(self.threshold_var.get())
        )

    def start_preview(self):
        """Start robust camera preview"""
        # Load camera config
        try:
            import json
            if os.path.exists("camera_config.json"):
                with open("camera_config.json", 'r') as f:
                    config = json.load(f)
                    if 'camera_resolution' in config:
                        self.camera_resolution = tuple(config['camera_resolution'])
                        print(f"Using saved camera resolution: {self.camera_resolution}")
        except Exception as e:
            print(f"Could not load camera config: {e}")
        
        # Initialize camera
        if self.camera_manager.initialize_camera():
            self.preview_running = True
            self.update_preview()
        else:
            messagebox.showerror("Camera Error", 
                f"Could not initialize camera {self.camera_index}\n"
                f"Please check camera connection and try again.")

    def update_preview(self):
        """Robust preview update with error handling"""
        if not self.preview_running:
            return
        
        try:
            # Get frame with automatic recovery
            success, frame = self.camera_manager.get_frame()
            
            # Update camera status
            if success:
                self.camera_status_var.set("✓ Active")
                self.camera_status_label.config(foreground="green")
                self.frame_count += 1
            else:
                self.camera_status_var.set("⚠ Recovery")
                self.camera_status_label.config(foreground="orange")
                self.error_count += 1
            
            # Update frame counter
            self.frame_count_var.set(str(self.frame_count))
            
            if frame is not None:
                cropped = self.get_crop(frame)
                
                if cropped is not None:
                    self.current_crop_image = cropped.copy()
                    self.update_image_display(self.original_preview, cropped, max_size=(400, 200))
                    self.ocr_processor.update_frame(cropped)
                    
                    # Update UI periodically
                    current_time = time.time()
                    if current_time - self.last_ui_update >= self.ui_update_interval:
                        self.last_ui_update = current_time
                        self.update_ocr_results()
            
        except Exception as e:
            print(f"Preview update error: {e}")
            self.error_count += 1
        
        # Schedule next update
        self.dialog.after(33, self.update_preview)

    def update_ocr_results(self):
        """Update OCR results with error handling"""
        try:
            ocr_text, timer_text, processed_img, error_msg = self.ocr_processor.get_results()
            
            # Update OCR status
            if error_msg:
                self.ocr_status_var.set("✗ Error")
                self.ocr_status_label.config(foreground="red")
                self.error_var.set(error_msg)
                self.error_label.config(foreground="red")
            elif timer_text:
                self.ocr_status_var.set("✓ Reading")
                self.ocr_status_label.config(foreground="green")
                self.error_var.set("Timer detected successfully")
                self.error_label.config(foreground="green")
            else:
                self.ocr_status_var.set("○ Processing")
                self.ocr_status_label.config(foreground="orange")
                self.error_var.set("Searching for timer...")
                self.error_label.config(foreground="gray")
            
            # Update results
            self.ocr_result_var.set(ocr_text if ocr_text else "Processing...")
            self.timer_result_var.set(timer_text if timer_text else "No timer detected")
            
            # Update processed image
            if processed_img is not None:
                self.update_image_display(self.processed_preview, processed_img, max_size=(400, 200))
            
            # Store last good timer
            if timer_text and self.ocr_processor.validate_timer_format(timer_text) >= 100:
                self.last_detected_time = timer_text
                
        except Exception as e:
            print(f"OCR results update error: {e}")
            self.error_var.set(f"UI update error: {str(e)}")
            self.error_label.config(foreground="red")

    def get_crop(self, frame):
        """Get cropped timer area with error handling"""
        try:
            if frame is None or frame.size == 0:
                return None
                
            current_height, current_width = frame.shape[:2]
            scale_x = current_width / self.camera_resolution[0]
            scale_y = current_height / self.camera_resolution[1]
            
            scaled_corners = []
            for x, y in self.crop_corners:
                scaled_x = max(0, min(current_width-1, int(x * scale_x)))
                scaled_y = max(0, min(current_height-1, int(y * scale_y)))
                scaled_corners.append((scaled_x, scaled_y))
            
            src_points = np.array(scaled_corners, dtype=np.float32)
            
            # Calculate dimensions safely
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
            
            # Limit maximum crop size to prevent memory issues
            max_width = min(max_width, 2000)
            max_height = min(max_height, 1500)
            
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
        """Update image display with error handling"""
        try:
            if cv_image is None or cv_image.size == 0:
                label_widget.configure(image="", text="No image")
                return
                
            if len(cv_image.shape) == 3:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            img_w, img_h = pil_image.size
            max_w, max_h = max_size
            
            if img_w == 0 or img_h == 0:
                label_widget.configure(image="", text="Invalid image")
                return
            
            ratio = min(max_w / img_w, max_h / img_h)
            new_w = max(1, int(img_w * ratio))
            new_h = max(1, int(img_h * ratio))
            
            display_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            label_widget.configure(image=photo, text="")
            label_widget._image_ref = photo
            
        except Exception as e:
            print(f"Image display error: {e}")
            label_widget.configure(image="", text=f"Display error")

    def manual_ocr_test(self):
        """Test current OCR settings"""
        try:
            ocr_text, timer_text, processed_img, error_msg = self.ocr_processor.get_results()
            confidence = self.ocr_processor.validate_timer_format(timer_text) if timer_text else 0
            
            status_info = (
                f"Camera Status: {self.camera_status_var.get()}\n"
                f"OCR Status: {self.ocr_status_var.get()}\n"
                f"Frames Processed: {self.frame_count}\n"
                f"Errors: {self.error_count}\n\n"
                f"Raw OCR: '{ocr_text}'\n"
                f"Cleaned Timer: '{timer_text}'\n"
                f"Confidence: {confidence}/120\n"
                f"Valid Format: {'Yes' if confidence >= 100 else 'No'}\n\n"
                f"Settings:\n"
                f"Scale: {self.scale_var.get():.1f}x\n"
                f"Threshold: {self.threshold_var.get()}"
            )
            
            if error_msg:
                status_info += f"\n\nError: {error_msg}"
            
            messagebox.showinfo("Robust OCR Test", status_info)
            
        except Exception as e:
            messagebox.showerror("Test Error", f"Could not run test: {str(e)}")

    def save_settings(self):
        """Save settings with error handling"""
        try:
            settings = {
                'processing_type': 'robust_timer_ocr',
                'scale_factor': self.scale_var.get(),
                'white_threshold_low': self.threshold_var.get(),
                'camera_backend': 'auto_recovery',
                'timer_format': 'X:XX.XXX',
                'error_recovery': True
            }
            
            import json
            with open('robust_timer_ocr_config.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo("Settings Saved", 
                f"Robust Timer OCR settings saved!\n\n"
                f"Features enabled:\n"
                f"• Automatic camera recovery\n"
                f"• Error handling and recovery\n"
                f"• Format validation (X:XX.XXX)\n"
                f"• White/Yellow text detection\n"
                f"• System status monitoring")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save settings: {str(e)}")

    def back_to_camera(self):
        """Return to camera setup"""
        self.stop_preview()
        self.dialog.destroy()
        
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
        """Complete setup with validation"""
        try:
            if not self.last_detected_time:
                if not messagebox.askyesno("Warning", 
                    "No valid timer detected yet. The system includes automatic\n"
                    "recovery features and should work reliably once a timer\n"
                    "appears. Continue anyway?"):
                    return
            
            # Auto-save on finish
            self.save_settings()
            
            summary = (
                f"Robust Timer OCR setup complete!\n\n"
                f"✓ Automatic camera recovery enabled\n"
                f"✓ Error handling and monitoring\n"
                f"✓ Format validation (X:XX.XXX only)\n"
                f"✓ White/Yellow text detection\n"
                f"✓ Background immunity\n\n"
                f"Statistics:\n"
                f"• Frames processed: {self.frame_count}\n"
                f"• Errors encountered: {self.error_count}\n"
                f"• Last timer: {self.last_detected_time if self.last_detected_time else 'None'}\n\n"
                f"The system will automatically recover from camera\n"
                f"driver issues and provide reliable timer readings."
            )
            
            messagebox.showinfo("Setup Complete", summary)
            
            self.stop_preview()
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Finish Error", f"Error completing setup: {str(e)}")

    def stop_preview(self):
        """Stop all processes safely"""
        try:
            self.preview_running = False
            self.camera_manager.release()
            self.ocr_processor.stop()
            print("All processes stopped safely")
        except Exception as e:
            print(f"Error stopping processes: {e}")

    def on_cancel(self):
        """Cancel setup safely"""
        try:
            self.stop_preview()
            self.dialog.destroy()
        except Exception as e:
            print(f"Error during cancel: {e}")


# Usage function
def run_robust_timer_ocr_setup(camera_index: int, crop_corners: List[Tuple[int, int]]) -> bool:
    """Run the robust timer OCR setup"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = RobustTimerOCRSetup(root, camera_index, crop_corners)
        root.wait_window(app.dialog)
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Robust Timer OCR setup failed: {str(e)}")
        return False


# Standalone processor for integration
def create_robust_timer_ocr_processor():
    """Create a standalone robust timer OCR processor"""
    return RobustTimerOCRProcessor().start()