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
import threading
import queue
import time

class OCRSetup:
    def __init__(self, parent: tk.Tk, camera_index: int, crop_corners: List[Tuple[int, int]]):
        self.parent = parent
        self.camera_index = camera_index
        self.crop_corners = crop_corners
        
        # Preview capture (low res)
        self.preview_capture: Optional[cv2.VideoCapture] = None
        self.preview_running = False
        self._preview_image = None
        
        # OCR capture (high res) - runs in separate thread
        self.ocr_capture: Optional[cv2.VideoCapture] = None
        self.ocr_thread_running = False
        self.ocr_thread = None
        
        # Thread communication
        self.ocr_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self.ocr_result_queue = queue.Queue()
        
        # Resolution settings
        self.preview_resolution = (1280, 720)  # Smooth preview
        self.ocr_resolution = (1920, 1080)     # High quality OCR
        self.resolution_scale_factor = 1.0
        
        # Configure Tesseract path
        self.setup_tesseract()
        
        # Color picker state
        self.color_picker_active = False
        self.selected_color_hsv = [0, 0, 255]
        self.color_range_h = 10
        self.color_range_s = 50
        self.color_range_v = 50
        
        # OCR parameters
        self.scale_factor = 3.0
        self.dilate_iterations = 1
        self.erode_iterations = 1
        
        # Timer pattern matching
        self.timer_pattern = r'\d:\d{2}\.\d{3}'
        self.last_detected_time = ""
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("OCR Setup - Timer Recognition")
        self.dialog.geometry("1200x800")
        self.dialog.minsize(800, 600)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.setup_ui()
        self.start_dual_capture()

    def setup_camera_resolutions(self):
        """Set up both preview and OCR cameras with appropriate resolutions"""
        # Preview camera setup (lower resolution for smooth UI)
        self.preview_capture = cv2.VideoCapture(self.camera_index)
        if not self.preview_capture.isOpened():
            return False
            
        # Set preview to lower resolution for smoothness
        preview_targets = [(1280, 720), (960, 540), (800, 600), (640, 480)]
        
        for width, height in preview_targets:
            self.preview_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.preview_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.preview_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.preview_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                self.preview_resolution = (actual_width, actual_height)
                print(f"Preview resolution: {self.preview_resolution}")
                break
        
        # OCR camera setup (maximum resolution for quality)
        self.ocr_capture = cv2.VideoCapture(self.camera_index)
        if not self.ocr_capture.isOpened():
            return False
            
        # Set OCR to maximum resolution
        ocr_targets = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (640, 480)]
        
        for width, height in ocr_targets:
            self.ocr_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.ocr_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.ocr_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.ocr_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                self.ocr_resolution = (actual_width, actual_height)
                print(f"OCR resolution: {self.ocr_resolution}")
                break
        
        # Calculate scale factor between preview and OCR resolutions
        self.resolution_scale_factor = self.ocr_resolution[0] / self.preview_resolution[0]
        print(f"Resolution scale factor: {self.resolution_scale_factor:.2f}")
        
        return True

    def start_dual_capture(self):
        """Start both preview and OCR capture threads"""
        if not self.setup_camera_resolutions():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
            return
            
        # Start preview thread
        self.preview_running = True
        self.update_preview()
        
        # Start OCR processing thread
        self.ocr_thread_running = True
        self.ocr_thread = threading.Thread(target=self.ocr_worker_thread, daemon=True)
        self.ocr_thread.start()
        
        # Start OCR result checker
        self.check_ocr_results()

    def stop_dual_capture(self):
        """Stop both capture threads"""
        self.preview_running = False
        self.ocr_thread_running = False
        
        if self.preview_capture:
            self.preview_capture.release()
            self.preview_capture = None
            
        if self.ocr_capture:
            self.ocr_capture.release()
            self.ocr_capture = None
            
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=1.0)

    def update_preview(self):
        """Update the preview displays (low resolution, high FPS)"""
        if not self.preview_running or not self.preview_capture:
            return
            
        ret, frame = self.preview_capture.read()
        if ret and frame is not None:
            # Convert crop corners from OCR resolution to preview resolution
            preview_corners = []
            for ocr_x, ocr_y in self.crop_corners:
                preview_x = ocr_x / self.resolution_scale_factor
                preview_y = ocr_y / self.resolution_scale_factor
                preview_corners.append((int(preview_x), int(preview_y)))
            
            # Get preview crop for display
            preview_crop = self.get_preview_crop(frame, preview_corners)
            if preview_crop is not None:
                self.update_image_display(self.original_preview, preview_crop, max_size=(300, 200))
                
                # Send frame to OCR thread for processing (non-blocking)
                try:
                    self.ocr_queue.put_nowait(time.time())  # Just send a timestamp signal
                except queue.Full:
                    pass  # Skip if OCR thread is busy
        
        # Schedule next preview update (high frequency for smooth preview)
        self.dialog.after(33, self.update_preview)  # ~30 FPS

    def ocr_worker_thread(self):
        """Background thread for high-resolution OCR processing"""
        last_process_time = 0
        process_interval = 0.5  # Process OCR every 500ms for responsiveness
        
        while self.ocr_thread_running:
            try:
                # Wait for signal from preview thread
                timestamp = self.ocr_queue.get(timeout=0.1)
                
                # Throttle OCR processing
                current_time = time.time()
                if current_time - last_process_time < process_interval:
                    continue
                    
                last_process_time = current_time
                
                # Capture high-resolution frame
                if self.ocr_capture:
                    ret, frame = self.ocr_capture.read()
                    if ret and frame is not None:
                        # Get high-resolution crop
                        cropped = self.get_ocr_crop(frame)
                        if cropped is not None:
                            # Process for OCR
                            processed = self.process_image_for_ocr(cropped)
                            if processed is not None:
                                # Perform OCR
                                raw_text, timer_text = self.perform_ocr(processed)
                                
                                # Send results back to main thread
                                try:
                                    self.ocr_result_queue.put_nowait({
                                        'processed_image': processed,
                                        'raw_text': raw_text,
                                        'timer_text': timer_text,
                                        'timestamp': current_time
                                    })
                                except queue.Full:
                                    # Remove old results if queue is full
                                    try:
                                        self.ocr_result_queue.get_nowait()
                                        self.ocr_result_queue.put_nowait({
                                            'processed_image': processed,
                                            'raw_text': raw_text,
                                            'timer_text': timer_text,
                                            'timestamp': current_time
                                        })
                                    except queue.Empty:
                                        pass
                                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"OCR thread error: {e}")
                time.sleep(0.1)

    def check_ocr_results(self):
        """Check for OCR results from background thread and update UI"""
        try:
            while True:
                result = self.ocr_result_queue.get_nowait()
                
                # Update UI with OCR results
                self.ocr_result_var.set(result['raw_text'] if result['raw_text'] else "No text detected")
                self.timer_result_var.set(result['timer_text'] if result['timer_text'] else "No timer detected")
                
                if result['timer_text']:
                    self.last_detected_time = result['timer_text']
                
                # Update processed image display
                if result['processed_image'] is not None:
                    self.update_image_display(self.processed_preview, result['processed_image'], max_size=(300, 200))
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error checking OCR results: {e}")
        
        # Schedule next check
        if self.preview_running:
            self.dialog.after(100, self.check_ocr_results)  # Check every 100ms

    def get_preview_crop(self, frame, preview_corners):
        """Get cropped area from preview frame"""
        try:
            src_points = np.array(preview_corners, dtype=np.float32)
            
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
            print(f"Preview crop error: {e}")
            return None

    def get_ocr_crop(self, frame):
        """Get cropped area from high-resolution frame for OCR"""
        try:
            src_points = np.array(self.crop_corners, dtype=np.float32)
            
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
            print(f"OCR crop error: {e}")
            return None

    def setup_tesseract(self):
        """Configure Tesseract executable path"""
        # [Keep existing tesseract setup code]
        possible_paths = []
        
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
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
        
        tesseract_found = False
        for path in possible_paths:
            try:
                if path in ["tesseract", "tesseract.exe"]:
                    test_result = pytesseract.image_to_string(
                        Image.new('RGB', (100, 30), color='white'),
                        config="--version"
                    )
                    tesseract_found = True
                    break
                elif os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    test_result = pytesseract.image_to_string(
                        Image.new('RGB', (100, 30), color='white'),
                        config="--version"
                    )
                    tesseract_found = True
                    break
            except:
                continue
        
        self.tesseract_available = tesseract_found

    def setup_ui(self):
        # Main container with two columns
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Preview
        left_frame = ttk.LabelFrame(main_frame, text="Timer Preview", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Original cropped preview with click binding
        self.original_preview = tk.Label(left_frame, text="Original Crop\n(Click to pick text color)", 
                                       bg="black", fg="white", cursor="crosshair")
        self.original_preview.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.original_preview.bind("<Button-1>", self.on_color_pick_click)
        self.original_preview.bind("<Motion>", self.on_mouse_motion)
        
        # Processed preview
        self.processed_preview = tk.Label(left_frame, text="Color Masked for OCR", bg="black", fg="white")
        self.processed_preview.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Controls and results
        right_frame = ttk.LabelFrame(main_frame, text="OCR Controls", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.config(width=400)
        
        # OCR Results
        results_frame = ttk.LabelFrame(right_frame, text="OCR Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_frame, text="Detected Text:").pack(anchor="w")
        self.ocr_result_var = tk.StringVar()
        self.ocr_result_label = ttk.Label(results_frame, textvariable=self.ocr_result_var, 
                                         foreground="blue", font=("Courier", 12, "bold"))
        self.ocr_result_label.pack(anchor="w", pady=(0, 5))
        
        ttk.Label(results_frame, text="Parsed Timer:").pack(anchor="w")
        self.timer_result_var = tk.StringVar()
        self.timer_result_label = ttk.Label(results_frame, textvariable=self.timer_result_var,
                                           foreground="green", font=("Courier", 14, "bold"))
        self.timer_result_label.pack(anchor="w")
        
        # Color Picker Controls
        color_frame = ttk.LabelFrame(right_frame, text="Text Color Selection", padding=10)
        color_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Instructions
        instruction_text = "Click on timer text in preview above to select color"
        ttk.Label(color_frame, text=instruction_text, foreground="gray", wraplength=300).pack(pady=(0, 5))
        
        # Color picker button and display
        picker_frame = ttk.Frame(color_frame)
        picker_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.color_picker_btn = ttk.Button(picker_frame, text="Activate Color Picker", 
                                          command=self.toggle_color_picker)
        self.color_picker_btn.pack(side=tk.LEFT)
        
        # Color display
        self.color_display = tk.Label(picker_frame, text="  ", bg="white", relief="sunken", width=4)
        self.color_display.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Selected color info
        self.color_info_var = tk.StringVar(value="HSV: (0, 0, 255)")
        ttk.Label(color_frame, textvariable=self.color_info_var, font=("Courier", 9)).pack(anchor="w")
        
        # Live color info (shows HSV under mouse when color picker is active)
        self.live_color_var = tk.StringVar(value="Hover over image to see live HSV")
        self.live_color_label = ttk.Label(color_frame, textvariable=self.live_color_var, 
                                         font=("Courier", 8), foreground="gray")
        self.live_color_label.pack(anchor="w")
        
        # Color tolerance controls
        ttk.Label(color_frame, text="Color Tolerance Ranges:").pack(anchor="w", pady=(10, 0))
        
        # Hue range
        hue_frame = ttk.Frame(color_frame)
        hue_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(hue_frame, text="Hue ±:").pack(side=tk.LEFT)
        self.hue_range_var = tk.IntVar(value=self.color_range_h)
        ttk.Scale(hue_frame, from_=0, to=50, variable=self.hue_range_var,
                 orient="horizontal", command=self.on_color_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Label(hue_frame, textvariable=self.hue_range_var, width=3).pack(side=tk.RIGHT)
        
        # Saturation range
        sat_frame = ttk.Frame(color_frame)
        sat_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(sat_frame, text="Sat ±:").pack(side=tk.LEFT)
        self.sat_range_var = tk.IntVar(value=self.color_range_s)
        ttk.Scale(sat_frame, from_=0, to=100, variable=self.sat_range_var,
                 orient="horizontal", command=self.on_color_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Label(sat_frame, textvariable=self.sat_range_var, width=3).pack(side=tk.RIGHT)
        
        # Value range
        val_frame = ttk.Frame(color_frame)
        val_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(val_frame, text="Val ±:").pack(side=tk.LEFT)
        self.val_range_var = tk.IntVar(value=self.color_range_v)
        ttk.Scale(val_frame, from_=0, to=100, variable=self.val_range_var,
                 orient="horizontal", command=self.on_color_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Label(val_frame, textvariable=self.val_range_var, width=3).pack(side=tk.RIGHT)
        
        # Reset button
        ttk.Button(color_frame, text="Reset to White", command=self.reset_to_white).pack(pady=(5, 0))
        
        # Image Processing Controls
        processing_frame = ttk.LabelFrame(right_frame, text="Image Enhancement", padding=10)
        processing_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scale factor
        ttk.Label(processing_frame, text="Scale Factor:").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=self.scale_factor)
        scale_frame = ttk.Frame(processing_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(scale_frame, from_=1.0, to=8.0, variable=self.scale_var, 
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(scale_frame, textvariable=self.scale_var, width=6).pack(side=tk.RIGHT)
        
        # Dilate iterations
        ttk.Label(processing_frame, text="Dilate (thicken):").pack(anchor="w")
        self.dilate_var = tk.IntVar(value=self.dilate_iterations)
        dilate_frame = ttk.Frame(processing_frame)
        dilate_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(dilate_frame, from_=0, to=5, variable=self.dilate_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(dilate_frame, textvariable=self.dilate_var, width=4).pack(side=tk.RIGHT)
        
        # Erode iterations
        ttk.Label(processing_frame, text="Erode (thin):").pack(anchor="w")
        self.erode_var = tk.IntVar(value=self.erode_iterations)
        erode_frame = ttk.Frame(processing_frame)
        erode_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Scale(erode_frame, from_=0, to=5, variable=self.erode_var,
                 orient="horizontal", command=self.on_param_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(erode_frame, textvariable=self.erode_var, width=4).pack(side=tk.RIGHT)
        
        # OCR Engine Controls
        ocr_frame = ttk.LabelFrame(right_frame, text="OCR Engine", padding=10)
        ocr_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tesseract status
        status_frame = ttk.Frame(ocr_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(status_frame, text="Tesseract Status:").pack(side=tk.LEFT)
        status_text = "✓ Ready" if self.tesseract_available else "✗ Not Found"
        status_color = "green" if self.tesseract_available else "red"
        ttk.Label(status_frame, text=status_text, foreground=status_color).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(ocr_frame, text="Tesseract Config:").pack(anchor="w")
        self.tesseract_config_var = tk.StringVar(value="--psm 8 -c tessedit_char_whitelist=0123456789:.")
        config_entry = ttk.Entry(ocr_frame, textvariable=self.tesseract_config_var, width=40)
        config_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(ocr_frame, text="Timer Pattern (Regex):").pack(anchor="w")
        self.pattern_var = tk.StringVar(value=self.timer_pattern)
        pattern_entry = ttk.Entry(ocr_frame, textvariable=self.pattern_var, width=40)
        pattern_entry.pack(fill=tk.X, pady=(0, 5))
        
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
                    
                    # Store as regular integers to avoid overflow warnings
                    self.selected_color_hsv = [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])]
                    self.update_color_display()
                    
                    # Don't auto-deactivate picker - let user keep picking colors

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

    def update_color_display(self):
        """Update the color display and info"""
        h, s, v = self.selected_color_hsv
        self.color_info_var.set(f"HSV: ({h}, {s}, {v})")
        
        # Convert HSV to RGB for display
        hsv_for_rgb = np.array([[[h, s, v]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_for_rgb, cv2.COLOR_HSV2RGB)[0, 0]
        
        # Update color display
        hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        self.color_display.config(bg=hex_color)

    def reset_to_white(self):
        """Reset color selection to white"""
        self.selected_color_hsv = [0, 0, 255]
        self.update_color_display()

    def on_color_param_change(self, *args):
        """Called when color tolerance parameters change"""
        self.color_range_h = int(self.hue_range_var.get())
        self.color_range_s = int(self.sat_range_var.get())
        self.color_range_v = int(self.val_range_var.get())

    def process_image_for_ocr(self, image):
        """Apply color masking and processing for OCR"""
        if image is None:
            return None
        
        # Store for color picking
        self.current_crop_image = image.copy()
        
        # Scale up the image first
        scale = float(self.scale_var.get())
        if scale > 1.0:
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to HSV for color masking
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            # Convert grayscale to HSV
            gray_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2HSV)
        
        # Create color mask based on selected color and tolerances
        # Convert to int to avoid overflow warnings
        h = int(self.selected_color_hsv[0])
        s = int(self.selected_color_hsv[1])
        v = int(self.selected_color_hsv[2])
        h_range = int(self.color_range_h)
        s_range = int(self.color_range_s)
        v_range = int(self.color_range_v)
        
        # Handle hue wrapping (0-179 in OpenCV) with proper bounds checking
        if h - h_range < 0:
            # Hue wraps around at 0
            lower1 = np.array([0, max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper1 = np.array([h + h_range, min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            lower2 = np.array([max(0, 180 + (h - h_range)), max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper2 = np.array([179, min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif h + h_range > 179:
            # Hue wraps around at 179
            lower1 = np.array([max(0, h - h_range), max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper1 = np.array([179, min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            lower2 = np.array([0, max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper2 = np.array([min(179, (h + h_range) - 180), min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Normal case - no hue wrapping
            lower = np.array([max(0, h - h_range), max(0, s - s_range), max(0, v - v_range)], dtype=np.uint8)
            upper = np.array([min(179, h + h_range), min(255, s + s_range), min(255, v + v_range)], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        dilate_iter = int(self.dilate_var.get())
        erode_iter = int(self.erode_var.get())
        
        if dilate_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
            
        if erode_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=erode_iter)
        
        # Additional cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    # [Keep all other existing methods: start_preview, stop_preview, get_perspective_corrected_crop, 
    #  update_preview, update_image_display, perform_ocr, on_param_change, manual_ocr_test, 
    #  save_settings, back_to_camera, finish_setup, on_cancel]

    def start_preview(self):
        """Start camera preview with crop applied"""
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
                return
            self.preview_running = True
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera preview: {str(e)}")

    def stop_preview(self):
        """Stop camera preview"""
        self.preview_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_perspective_corrected_crop(self):
        """Get high-resolution crop for OCR processing"""
        if not self.crop_corners:
            return None
            
        try:
            # Create a separate camera instance for high-resolution capture
            temp_capture = cv2.VideoCapture(self.camera_index)
            if not temp_capture.isOpened():
                print("Could not open camera for high-resolution capture")
                return None
            
            # Detect maximum resolution (same logic as camera_setup)
            resolutions_to_try = [
                (3840, 2160),  # 4K
                (2560, 1440),  # 1440p
                (1920, 1080),  # 1080p
                (1280, 720),   # 720p
                (1024, 768),   # XGA
                (800, 600),    # SVGA
                (640, 480)     # VGA (fallback)
            ]
            
            max_resolution = (640, 480)
            
            for width, height in resolutions_to_try:
                temp_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                temp_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(temp_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(temp_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                    max_resolution = (actual_width, actual_height)
                    break
            
            print(f"OCR using resolution: {max_resolution}")
            
            # Capture at high resolution
            ret = False
            for _ in range(5):  # Try a few times to get a good frame
                ret, frame = temp_capture.read()
                if ret and frame is not None:
                    break
            
            temp_capture.release()
            
            if not ret or frame is None:
                print("Could not capture high-resolution frame for OCR")
                return None
                
            print(f"OCR processing frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            # Use the crop corners directly (they're in the coordinate system we need)
            src_points = np.array(self.crop_corners, dtype=np.float32)
            
            widths = [
                np.linalg.norm(src_points[1] - src_points[0]),
                np.linalg.norm(src_points[2] - src_points[3])
            ]
            heights = [
                np.linalg.norm(src_points[3] - src_points[0]),
                np.linalg.norm(src_points[2] - src_points[1])
            ]
            
            max_width = int(max(widths))
            max_height = int(max(heights))
            
            dst_points = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            corrected = cv2.warpPerspective(frame, matrix, (max_width, max_height))
            
            print(f"Final cropped size: {corrected.shape[1]}x{corrected.shape[0]}")
            return corrected
            
        except Exception as e:
            print(f"Error in get_perspective_corrected_crop: {e}")
            return None

    def update_preview(self):
        """Update the preview displays and run OCR"""
        if not self.preview_running:
            return
            
        cropped = self.get_perspective_corrected_crop()
        
        if cropped is not None:
            self.update_image_display(self.original_preview, cropped, max_size=(300, 200))
            processed = self.process_image_for_ocr(cropped)
            
            if processed is not None:
                self.update_image_display(self.processed_preview, processed, max_size=(300, 200))
                raw_text, timer_text = self.perform_ocr(processed)
                self.ocr_result_var.set(raw_text if raw_text else "No text detected")
                self.timer_result_var.set(timer_text if timer_text else "No timer detected")
                
                if timer_text:
                    self.last_detected_time = timer_text
        
        self.dialog.after(100, self.update_preview)

    def update_image_display(self, label_widget, cv_image, max_size=(300, 200)):
        """Update a label widget with a CV2 image"""
        try:
            if len(cv_image.shape) == 3:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            img_w, img_h = pil_image.size
            max_w, max_h = max_size
            
            ratio = min(max_w / img_w, max_h / img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            
            display_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            label_widget.configure(image=photo, text="")
            label_widget._image_ref = photo
            
        except Exception as e:
            label_widget.configure(image="", text=f"Display error: {str(e)}")

    def perform_ocr(self, processed_image):
        """Perform OCR on processed image"""
        if processed_image is None or not self.tesseract_available:
            return "No image/Tesseract unavailable", ""
            
        try:
            config = self.tesseract_config_var.get()
            text = pytesseract.image_to_string(processed_image, config=config).strip()
            
            pattern = self.pattern_var.get()
            timer_match = re.search(pattern, text)
            timer_text = timer_match.group(0) if timer_match else ""
            
            return text, timer_text
        except Exception as e:
            return f"OCR Error: {str(e)}", ""

    def on_param_change(self, *args):
        """Called when any parameter changes"""
        self.scale_factor = float(self.scale_var.get())
        self.dilate_iterations = int(self.dilate_var.get())
        self.erode_iterations = int(self.erode_var.get())

    def manual_ocr_test(self):
        """Manually trigger OCR test"""
        cropped = self.get_perspective_corrected_crop()
        if cropped is not None:
            processed = self.process_image_for_ocr(cropped)
            if processed is not None:
                raw_text, timer_text = self.perform_ocr(processed)
                messagebox.showinfo("OCR Test", f"Raw OCR: {raw_text}\n\nParsed Timer: {timer_text}")

    def save_settings(self):
        """Save current OCR settings"""
        settings = {
            'selected_color_hsv': self.selected_color_hsv,  # Already a list now
            'color_ranges': [self.color_range_h, self.color_range_s, self.color_range_v],
            'scale_factor': self.scale_factor,
            'dilate_iterations': self.dilate_iterations,
            'erode_iterations': self.erode_iterations,
            'tesseract_config': self.tesseract_config_var.get(),
            'timer_pattern': self.pattern_var.get()
        }
        messagebox.showinfo("Settings", f"Settings saved:\n{settings}")

    def back_to_camera(self):
        """Go back to camera setup"""
        self.stop_dual_capture()
        self.dialog.destroy()
        # Import and restart camera setup
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
        """Finish the setup process"""
        if not self.last_detected_time:
            if not messagebox.askyesno("Warning", 
                "No timer has been detected yet. Continue anyway?"):
                return
        
        settings = {
            'camera_index': self.camera_index,
            'crop_corners': self.crop_corners,
            'selected_color_hsv': self.selected_color_hsv,  # Already a list now
            'color_ranges': [self.color_range_h, self.color_range_s, self.color_range_v],
            'scale_factor': self.scale_factor,
            'dilate_iterations': self.dilate_iterations,
            'erode_iterations': self.erode_iterations,
            'tesseract_config': self.tesseract_config_var.get(),
            'timer_pattern': self.pattern_var.get()
        }
        
        # Save to file
        try:
            import json
            with open('ocr_config.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo("Setup Complete", 
                f"OCR setup complete and saved!\n"
                f"Last detected time: {self.last_detected_time}\n"
                f"Selected color (HSV): {self.selected_color_hsv}\n"
                f"Config saved to: ocr_config.json")
        except Exception as e:
            messagebox.showwarning("Save Warning", 
                f"Setup complete but failed to save config: {str(e)}\n"
                f"Last detected time: {self.last_detected_time}")
        
        self.stop_dual_capture()
        self.dialog.destroy()

    def on_cancel(self):
        """Cancel OCR setup"""
        self.stop_dual_capture()
        self.dialog.destroy()


def run_ocr_setup(camera_index: int, crop_corners: List[Tuple[int, int]]) -> bool:
    """Run the OCR setup dialog with dual-threaded capture"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = OCRSetup(root, camera_index, crop_corners)
        root.wait_window(app.dialog)
        return True
    except Exception as e:
        messagebox.showerror("Error", f"OCR setup failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test with dummy data
    test_corners = [(100, 50), (300, 50), (300, 100), (100, 100)]
    run_ocr_setup(0, test_corners)