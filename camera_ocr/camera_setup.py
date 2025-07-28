import contextlib
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from typing import Optional, Tuple, List
import numpy as np
import json

from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph

class CameraSetup:
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.selected_index: Optional[int] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.preview_running = False
        self._preview_image = None
        
        # Camera resolution tracking
        self.camera_resolution = (640, 480)  # Will be updated when camera opens
        self.preview_resolution = (1280, 720)  # Lower resolution for smooth preview
        self.resolution_scale_factor = 1.0  # Scale factor from preview to full resolution

        self.zoom_level = 1.0
        self.pan_x = 0  # Pan offset in canvas pixels
        self.pan_y = 0  # Pan offset in canvas pixels
        
        # Image display properties
        self.image_scale = 1.0  # Current scale factor for the image
        self.image_width = 0    # Scaled image width
        self.image_height = 0   # Scaled image height
        self.image_x = 0        # Image position on canvas
        self.image_y = 0        # Image position on canvas

        # Configuration file path
        self.config_file = "camera_config.json"

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Camera Setup - Timer Crop")
        self.dialog.geometry("1400x900")
        self.dialog.minsize(400, 300)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        graph = FilterGraph()
        self.camera_names = graph.get_input_devices()
        self.camera_options = list(range(len(self.camera_names)))

        # Camera selection frame with resolution info
        camera_frame = ttk.Frame(self.dialog)
        camera_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.device_var = tk.StringVar()
        self.device_dropdown = ttk.Combobox(
            camera_frame,
            textvariable=self.device_var,
            state="readonly",
            width=30
        )
        self.device_dropdown["values"] = [
            f"{name} (Camera {i})" for i, name in enumerate(self.camera_names)
        ]
        
        # Set default selection (will be overridden by config if available)
        self.device_dropdown.current(0)
        self.device_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        # Resolution info label
        self.resolution_label = ttk.Label(camera_frame, text="Resolution: Unknown", foreground="blue")
        self.resolution_label.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.dialog, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # Mouse bindings for corner adjustment
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press_left)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release_left)

        # Pan with middle mouse button or right mouse button
        self.canvas.bind("<ButtonPress-2>", self.on_mouse_press_middle)
        self.canvas.bind("<B2-Motion>", self.on_mouse_drag_middle)
        self.canvas.bind("<ButtonRelease-2>", self.on_mouse_release_middle)
        self.canvas.bind("<ButtonPress-3>", self.on_mouse_press_middle)
        self.canvas.bind("<B3-Motion>", self.on_mouse_drag_middle)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_release_middle)

        self.canvas_size = (640, 360)
        self.drag_start = (0, 0)
        self.current_drag_end = None  # For live preview during creation

        # 4-corner crop system
        self.corner_points = None  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in original image coordinates
        self.canvas_corners = None  # Corner positions in canvas coordinates
        self.selected_corner = None  # Which corner is being dragged
        self.corner_radius = 8  # Radius for corner selection
        self.crop_mode = "none"  # "none", "creating", "adjusting"

        # Zoom controls
        zoom_frame = ttk.Frame(self.dialog)
        zoom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=1.0, to=10.0, value=1.0, orient="horizontal", command=self.on_zoom_slider)
        self.zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.RIGHT)

        instruction_text = ("Instructions:\n"
                          "1. Scroll or use slider to zoom, middle-click drag to pan\n"
                          "2. Left-click and drag to create initial timer crop area\n"
                          "3. Drag the corner circles to adjust perspective around timer\n"
                          "4. Preview shows perspective-corrected timer\n"
                          "5. Settings are automatically saved and restored")
        
        instructions = ttk.Label(self.dialog, text=instruction_text, foreground="gray", justify="left")
        instructions.pack(padx=10, pady=5)

        # Crop preview frame with fixed size
        crop_frame = ttk.LabelFrame(self.dialog, text="Timer Preview (Perspective Corrected)", padding=10)
        crop_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a frame with fixed dimensions for the preview
        preview_container = ttk.Frame(crop_frame, width=400, height=200)
        preview_container.pack_propagate(False)
        preview_container.pack()
        
        self.crop_preview = ttk.Label(preview_container, text="Draw crop area around timer", background="black", foreground="white")
        self.crop_preview.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Clear Crop", command=self.clear_crop).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Rectangle", command=self.reset_to_rectangle).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Config", command=self.load_and_apply_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Debug Resolution", command=self.debug_current_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Next: OCR Setup", command=self.on_next_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Select", command=self.on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=5)

        self.device_dropdown.bind("<<ComboboxSelected>>", lambda e: self.start_preview())
        
        # Load saved configuration after UI is fully set up
        saved_config = self.load_config()
        if saved_config:
            self.apply_saved_config(saved_config)
            
        self.start_preview()

    def setup_camera_max_resolution(self, camera_index):
        """Configure camera for smooth preview resolution and detect max resolution capability"""
        capture = cv2.VideoCapture(camera_index)
        
        if not capture.isOpened():
            return None, (640, 480), (640, 480), 1.0
        
        # First, detect the maximum supported resolution (without using it)
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
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                max_resolution = (actual_width, actual_height)
                print(f"Camera {camera_index} max resolution detected: {actual_width}x{actual_height}")
                break
        
        # Now set to a good preview resolution for smooth UI
        preview_targets = [
            (1280, 720),   # 720p - good balance
            (960, 540),    # Half of 1080p
            (800, 600),    # SVGA
            (640, 480)     # VGA fallback
        ]
        
        preview_resolution = (640, 480)
        
        for width, height in preview_targets:
            if width <= max_resolution[0] and height <= max_resolution[1]:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                    preview_resolution = (actual_width, actual_height)
                    print(f"Camera {camera_index} preview resolution: {actual_width}x{actual_height}")
                    break
        
        # Calculate scale factor from preview to max resolution
        scale_factor = max_resolution[0] / preview_resolution[0]
        
        # Optimize camera settings
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass
        
        return capture, max_resolution, preview_resolution, scale_factor

    def debug_current_camera(self):
        """Debug function to see current camera capabilities"""
        camera_index = self.camera_options[self.device_dropdown.current()]
        
        # Show current resolution
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                current_res = f"{frame.shape[1]}x{frame.shape[0]}"
            else:
                current_res = "Unable to read frame"
        else:
            current_res = "Camera not open"
        
        # Test capabilities
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            messagebox.showerror("Debug", f"Cannot open camera {camera_index}")
            return
        
        # Get default resolution
        default_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test various resolutions
        test_resolutions = [
            (4096, 2160), (3840, 2160), (2560, 1440), 
            (1920, 1080), (1280, 720), (800, 600), (640, 480)
        ]
        
        supported_resolutions = []
        for width, height in test_resolutions:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width > 0 and actual_height > 0:
                supported_resolutions.append(f"{width}x{height} -> {actual_width}x{actual_height}")
        
        capture.release()
        
        debug_info = (f"Camera {camera_index} Debug Info:\n\n"
                     f"Current Active Resolution: {current_res}\n"
                     f"Default Resolution: {default_width}x{default_height}\n\n"
                     f"Supported Resolutions:\n" + "\n".join(supported_resolutions))
        
        messagebox.showinfo("Camera Debug", debug_info)

    def start_preview(self):
        """Enhanced start_preview with resolution optimization"""
        self.stop_preview()
        index = self.camera_options[self.device_dropdown.current()]
        
        # Check if we have saved resolution settings
        saved_config = self.load_config()
        if saved_config and 'camera_resolution' in saved_config:
            # Use saved resolution directly - much faster!
            self.camera_resolution = tuple(saved_config['camera_resolution'])
            if 'preview_resolution' in saved_config:
                self.preview_resolution = tuple(saved_config['preview_resolution'])
            else:
                self.preview_resolution = (1280, 720)  # Default fallback
            
            if 'resolution_scale_factor' in saved_config:
                self.resolution_scale_factor = saved_config['resolution_scale_factor']
            else:
                self.resolution_scale_factor = self.camera_resolution[0] / self.preview_resolution[0]
            
            # Set up camera with saved settings
            self.capture = cv2.VideoCapture(index)
            if self.capture and self.capture.isOpened():
                # Directly set the preview resolution
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_resolution[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_resolution[1])
                print(f"Using saved camera resolution: {self.camera_resolution}")
                print(f"Using saved preview resolution: {self.preview_resolution}")
            else:
                messagebox.showerror("Error", f"Could not open camera {index}")
                return
        else:
            # No saved config - do full resolution detection (slower)
            result = self.setup_camera_max_resolution(index)
            if result[0] is None:
                messagebox.showerror("Error", f"Could not open camera {index}")
                self.resolution_label.config(text="Resolution: Error")
                return
            
            self.capture, self.camera_resolution, self.preview_resolution, self.resolution_scale_factor = result
        
        # Update resolution display
        if self.camera_resolution == self.preview_resolution:
            res_text = f"Resolution: {self.camera_resolution[0]}x{self.camera_resolution[1]}"
        else:
            res_text = f"Resolution: {self.camera_resolution[0]}x{self.camera_resolution[1]} (preview: {self.preview_resolution[0]}x{self.preview_resolution[1]})"
        
        self.resolution_label.config(text=res_text)
        
        self.preview_running = True
        self.update_preview()

    def stop_preview(self):
        self.preview_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def calculate_image_layout(self, frame):
        """Calculate how the image should be positioned and scaled on the canvas"""
        h, w = frame.shape[:2]
        canvas_w, canvas_h = self.canvas_size
        
        # Calculate base scale to fit image in canvas
        base_scale = min(canvas_w / w, canvas_h / h)
        
        # Apply zoom to the base scale
        self.image_scale = base_scale * self.zoom_level
        
        # Calculate the image dimensions at current zoom
        self.image_width = int(w * self.image_scale)
        self.image_height = int(h * self.image_scale)
        
        # Calculate image position (center by default, then apply pan)
        default_x = (canvas_w - self.image_width) // 2
        default_y = (canvas_h - self.image_height) // 2
        
        # Apply pan offset
        self.image_x = default_x + self.pan_x
        self.image_y = default_y + self.pan_y
        
        # Clamp pan to reasonable bounds (allow image to be dragged partially off screen)
        max_pan_x = self.image_width // 2
        max_pan_y = self.image_height // 2
        min_pan_x = -self.image_width + max_pan_x
        min_pan_y = -self.image_height + max_pan_y
        
        # Only clamp if image is larger than canvas
        if self.image_width > canvas_w:
            self.pan_x = max(min_pan_x, min(max_pan_x, self.pan_x))
        else:
            self.pan_x = 0  # Center horizontally if image fits
            
        if self.image_height > canvas_h:
            self.pan_y = max(min_pan_y, min(max_pan_y, self.pan_y))
        else:
            self.pan_y = 0  # Center vertically if image fits
            
        # Recalculate position with clamped pan
        self.image_x = default_x + self.pan_x
        self.image_y = default_y + self.pan_y

    def update_preview(self):
        if not self.preview_running or self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.dialog.after(30, self.update_preview)
            return

        self.latest_frame = frame
        self.original_frame = frame.copy()

        # Calculate image layout (position and scale)
        self.calculate_image_layout(frame)

        # Convert frame to RGB and create PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Resize image to display size
        resized_image = pil_image.resize((self.image_width, self.image_height), Image.BILINEAR)
        img_tk = ImageTk.PhotoImage(resized_image)
        self._preview_image = img_tk

        # Clear canvas and draw image
        self.canvas.delete("all")
        self.canvas.create_image(self.image_x, self.image_y, image=img_tk, anchor="nw")

        # Draw crop overlay and corners
        self.draw_crop_overlay()
        self.update_crop_preview()

        self.dialog.after(30, self.update_preview)

    def get_full_resolution_crop(self):
        """Get crop at full camera resolution (for OCR processing)"""
        if not self.corner_points:
            return None
        
        try:
            # Use a separate camera instance for full resolution capture
            temp_capture = cv2.VideoCapture(self.camera_options[self.device_dropdown.current()])
            if not temp_capture.isOpened():
                print("Could not open camera for full resolution capture")
                return None
            
            # Set to full resolution
            temp_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            temp_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            
            # Give camera time to adjust and capture a frame
            ret = False
            for _ in range(5):  # Try a few times to get a good frame
                ret, frame = temp_capture.read()
                if ret and frame is not None:
                    break
            
            temp_capture.release()
            
            if not ret or frame is None:
                print("Could not capture full resolution frame")
                return None
        
            print(f"Full resolution frame: {frame.shape[1]}x{frame.shape[0]}")
            
            # Use the stored corner points (already in full resolution coordinates)
            src_points = np.array(self.corner_points, dtype=np.float32)
            
            # Calculate output dimensions from the quadrilateral
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
            
            if max_width <= 0 or max_height <= 0:
                print("Invalid crop dimensions")
                return None
            
            # Define destination rectangle
            dst_points = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)
            
            # Apply perspective transformation at full resolution
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            corrected = cv2.warpPerspective(frame, matrix, (max_width, max_height))
            
            print(f"Cropped region: {corrected.shape[1]}x{corrected.shape[0]}")
            return corrected
            
        except Exception as e:
            print(f"Error in get_full_resolution_crop: {e}")
            return None

    def draw_crop_overlay(self):
        """Draw the 4-corner crop area and corner handles"""
        # If we're in creating mode, draw the preview rectangle
        if self.crop_mode == "creating" and hasattr(self, 'current_drag_end'):
            x1, y1 = self.drag_start
            x2, y2 = self.current_drag_end
            
            # Draw preview rectangle
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="lime", width=2, fill="", stipple="gray50"
            )
            return
        
        if not self.corner_points:
            return

        # Convert image coordinates to canvas coordinates
        canvas_corners = []
        for img_x, img_y in self.corner_points:
            canvas_coords = self.image_to_canvas_coords(img_x, img_y)
            if canvas_coords:
                canvas_corners.append(canvas_coords)
            else:
                return

        self.canvas_corners = canvas_corners

        # Draw crop area outline
        if len(canvas_corners) == 4:
            for i in range(4):
                x1, y1 = canvas_corners[i]
                x2, y2 = canvas_corners[(i + 1) % 4]
                self.canvas.create_line(x1, y1, x2, y2, fill="lime", width=2)

        # Draw corner handles
        for i, (x, y) in enumerate(canvas_corners):
            color = "red" if i == self.selected_corner else "yellow"
            self.canvas.create_oval(
                x - self.corner_radius, y - self.corner_radius,
                x + self.corner_radius, y + self.corner_radius,
                fill=color, outline="white", width=2
            )
            self.canvas.create_text(x, y - 15, text=str(i+1), fill="white", font=("Arial", 10, "bold"))

    def image_to_canvas_coords(self, img_x, img_y):
        """Convert full resolution image coordinates to canvas coordinates"""
        # Scale down from full resolution to preview resolution
        preview_x = img_x / self.resolution_scale_factor
        preview_y = img_y / self.resolution_scale_factor
        
        # Then scale from preview to canvas display size
        canvas_x = self.image_x + (preview_x * self.image_scale)
        canvas_y = self.image_y + (preview_y * self.image_scale)
        return (canvas_x, canvas_y)

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to original image coordinates (at full resolution)"""
        if not hasattr(self, 'image_scale') or self.image_scale == 0:
            return None
            
        # Convert from canvas to preview image coordinates
        preview_img_x = (canvas_x - self.image_x) / self.image_scale
        preview_img_y = (canvas_y - self.image_y) / self.image_scale
        
        # Scale up to full resolution coordinates
        full_img_x = preview_img_x * self.resolution_scale_factor
        full_img_y = preview_img_y * self.resolution_scale_factor
        
        # Clamp to full resolution image bounds
        full_img_x = max(0, min(self.camera_resolution[0] - 1, full_img_x))
        full_img_y = max(0, min(self.camera_resolution[1] - 1, full_img_y))
            
        return (int(full_img_x), int(full_img_y))

    def on_mouse_wheel(self, event):
        """Handle zoom with mouse wheel"""
        if not hasattr(self, "latest_frame"):
            return
            
        old_zoom = self.zoom_level
        delta = 0.1 if event.delta > 0 else -0.1
        new_zoom = max(1.0, min(10.0, old_zoom + delta))

        if new_zoom == old_zoom:
            return

        # Calculate zoom center point (mouse position in image coordinates)
        mouse_img_coords = self.canvas_to_image_coords(event.x, event.y)
        if mouse_img_coords:
            # Store the point we want to zoom towards
            zoom_center_img_x, zoom_center_img_y = mouse_img_coords
            
            # Calculate where this point is currently on the canvas
            old_canvas_x, old_canvas_y = self.image_to_canvas_coords(zoom_center_img_x, zoom_center_img_y)
            
            # Update zoom level
            self.zoom_level = new_zoom
            
            # Recalculate layout with new zoom
            self.calculate_image_layout(self.latest_frame)
            
            # Calculate where the zoom center point would be with new zoom
            new_canvas_x, new_canvas_y = self.image_to_canvas_coords(zoom_center_img_x, zoom_center_img_y)
            
            # Adjust pan to keep the zoom center point under the mouse
            self.pan_x += (old_canvas_x - new_canvas_x)
            self.pan_y += (old_canvas_y - new_canvas_y)
        else:
            # If mouse is outside image, just zoom normally
            self.zoom_level = new_zoom

        # Update zoom display
        zoom_percent = int(new_zoom * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")
        self.zoom_slider.set(new_zoom)

    def on_mouse_press_left(self, event):
        """Handle left mouse press for crop creation or corner adjustment"""
        # Check if clicking on existing corner
        if self.canvas_corners:
            for i, (cx, cy) in enumerate(self.canvas_corners):
                if self.point_distance(event.x, event.y, cx, cy) <= self.corner_radius:
                    self.selected_corner = i
                    self.crop_mode = "adjusting"
                    return

        # If no crop exists, start creating one
        if not self.corner_points:
            self.crop_mode = "creating"
            self.drag_start = (event.x, event.y)
            self.current_drag_end = (event.x, event.y)

    def on_mouse_drag_left(self, event):
        """Handle mouse drag for crop creation or corner adjustment"""
        if self.crop_mode == "creating":
            # Show live preview while creating crop
            self.current_drag_end = (event.x, event.y)
        elif self.crop_mode == "adjusting" and self.selected_corner is not None:
            # Adjust selected corner
            img_coords = self.canvas_to_image_coords(event.x, event.y)
            if img_coords:
                self.corner_points[self.selected_corner] = img_coords

    def on_mouse_release_left(self, event):
        """Handle mouse release for crop creation"""
        if self.crop_mode == "creating":
            # Create initial 4-corner rectangle from drag
            start_img = self.canvas_to_image_coords(self.drag_start[0], self.drag_start[1])
            end_img = self.canvas_to_image_coords(event.x, event.y)
            
            if start_img and end_img:
                x1, y1 = start_img
                x2, y2 = end_img
                
                # Ensure proper ordering
                min_x, max_x = min(x1, x2), max(x1, x2)
                min_y, max_y = min(y1, y2), max(y1, y2)
                
                # Create 4 corners in clockwise order
                self.corner_points = [
                    (min_x, min_y),  # top-left
                    (max_x, min_y),  # top-right
                    (max_x, max_y),  # bottom-right
                    (min_x, max_y)   # bottom-left
                ]
                self.crop_mode = "none"
                # Clear the drag preview
                self.current_drag_end = None
                # Auto-save when crop is created
                self.auto_save_on_changes()
        elif self.crop_mode == "adjusting":
            self.crop_mode = "none"
            # Auto-save when crop is adjusted
            self.auto_save_on_changes()
        
        self.selected_corner = None

    def on_mouse_press_middle(self, event):
        """Start panning operation"""
        self.drag_start = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def on_mouse_drag_middle(self, event):
        """Handle panning with middle/right mouse button"""
        if not hasattr(self, "drag_start"):
            return
            
        # Calculate how much the mouse moved
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.drag_start = (event.x, event.y)

        # Update pan offset
        self.pan_x += dx
        self.pan_y += dy

    def on_mouse_release_middle(self, event):
        """End panning operation"""
        self.canvas.config(cursor="")

    def point_distance(self, x1, y1, x2, y2):
        """Calculate distance between two points"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def update_crop_preview(self):
        """Update crop preview with perspective correction using preview resolution"""
        if not self.corner_points or not hasattr(self, "original_frame"):
            self.crop_preview.configure(image="", text="Draw crop area around timer")
            return

        try:
            # Convert full-resolution crop coordinates to preview resolution
            preview_corners = []
            for full_x, full_y in self.corner_points:
                preview_x = full_x / self.resolution_scale_factor
                preview_y = full_y / self.resolution_scale_factor
                preview_corners.append((preview_x, preview_y))
            
            src_points = np.array(preview_corners, dtype=np.float32)
            
            # Calculate output dimensions
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
            
            if max_width <= 0 or max_height <= 0:
                self.crop_preview.configure(image="", text="Invalid crop dimensions")
                return
            
            # Define destination points for a perfect rectangle
            dst_points = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transformation on preview resolution frame
            corrected = cv2.warpPerspective(
                self.original_frame, 
                matrix, 
                (max_width, max_height)
            )
            
            if corrected.size == 0:
                self.crop_preview.configure(image="", text="Invalid crop region")
                return
            
            # Convert BGR to RGB
            corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            pil_corrected = Image.fromarray(corrected_rgb)
            
            # Resize for preview maintaining aspect ratio
            crop_w, crop_h = pil_corrected.size
            max_w, max_h = 380, 180
            
            ratio = min(max_w / crop_w, max_h / crop_h)
            new_w = int(crop_w * ratio)
            new_h = int(crop_h * ratio)
            
            if new_w > 0 and new_h > 0:
                preview_image = pil_corrected.resize((new_w, new_h), Image.LANCZOS)
                crop_tk = ImageTk.PhotoImage(preview_image)
                
                self.crop_preview.configure(image=crop_tk, text="")
                self.crop_preview.image = crop_tk
            else:
                self.crop_preview.configure(image="", text="Invalid preview size")
            
        except Exception as e:
            self.crop_preview.configure(image="", text=f"Preview error: {str(e)}")
            print(f"Crop preview error: {e}")

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    print(f"Loaded config: {config}")
                    return config
        except Exception as e:
            print(f"Error loading config: {e}")
        return None

    def save_config(self):
        """Save current configuration to file"""
        try:
            config = {
                'camera_index': self.camera_options[self.device_dropdown.current()],
                'camera_name': self.camera_names[self.device_dropdown.current()],
                'camera_resolution': self.camera_resolution,
                'preview_resolution': self.preview_resolution,
                'resolution_scale_factor': self.resolution_scale_factor,
                'crop_corners': self.corner_points,
                'zoom_level': self.zoom_level,
                'pan_x': self.pan_x,
                'pan_y': self.pan_y
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Config Saved", f"Configuration saved to {self.config_file}")
            print(f"Saved config: {config}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save config: {str(e)}")

    def apply_saved_config(self, config):
        """Apply saved configuration to UI"""
        try:
            # Find matching camera by name first, fall back to index
            saved_camera_name = config.get('camera_name', '')
            saved_camera_index = config.get('camera_index', 0)
            
            # Try to find camera by name first (more reliable)
            camera_found = False
            for i, name in enumerate(self.camera_names):
                if name == saved_camera_name:
                    self.device_dropdown.current(i)
                    camera_found = True
                    print(f"Found camera by name: {name} at index {i}")
                    break
            
            # If not found by name, try by index
            if not camera_found and 0 <= saved_camera_index < len(self.camera_names):
                self.device_dropdown.current(saved_camera_index)
                camera_found = True
                print(f"Found camera by index: {saved_camera_index}")
            
            if not camera_found:
                print(f"Saved camera not found, using default")
                self.device_dropdown.current(0)
            
            # Apply crop corners if they exist
            if config.get('crop_corners'):
                self.corner_points = config['crop_corners']
                print(f"Restored crop corners: {self.corner_points}")
            
            # Apply zoom and pan settings (check if UI elements exist)
            if 'zoom_level' in config and hasattr(self, 'zoom_slider'):
                self.zoom_level = config['zoom_level']
                self.zoom_slider.set(self.zoom_level)
                if hasattr(self, 'zoom_label'):
                    zoom_percent = int(self.zoom_level * 100)
                    self.zoom_label.config(text=f"{zoom_percent}%")
            elif 'zoom_level' in config:
                # Store for later application
                self.zoom_level = config['zoom_level']
            
            if 'pan_x' in config:
                self.pan_x = config['pan_x']
            
            if 'pan_y' in config:
                self.pan_y = config['pan_y']
                
            # Show saved resolution if available
            saved_resolution = config.get('camera_resolution', 'Unknown')
            
            messagebox.showinfo("Config Loaded", 
                f"Loaded saved configuration:\n"
                f"Camera: {saved_camera_name}\n"
                f"Resolution: {saved_resolution[0]}x{saved_resolution[1]} (saved)\n"
                f"Crop: {'Yes' if self.corner_points else 'No'}\n"
                f"Zoom: {int(self.zoom_level * 100)}%")
                
        except Exception as e:
            print(f"Error applying config: {e}")
            messagebox.showwarning("Config Error", f"Could not fully apply saved config: {str(e)}")

    def load_and_apply_config(self):
        """Manually load and apply configuration"""
        config = self.load_config()
        if config:
            self.apply_saved_config(config)
            # Restart preview with new settings
            self.start_preview()
        else:
            messagebox.showinfo("No Config", "No saved configuration found.")

    def auto_save_on_changes(self):
        """Auto-save configuration when important changes are made"""
        if hasattr(self, 'corner_points') and self.corner_points:
            try:
                config = {
                    'camera_index': self.camera_options[self.device_dropdown.current()],
                    'camera_name': self.camera_names[self.device_dropdown.current()],
                    'camera_resolution': self.camera_resolution,
                    'preview_resolution': self.preview_resolution,
                    'resolution_scale_factor': self.resolution_scale_factor,
                    'crop_corners': self.corner_points,
                    'zoom_level': self.zoom_level,
                    'pan_x': self.pan_x,
                    'pan_y': self.pan_y
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                print("Auto-saved configuration")
                
            except Exception as e:
                print(f"Auto-save failed: {e}")

    def clear_crop(self):
        """Clear the current crop selection"""
        self.corner_points = None
        self.canvas_corners = None
        self.selected_corner = None
        self.crop_mode = "none"
        self.crop_preview.configure(image="", text="Draw crop area around timer")

    def reset_to_rectangle(self):
        """Reset crop to a perfect rectangle based on current corners"""
        if not self.corner_points:
            return
        
        xs = [p[0] for p in self.corner_points]
        ys = [p[1] for p in self.corner_points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        self.corner_points = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ]
        
        # Auto-save when crop is reset
        self.auto_save_on_changes()

    def on_next_ocr(self):
        """Proceed to OCR setup with current camera and crop settings"""
        if not self.corner_points:
            messagebox.showwarning("No Crop", "Please create a crop area around the timer first.")
            return
            
        camera_index = self.camera_options[self.device_dropdown.current()]
        crop_corners = self.corner_points.copy()  # Make a copy
        self.stop_preview()
        self.dialog.destroy()
        
        # Import and run OCR setup with the same parent
        try:
            from .robust_timer_ocr import RobustTimerOCRSetup
        
            # Create OCR setup with the same parent root
            ocr_app = RobustTimerOCRSetup(self.parent, camera_index, crop_corners)
            self.parent.wait_window(ocr_app.dialog)
            
        except ImportError as e:
            messagebox.showerror("Error", f"ocr_setup.py not found: {str(e)}\nPlease make sure it's in the same directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start OCR setup: {str(e)}")

    def on_select(self):
        self.selected_index = self.camera_options[self.device_dropdown.current()]
        self.stop_preview()
        self.dialog.destroy()
        
        crop_info = ""
        if self.corner_points:
            crop_info = f"\nTimer crop corners: {self.corner_points}"
        messagebox.showinfo("Camera Selected", f"Camera {self.selected_index} selected.\nResolution: {self.camera_resolution[0]}x{self.camera_resolution[1]}{crop_info}")

    def on_cancel(self):
        self.selected_index = None
        self.stop_preview()
        self.dialog.destroy()

    def on_canvas_resize(self, event):
        if event.width < 50 or event.height < 50:
            return
        self.canvas_size = (event.width, event.height)

    def on_zoom_slider(self, value):
        self.zoom_level = float(value)
        zoom_percent = int(self.zoom_level * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")


def run_camera_setup() -> Optional[int]:
    root = tk.Tk()
    root.withdraw()
    app = CameraSetup(root)
    root.wait_window(app.dialog)
    return app.selected_index


if __name__ == "__main__":
    selected = run_camera_setup()
    if selected is not None:
        print(f"[INFO] Camera {selected} selected.")
    else:
        print("[INFO] Camera selection cancelled.")