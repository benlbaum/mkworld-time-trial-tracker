import contextlib
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from typing import Optional

from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph

class CameraSetup:
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.selected_index: Optional[int] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.preview_running = False
        self._preview_image = None
        self._needs_redraw = False

        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Camera Setup")
        self.dialog.geometry("1200x900")
        self.dialog.minsize(400, 300)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        graph = FilterGraph()
        self.camera_names = graph.get_input_devices()  # List of camera names
        self.camera_options = list(range(len(self.camera_names)))  # List of corresponding indices

        self.device_var = tk.StringVar()
        self.device_dropdown = ttk.Combobox(
            self.dialog,
            textvariable=self.device_var,
            state="readonly",
            width=30  # adjust this value for width; default is ~20
        )
        self.device_dropdown["values"] = [
            f"{name} (Camera {i})" for i, name in enumerate(self.camera_names)
        ]
        self.device_dropdown.current(0)
        self.device_dropdown.pack(padx=10, pady=10)

        self.camera_aspect_ratio = 4 / 3

        self.canvas = tk.Canvas(self.dialog, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)

        self.canvas_size = (640, 360)
        self.drag_start = (0, 0)

        self.zoom_slider = ttk.Scale(self.dialog, from_=1.0, to=3.0, value=1.0, orient="horizontal", command=self.on_zoom_slider)
        self.zoom_slider.pack(fill=tk.X, padx=10, pady=5)

        zoom_hint = ttk.Label(self.dialog, text="Scroll or use slider to zoom. Click and drag to pan.", foreground="gray")
        zoom_hint.pack()

        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Select", command=self.on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=5)

        self.device_dropdown.bind("<<ComboboxSelected>>", lambda e: self.start_preview())
        self.start_preview()

    def validate_camera_indices(self, names: list[str]) -> list[int]:
        """Optional: confirm camera indices work with OpenCV"""
        valid_indices = []
        for i in range(len(names)):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.read()[0]:
                valid_indices.append(i)
            cap.release()
        return valid_indices

    def start_preview(self):
        self.stop_preview()
        index = self.camera_options[self.device_dropdown.current()]
        self.capture = cv2.VideoCapture(index)
        self.preview_running = True
        self.update_preview()

    def stop_preview(self):
        self.preview_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def update_preview(self):
        if not self.preview_running or self.capture is None:
            return

        # Only grab a new frame if not zooming/panning
        if not self._needs_redraw:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.dialog.after(30, self.update_preview)
                return
            self.latest_frame = frame
        else:
            frame = getattr(self, "latest_frame", None)

        if frame is not None:
            h, w = frame.shape[:2]
            if not hasattr(self, "camera_aspect_ratio_initialized"):
                self.camera_aspect_ratio = w / h
                self.camera_aspect_ratio_initialized = True

            frame = self.apply_zoom_and_pan(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)

            canvas_w, canvas_h = self.canvas_size
            img_w, img_h = pil_image.size
            ratio = min(canvas_w / img_w, canvas_h / img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))
            resized_image = pil_image.resize(new_size, Image.BILINEAR)  # Faster than LANCZOS for real-time

            img_tk = ImageTk.PhotoImage(resized_image)
            self._preview_image = img_tk

            x = (canvas_w - new_size[0]) // 2
            y = (canvas_h - new_size[1]) // 2
            self.canvas.delete("all")
            self.canvas.create_image(x, y, image=img_tk, anchor="nw")

        self._needs_redraw = False
        self.dialog.after(30, self.update_preview)


    def apply_zoom_and_pan(self, frame):
        h, w = frame.shape[:2]
        zoomed_w, zoomed_h = int(w / self.zoom_level), int(h / self.zoom_level)

        max_x = w - zoomed_w
        max_y = h - zoomed_h

        x1 = min(max(self.pan_x, 0), max_x)
        y1 = min(max(self.pan_y, 0), max_y)
        x2 = x1 + zoomed_w
        y2 = y1 + zoomed_h

        # Save corrected values in case they were clamped
        self.pan_x = x1
        self.pan_y = y1

        return frame[y1:y2, x1:x2]


    def on_mouse_wheel(self, event):
        old_zoom = self.zoom_level
        delta = 0.1 if event.delta > 0 else -0.1
        new_zoom = max(1.0, min(5.0, old_zoom + delta))

        if new_zoom == old_zoom or not hasattr(self, "latest_frame"):
            return

        # Get actual image size (full resolution)
        frame_h, frame_w = self.latest_frame.shape[:2]

        # Get canvas and image display size
        canvas_w, canvas_h = self.canvas_size
        visible_w = int(frame_w / old_zoom)
        visible_h = int(frame_h / old_zoom)

        # Determine image position in canvas (for centering)
        scale_ratio = min(canvas_w / visible_w, canvas_h / visible_h)
        display_w = int(visible_w * scale_ratio)
        display_h = int(visible_h * scale_ratio)
        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2

        # Get mouse position relative to image on canvas
        mouse_x = event.x - offset_x
        mouse_y = event.y - offset_y

        if mouse_x < 0 or mouse_y < 0 or mouse_x >= display_w or mouse_y >= display_h:
            # Outside image bounds — zoom without changing pan
            self.zoom_level = new_zoom
            self._needs_redraw = True
            return

        # Map canvas mouse position to image coordinate (before zoom)
        rel_x = mouse_x / display_w
        rel_y = mouse_y / display_h
        image_x = self.pan_x + int(rel_x * visible_w)
        image_y = self.pan_y + int(rel_y * visible_h)

        # Calculate new pan so image_x/image_y stays under cursor after zoom
        new_visible_w = int(frame_w / new_zoom)
        new_visible_h = int(frame_h / new_zoom)
        self.pan_x = image_x - int(rel_x * new_visible_w)
        self.pan_y = image_y - int(rel_y * new_visible_h)

        # Clamp pan
        self.pan_x = max(0, min(self.pan_x, frame_w - new_visible_w))
        self.pan_y = max(0, min(self.pan_y, frame_h - new_visible_h))

        self.zoom_level = new_zoom
        self.zoom_slider.set(new_zoom)
        self._needs_redraw = True

    def on_mouse_press(self, event):
        self.drag_start = (event.x, event.y)

    def on_mouse_drag(self, event):
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.drag_start = (event.x, event.y)

        if not hasattr(self, "latest_frame"):
            return

        frame_h, frame_w = self.latest_frame.shape[:2]
        visible_w = int(frame_w / self.zoom_level)
        visible_h = int(frame_h / self.zoom_level)

        # Scale drag distance by visible image size vs canvas size
        canvas_w, canvas_h = self.canvas_size
        scale_x = visible_w / canvas_w
        scale_y = visible_h / canvas_h

        # Apply scaled pan movement
        self.pan_x -= int(dx * scale_x)
        self.pan_y -= int(dy * scale_y)

        # Clamp to bounds
        self.pan_x = max(0, min(self.pan_x, frame_w - visible_w))
        self.pan_y = max(0, min(self.pan_y, frame_h - visible_h))

        self._needs_redraw = True

    def on_select(self):
        self.selected_index = self.camera_options[self.device_dropdown.current()]
        self.stop_preview()
        self.dialog.destroy()
        messagebox.showinfo("Camera Selected", f"Camera {self.selected_index} selected.")

    def on_cancel(self):
        self.selected_index = None
        self.stop_preview()
        self.dialog.destroy()

    def on_canvas_resize(self, event):
        if event.width < 50 or event.height < 50:
            return
        self.canvas_size = (event.width, event.height)
        if not hasattr(self, "_resize_pending"):
            self._resize_pending = True
            self.dialog.after(100, self._finish_resize)

    def _finish_resize(self):
        self._resize_pending = False

    def on_zoom_slider(self, value):
        self.zoom_level = float(value)
        self._needs_redraw = True



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