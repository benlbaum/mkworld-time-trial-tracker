from .camera_setup import run_camera_setup

if __name__ == "__main__":
    selected = run_camera_setup()
    if selected is not None:
        print(f"[INFO] Camera {selected} selected.")
    else:
        print("[INFO] Camera selection cancelled.")