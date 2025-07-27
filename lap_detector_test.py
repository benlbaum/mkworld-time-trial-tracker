import cv2
import json
import time
import pytesseract
import numpy as np
import re
import os

# ----- CONFIG -----
CROP_CONFIG_PATH = "crop_config.json"
STABLE_FRAME_THRESHOLD = 3
LAP_FLASH_COUNT_REQUIRED = 3
FINAL_TIME_HOLD_SECONDS = 2.0
LAP_COOLDOWN_SECONDS = 5.0
CAMERA_INDEX = 2

def load_crop_config():
    if os.path.exists(CROP_CONFIG_PATH):
        with open(CROP_CONFIG_PATH, "r") as f:
            return json.load(f)
    return None

def is_valid_time(text):
    return bool(re.match(r"^\d+:\d{2}\.\d{3}$", text))

def main():
    crop = load_crop_config()
    if not crop:
        print("No crop config found.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    prev_flash = False
    last_time_seen = ""
    flash_count = 0
    stable_frame_count = 0
    last_time_change = time.time()
    last_lap_detected_time = 0

    print("Testing lap detection on live video stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        roi = frame[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        config = '--psm 7 -c tessedit_char_whitelist=0123456789.:'
        time_text = pytesseract.image_to_string(thresh, config=config).strip()

        # Flash detection logic
        mask = thresh.astype(bool)
        masked_pixels = roi[mask]
        avg_color = np.mean(masked_pixels, axis=0) if masked_pixels.size else [0, 0, 0]
        r, g, b = avg_color

        is_flash = (
            r < 200 and g > 160 and b > 180 and
            abs(r - g) > 30 and abs(g - b) > 30
        )
        flash_started = is_flash and not prev_flash
        prev_flash = is_flash

        if not is_valid_time(time_text):
            continue

        if time_text == last_time_seen:
            stable_frame_count += 1
        else:
            stable_frame_count = 1
            last_time_change = now
            flash_count = 0
        last_time_seen = time_text
        is_stable = stable_frame_count >= STABLE_FRAME_THRESHOLD

        if flash_started and is_stable:
            flash_count += 1

        if (flash_count >= LAP_FLASH_COUNT_REQUIRED and
                (now - last_lap_detected_time > LAP_COOLDOWN_SECONDS)):
            print(f"LAP DETECTED: {time_text}")
            last_lap_detected_time = now
            flash_count = 0

        if (now - last_time_change > FINAL_TIME_HOLD_SECONDS and
                (now - last_lap_detected_time > LAP_COOLDOWN_SECONDS)):
            print(f"FINAL TIME DETECTED: {time_text}")
            last_lap_detected_time = now

        overlay = roi.copy()
        cv2.putText(overlay, f"OCR: {time_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(overlay, f"FLASH: {'YES' if is_flash else 'no'}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_flash else (0, 0, 255), 1)
        cv2.putText(overlay, f"Flashes: {flash_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Lap Detection Test", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
