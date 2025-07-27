import cv2
import numpy as np
import pytesseract
from time_trial_tracker.utils import is_valid_time

def process_frame(
    frame: np.ndarray,
    crop_config: dict[str, int]
) -> tuple[str, np.ndarray, np.ndarray, str, dict[str, float]]:
    """
    Crop, preprocess, and extract timer text from the frame.

    Returns:
        time_text: str — the detected lap time
        roi_resized: np.ndarray — cropped and enlarged region of interest
        best_thresh: np.ndarray — thresholded image used for OCR
        method_used: str — threshold method name
        debug_info: dict — metrics for flash detection
    """
    crop = crop_config
    roi = frame[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]

    if roi.size == 0:
        return "", roi, roi, "Invalid ROI", {}

    roi_resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    # Thresholding methods
    _, thresh1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh4 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    thresholds = [thresh1, thresh2, thresh3, thresh4]
    threshold_names = ["Fixed", "Adaptive", "OTSU", "Inverted"]
    config = '--psm 7 -c tessedit_char_whitelist=0123456789.:'

    best_time_text = ""
    best_thresh = thresh1
    best_method = "Fixed"
    time_results = []

    for i, thresh in enumerate(thresholds):
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            if is_valid_time(text):
                best_time_text = text
                best_thresh = thresh
                best_method = threshold_names[i]
                break
            elif text:
                time_results.append((text, thresh, threshold_names[i]))
        except Exception:
            continue

    if not best_time_text and time_results:
        best_time_text, best_thresh, best_method = time_results[0]

    if not best_time_text:
        best_time_text = pytesseract.image_to_string(thresh1, config=config).strip()
        best_thresh = thresh1
        best_method = "Fixed (fallback)"

    # Flash detection
    debug_info = {}
    text_mask = best_thresh > 127
    if np.any(text_mask):
        text_pixels = roi_resized[text_mask].astype(np.float32)
        mean_rgb = np.mean(text_pixels, axis=0)
        r, g, b = mean_rgb[:3]  # Just in case it's longer than 3 values

        cyan_balance = abs(g - b)
        red_suppression = (g + b) / 2 - r

        is_flash = (
            cyan_balance < 20 and red_suppression > 80 and g > 180 and b > 180
        )

        debug_info = {
            "r": r,
            "g": g,
            "b": b,
            "cyan_balance": cyan_balance,
            "red_suppression": red_suppression,
            "is_flash": float(is_flash),
        }

    return best_time_text, roi_resized, best_thresh, best_method, debug_info
