import cv2
import os
import json
import time
import pytesseract
import numpy as np
import re
from datetime import datetime
from datetime import timedelta

# ---------- CONFIG ----------
CROP_CONFIG_PATH = "crop_config.json"
GLOBAL_STATS_FILE = "global_stats.json"
STABLE_FRAME_THRESHOLD = 3
FINAL_STABLE_SECONDS = 2.0
LAP_FLASH_COUNT_REQUIRED = 3
LAP_COOLDOWN_SECONDS = 5

# ---------- TRACK SELECTION ----------
tracks = [
    "Mario Bros. Circuit", "Crown City", "Whistlestop Summit", "DK Spaceport", "Desert Hills",
    "Shy Guy Bazaar", "Wario Stadium", "Airship Fortress", "DK Pass", "Starview Peak",
    "Sky-High Sundae", "Wario Shipyard", "Koopa Troopa Beach", "Faraway Oasis", "Peach Stadium",
    "Peach Beach", "Salty Salty Speedway", "Dino Dino Jungle", "Great ? Block Ruins", "Cheep Cheep Falls",
    "Dandelion Depths", "Boo Cinema", "Dry Bones Burnout", "Moo Moo Meadows", "Choco Mountain",
    "Toad's Factory", "Bowser's Castle", "Acorn Heights", "Mario Circuit", "Rainbow Road"
]

# ---------- UTILITIES ------------
def choose_track():
    print("Select the track you're playing:\n")
    for i, name in enumerate(tracks):
        print(f"{i + 1:2d}. {name}")
    while True:
        try:
            choice = int(input("\nEnter track number: "))
            if 1 <= choice <= len(tracks):
                return tracks[choice - 1]
            else:
                print("Invalid number, try again.")
        except ValueError:
            print("Please enter a number.")
        
# ---------- CROP SELECTION ----------
def manual_crop_selection(camera_index=1):
    zoom_factor = 2.5  # You can adjust this for more/less zoom

    cap = cv2.VideoCapture(camera_index)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read from camera")

    # Zoom in for better selection accuracy
    zoomed = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
    r = cv2.selectROI("Select Timer Region (Zoomed In)", zoomed, False, False)
    cv2.destroyAllWindows()

    # Convert zoomed selection back to original scale
    return {
        "x1": int(r[0] / zoom_factor),
        "y1": int(r[1] / zoom_factor),
        "x2": int((r[0] + r[2]) / zoom_factor),
        "y2": int((r[1] + r[3]) / zoom_factor)
    }

def load_crop_config():
    if os.path.exists(CROP_CONFIG_PATH):
        with open(CROP_CONFIG_PATH, "r") as f:
            return json.load(f)
    return None

def save_crop_config(config):
    with open(CROP_CONFIG_PATH, "w") as f:
        json.dump(config, f)

# ---------- TIME PARSING ----------
def is_valid_time(text):
    return bool(re.match(r"^\d+:\d{2}\.\d{3}$", text))

# ---------- GLOBAL STATS -----------
def load_global_stats():
    if os.path.exists(GLOBAL_STATS_FILE):
        with open(GLOBAL_STATS_FILE, "r") as f:
            return json.load(f)
    return {"global_attempt_id": 1, "track_stats": {}}

def save_global_stats(stats):
    with open(GLOBAL_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def start_new_attempt(track_name):
    stats = load_global_stats()
    global_id = stats["global_attempt_id"]

    if track_name not in stats["track_stats"]:
        stats["track_stats"][track_name] = {
            "best_laps": {},  # Per-lap bests
            "attempts": []
        }

    # Track new attempt
    stats["track_stats"][track_name]["attempts"].append({
        "global_id": global_id,
        "datetime": datetime.now().isoformat()
    })

    stats["global_attempt_id"] += 1
    save_global_stats(stats)

    return global_id, stats["track_stats"][track_name]["best_laps"]

def update_best_lap(track_name, lap_label, lap_time):
    stats = load_global_stats()
    best = stats["track_stats"][track_name]["best_laps"].get(lap_label)
    if best is None or lap_time < best:
        stats["track_stats"][track_name]["best_laps"][lap_label] = lap_time
    save_global_stats(stats)

def update_global_stats_with_lap_times(track_name, attempt_id, lap_times):
    stats = load_global_stats()
    track_data = stats["track_stats"].setdefault(track_name, {"best_laps": {}, "attempts": []})
    for attempt in track_data["attempts"]:
        if attempt["global_id"] == attempt_id:
            attempt["laps"] = lap_times
            break
    save_global_stats(stats)


# ---------- MAIN LAP TIMER ----------
def main():
    crop = load_crop_config()
    use_existing = crop and input("Use existing crop? (y/n): ").strip().lower() == "y"
    if not use_existing:
        crop = manual_crop_selection()
        save_crop_config(crop)
        print(f"Using crop: x={crop['x1']}:{crop['x2']}, y={crop['y1']}:{crop['y2']}")

    track_name = choose_track()
    cap = cv2.VideoCapture(1)

    lap_times = []
    attempt_id, best_lap_time = start_new_attempt(track_name)
    current_attempt_id = attempt_id

    print(f"\n>>> Attempt {attempt_id} Started (Best Lap: {best_lap_time or 'N/A'})")

    race_ready = False
    race_active = False
    last_time_seen = ""
    stable_frame_count = 0
    last_time_change = time.time()
    last_lap_logged_time = 0
    flash_counter = {}
    prev_flash = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        config = '--psm 7 -c tessedit_char_whitelist=0123456789.:'
        time_text = pytesseract.image_to_string(thresh, config=config).strip()

        mask = thresh.astype(bool)
        masked_pixels = roi[mask]
        avg_color = np.mean(masked_pixels, axis=0) if masked_pixels.size else [0, 0, 0]
        r, g, b = avg_color

        is_flash = (r < 200 and g > 160 and b > 180 and abs(r - g) > 30 and abs(g - b) > 30)
        flash_started = is_flash and not prev_flash
        prev_flash = is_flash

        now = time.time()
        stable_frame_count = stable_frame_count + 1 if time_text == last_time_seen else 1
        if time_text != last_time_seen:
            last_time_seen = time_text
            last_time_change = now

        is_stable = stable_frame_count >= STABLE_FRAME_THRESHOLD
        stable_time_duration = now - last_time_change

        if is_valid_time(time_text):
            if time_text == "0:00.000":
                race_ready = True
                race_active = False
            elif race_ready and not race_active:
                if lap_times:
                    update_global_stats_with_lap_times(track_name, current_attempt_id, lap_times)
                race_active = True
                race_ready = False
                lap_times = []
                flash_counter = {}
                last_lap_logged_time = 0
                current_attempt_id, best_lap_time = start_new_attempt(track_name)  # ← Use the returned ID
                print(f"\n>>> Attempt {current_attempt_id} Started")

            if race_active:
                if flash_started:
                    flash_counter[time_text] = flash_counter.get(time_text, 0) + 1

                if flash_counter.get(time_text, 0) == LAP_FLASH_COUNT_REQUIRED:
                    if len(lap_times) == 0 or time_text != lap_times[-1]:
                        if now - last_lap_logged_time >= LAP_COOLDOWN_SECONDS:
                            lap_label = f"Lap {len(lap_times) + 1}"
                            prev_best_laps = load_global_stats()["track_stats"].get(track_name, {}).get("best_laps", {})
                            prev_best = prev_best_laps.get(lap_label)
                            
                            lap_times.append(time_text)
                            
                            if prev_best:
                                def parse(t): return timedelta(minutes=int(t.split(":")[0]), seconds=int(t.split(":")[1].split(".")[0]), milliseconds=int(t.split(".")[1]))
                                prev_td = parse(prev_best)
                                curr_td = parse(time_text)
                                delta_td = curr_td - prev_td
                                sign = "+" if delta_td.total_seconds() >= 0 else "-"
                                print(f">>> Delta vs old best for {lap_label}: {sign}{abs(delta_td.total_seconds()):.3f}s")

                            update_best_lap(track_name, lap_label, time_text)
                            last_lap_logged_time = now
                            print(f">>> Attempt {current_attempt_id} - {lap_label}: {time_text}")

                if not is_flash and stable_time_duration > FINAL_STABLE_SECONDS:
                    if len(lap_times) > 0 and time_text != lap_times[-1]:
                        final_time_str = time_text
                        lap_times.append(final_time_str)
                        update_best_lap(track_name, "Final", final_time_str)

                        if len(lap_times) == 3:
                            try:
                                def parse(t): return timedelta(minutes=int(t.split(":")[0]), seconds=int(t.split(":")[1].split(".")[0]), milliseconds=int(t.split(".")[1]))
                                def fmt(t): return f"{t.seconds//60}:{t.seconds%60:02d}.{int(t.microseconds/1000):03d}"
                                lap3 = fmt(parse(final_time_str) - parse(lap_times[0]) - parse(lap_times[1]))
                                update_best_lap(track_name, "Lap 3", lap3)
                                print(f">>> Attempt {current_attempt_id} - Lap 3 (calculated): {lap3}")
                            except Exception as e:
                                print(f"Lap 3 calc failed: {e}")

                        update_global_stats_with_lap_times(track_name, current_attempt_id, lap_times)
                        print(f">>> Attempt {current_attempt_id} - Lap {len(lap_times)} (Final): {final_time_str}")
                        print(f"\n>>> Attempt {current_attempt_id} Finished")
                        race_active = False
                        race_ready = False

        # GUI
        cv2.imshow("Lap Timer - Cropped Timer View", roi)
        cv2.imshow("Lap Timer - Thresholded Timer", thresh)

        debug_panel = np.zeros((500, 600, 3), dtype=np.uint8)
        def draw_debug_text(text, line, color=(255, 255, 255)): cv2.putText(debug_panel, text, (10, 30 + line*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_debug_text(f"OCR: {time_text}", 0)
        draw_debug_text(f"RGB: ({int(r)}, {int(g)}, {int(b)})", 1)
        draw_debug_text(f"Flash: {'YES' if is_flash else 'no'}", 2, (0, 255, 0) if is_flash else (0, 0, 255))
        draw_debug_text(f"Stable Frame Count: {stable_frame_count}", 3)
        draw_debug_text(f"Lap Count: {len(lap_times)}", 4)

        best_laps = load_global_stats()["track_stats"].get(track_name, {}).get("best_laps", {})

        draw_debug_text("Lap      Best Time     Current Time     Lap Delta", 5, (0, 255, 255))
        current_line = 6

        def parse_time(t):
            try:
                minutes, rest = t.split(":")
                seconds, millis = rest.split(".")
                return timedelta(minutes=int(minutes), seconds=int(seconds), milliseconds=int(millis))
            except:
                return None

        def format_time(td):
            return f"{td.seconds // 60}:{td.seconds % 60:02d}.{int(td.microseconds / 1000):03d}"

        current_laps = lap_times.copy()
        if len(current_laps) == 3:  # Final time is third
            lap1_td = parse_time(current_laps[0])
            lap2_td = parse_time(current_laps[1])
            final_td = parse_time(current_laps[2])
            if lap1_td and lap2_td and final_td:
                lap3_td = final_td - lap1_td - lap2_td
                current_laps.insert(2, format_time(lap3_td))

        for i, label in enumerate(["Lap 1", "Lap 2", "Lap 3", "Final"]):
            best = best_laps.get(label)
            best_display = best or "--:--.---"

            current = current_laps[i] if i < len(current_laps) else ""
            delta = ""
            delta_color = (255, 255, 255)
            current_color = (200, 200, 200)

            if best and current:
                best_td = parse_time(best)
                curr_td = parse_time(current)
                if best_td and curr_td:
                    diff = curr_td - best_td
                    seconds = round(diff.total_seconds(), 3)
                    sign = "+" if seconds >= 0 else "-"
                    delta_val = f"{sign}{abs(seconds):.3f}s"
                    delta_color = (0, 0, 255) if seconds > 0 else (0, 255, 0)
                    delta = delta_val
                    if seconds < 0:
                        current_color = (0, 255, 255)

            y = 30 + current_line * 30
            cv2.putText(debug_panel, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_panel, best_display, (120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if current:
                cv2.putText(debug_panel, current, (290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

            if delta:
                cv2.putText(debug_panel, delta, (440, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, delta_color, 2)

            current_line += 1

        cv2.imshow("Lap Timer Debug Info", debug_panel)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if race_active and lap_times:
                update_global_stats_with_lap_times(track_name, current_attempt_id, lap_times)
                print(f"\n>>> Attempt {current_attempt_id} ended early, saving partial laps")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
