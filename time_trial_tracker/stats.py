from datetime import datetime
from time_trial_tracker.config import save_global_stats
from time_trial_tracker.utils import parse_time, format_time
from datetime import timedelta
from typing import Any

def start_new_attempt(global_stats: dict[str, Any], track_name: str) -> tuple[int, dict[str, str]]:
    """Create and record a new attempt entry"""
    global_id = global_stats["global_attempt_id"]

    if track_name not in global_stats["track_stats"]:
        global_stats["track_stats"][track_name] = {
            "best_laps": {},
            "attempts": []
        }

    global_stats["track_stats"][track_name]["attempts"].append({
        "global_id": global_id,
        "datetime": datetime.now().isoformat()
    })

    global_stats["global_attempt_id"] += 1
    save_global_stats(global_stats)

    best_laps = global_stats["track_stats"][track_name]["best_laps"]
    return global_id, best_laps

def update_best_lap(global_stats: dict[str, Any], track_name: str, lap_label: str, lap_time: str) -> None:
    """Update best lap time for the current track if better"""
    track_data = global_stats["track_stats"][track_name]
    best = track_data["best_laps"].get(lap_label)

    if best is None or lap_time < best:
        track_data["best_laps"][lap_label] = lap_time

    save_global_stats(global_stats)

def update_global_stats_with_lap_times(
    global_stats: dict[str, Any], 
    track_name: str, 
    attempt_id: int, 
    lap_times: list[str]
) -> None:
    """Save lap times for a completed attempt"""
    track_data = global_stats["track_stats"].setdefault(track_name, {"best_laps": {}, "attempts": []})

    for attempt in track_data["attempts"]:
        if attempt["global_id"] == attempt_id:
            attempt["laps"] = lap_times
            break

    save_global_stats(global_stats)

def calculate_lap3_from_final(lap1: str, lap2: str, final: str) -> str | None:
    """Back-calculate Lap 3 using total minus Lap 1 and Lap 2"""
    td1 = parse_time(lap1)
    td2 = parse_time(lap2)
    td_final = parse_time(final)

    if td1 and td2 and td_final:
        td3 = td_final - td1 - td2
        return format_time(td3)
    return None
