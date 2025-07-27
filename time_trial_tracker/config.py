import os
import json
from typing import Any

# Define the path to the assets folder (relative to this file)
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
CROP_CONFIG_FILE = os.path.join(ASSETS_DIR, "crop_config.json")
GLOBAL_STATS_FILE = os.path.join(ASSETS_DIR, "global_stats.json")

# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_crop_config() -> dict[str, int] | None:
    """Load the crop region config from JSON"""
    if not os.path.exists(CROP_CONFIG_FILE):
        return None
    try:
        with open(CROP_CONFIG_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[Warning] Corrupted crop config file. Ignoring.")
        return None

def save_crop_config(config: dict[str, int]) -> None:
    """Save the crop region config to JSON"""
    with open(CROP_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def load_global_stats() -> dict[str, Any]:
    """Load the global statistics from JSON"""
    if not os.path.exists(GLOBAL_STATS_FILE):
        with open(GLOBAL_STATS_FILE, "w") as f:
            json.dump({"track_stats": {}}, f)
    try:
        with open(GLOBAL_STATS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[Warning] Corrupted global stats file. Starting fresh.")
        return {"track_stats": {}}

def save_global_stats(stats: dict[str, Any]) -> None:
    """Save the global statistics to JSON"""
    with open(GLOBAL_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)