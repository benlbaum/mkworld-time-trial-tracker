import re
from datetime import timedelta

def is_valid_time(text: str) -> bool:
    """Check if text is in valid lap time format, e.g. 1:23.456"""
    return bool(re.match(r"^\d+:\d{2}\.\d{3}$", text))

def parse_time(time_str: str) -> timedelta | None:
    """Convert 'M:SS.mmm' string into a timedelta object"""
    try:
        minutes, rest = time_str.split(":")
        seconds, millis = rest.split(".")
        return timedelta(minutes=int(minutes), seconds=int(seconds), milliseconds=int(millis))
    except Exception:
        return None

def format_time(td: timedelta) -> str:
    """Format a timedelta object into 'M:SS.mmm'"""
    return f"{td.seconds // 60}:{td.seconds % 60:02d}.{int(td.microseconds / 1000):03d}"

def calculate_delta(best_time: str, current_time: str) -> str:
    """Return +/– delta in seconds between two time strings"""
    best_td = parse_time(best_time)
    curr_td = parse_time(current_time)
    if best_td and curr_td:
        diff = curr_td - best_td
        seconds = round(diff.total_seconds(), 3)
        sign = "+" if seconds >= 0 else ""
        return f"{sign}{seconds:.3f}s"
    return ""
