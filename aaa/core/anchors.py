"""
Physical Anchors System.

An anchor links a physical pose/gesture to a file/module at a moment in time.
When the user revisits that file later, aaa gently reminds them to recreate the pose.

Data stored in ~/.aaa/anchors.json as a list of:
  {
    "pose": "hand on chin, lean left",
    "file": "/abs/path/to/file.py",
    "module": "aaa.detectors.posture",
    "timestamp": "2026-05-29T12:00:00",
    "project": "aaa",
    "event": "big_refactor" | "focus_session" | "manual",
  }
"""

import json
import time
from pathlib import Path
from typing import Optional

_AAA_DIR = Path.home() / ".aaa"
_ANCHORS_PATH = _AAA_DIR / "anchors.json"

POSE_SUGGESTIONS = [
    "hand on chin, lean left",
    "hand on chin, lean right",
    "both hands behind head",
    "crossed arms, lean back",
    "one hand up, palm open",
    "fist on desk, slight lean forward",
    "touch your left shoulder with right hand",
    "interlocked fingers behind neck",
    "palms on thighs, straight back",
    "stretch arms forward, clasp hands",
]


def _ensure_file():
    _AAA_DIR.mkdir(parents=True, exist_ok=True)
    if not _ANCHORS_PATH.exists():
        _ANCHORS_PATH.write_text("[]")


def load_anchors() -> list[dict]:
    _ensure_file()
    try:
        return json.loads(_ANCHORS_PATH.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_anchors(anchors: list[dict]):
    _ensure_file()
    _ANCHORS_PATH.write_text(json.dumps(anchors, indent=2))


def add_anchor(
    file_path: str,
    module_name: str = "",
    event: str = "manual",
    pose: Optional[str] = None,
) -> dict:
    anchors = load_anchors()
    import random
    anchor = {
        "pose": pose or random.choice(POSE_SUGGESTIONS),
        "file": file_path,
        "module": module_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project": Path(file_path).parent.name if file_path else "unknown",
        "event": event,
    }
    anchors.append(anchor)
    save_anchors(anchors)
    return anchor


def find_anchor_for_file(file_path: str) -> Optional[dict]:
    anchors = load_anchors()
    for anchor in reversed(anchors):
        if anchor.get("file") == file_path:
            return anchor
    for anchor in reversed(anchors):
        if anchor.get("module") and file_path and anchor["module"] in file_path:
            return anchor
    return None


def get_recent_anchors(limit: int = 10) -> list[dict]:
    anchors = load_anchors()
    return anchors[-limit:]


def count_anchors() -> int:
    return len(load_anchors())


def clear_anchors():
    save_anchors([])
