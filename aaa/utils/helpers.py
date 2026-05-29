import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import psutil

_EDITOR_KEYWORDS = [
    "code", "vim", "nvim", "emacs", "pycharm", "intellij", "webstorm",
    "sublime", "atom", "xcode", "android studio", "zed", "helix",
    "neovide", "cursor", "windsurf",
]


def get_active_window() -> Optional[str]:
    system = platform.system()
    try:
        if system == "Darwin":
            script = (
                'tell application "System Events" to get name of first process '
                'whose frontmost is true'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=3,
            )
            return result.stdout.strip() or None
        elif system == "Windows":
            import pygetwindow as gw
            active = gw.getActiveWindow()
            return active.title if active else None
        elif system == "Linux":
            try:
                import Xlib.display
                display = Xlib.display.Display()
                window = display.get_input_focus().focus
                name = window.get_wm_name()
                return name.decode("utf-8") if isinstance(name, bytes) else name
            except Exception:
                return None
    except Exception:
        return None
    return None


def window_is_editor(title: Optional[str]) -> bool:
    if not title:
        return False
    title_lower = title.lower()
    for kw in _EDITOR_KEYWORDS:
        if kw in title_lower:
            return True
    return False


def is_session_active() -> bool:
    now = time.time()
    for proc in psutil.process_iter(["create_time"]):
        try:
            if now - proc.info["create_time"] < 5.0:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def smooth_value(new: float, old: float, alpha: float = 0.3) -> float:
    return alpha * new + (1.0 - alpha) * old


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def safe_camera_open(
    device_index: int = 0,
    max_retries: int = 3,
    ip_url: str | None = None,
) -> Optional[cv2.VideoCapture]:
    if ip_url:
        cap = cv2.VideoCapture(ip_url)
        if cap is not None and cap.isOpened():
            return cap
        return None

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(device_index, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass
        time.sleep(0.5)
    return None


def get_file_mod_times(path: Path) -> dict[str, float]:
    if not path.is_dir():
        return {}
    mtimes = {}
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".java", ".rb", ".lua"}:
            mtimes[str(f)] = f.stat().st_mtime
    return mtimes


def detect_file_changes(previous: dict[str, float], current: dict[str, float]) -> list[str]:
    changed = []
    for path, mtime in current.items():
        if path not in previous:
            changed.append(path)
        elif mtime != previous.get(path):
            changed.append(path)
    return changed
