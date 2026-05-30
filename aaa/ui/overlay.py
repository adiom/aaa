"""
Mini HUD overlay — shows real-time camera/pose/face/state status in a small
frameless window that stays on top.
"""

import json
import threading
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

_STATUS_PATH = Path.home() / ".aaa" / "state.json"

_status_lock = threading.Lock()
_cached_status: dict = {}

GESTURE_RU = {
    "nod": "Кивок",
    "tilt_left": "Наклон ←",
    "tilt_right": "Наклон →",
}


def _read_status():
    global _cached_status
    try:
        if _STATUS_PATH.exists():
            with _STATUS_PATH.open() as f:
                data = json.load(f)
                with _status_lock:
                    _cached_status = data.get("current", {})
    except Exception:
        pass
    return _cached_status


class HudWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(260, 170)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.width() - 280, 20)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

        self._font = QtGui.QFont("SF Mono", 11, QtGui.QFont.Weight.Bold)
        self._font.setFamilies(["SF Mono", "Menlo", "Monaco", "Courier New", "monospace"])
        self._small_font = QtGui.QFont("SF Mono", 9)
        self._small_font.setFamilies(["SF Mono", "Menlo", "Monaco", "Courier New", "monospace"])

        self._last_gesture = ""
        self._gesture_fade = 0.0

        self.show()

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(15, 15, 30, 200))
        p.drawRoundedRect(self.rect(), 12, 12)

        p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 140), 1))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect(), 12, 12)

        status = _read_status()

        camera = bool(status)
        pose = status.get("pose_detected", False)
        face = status.get("face_detected", False)
        state = status.get("state", "unknown")
        last_gesture = status.get("last_gesture", "")

        y = 14
        self._draw_dot(p, 16, y, camera, "cam")
        p.setFont(self._font)
        p.setPen(QtGui.QColor("#E0E0E0"))
        p.drawText(32, y + 4, "aaa")

        y += 24
        self._draw_dot(p, 16, y, pose, "pose")
        p.setFont(self._small_font)
        p.setPen(self._color_for(state))
        p.drawText(32, y + 4, self._state_label(status))

        y += 22
        self._draw_dot(p, 16, y, face, "face")
        p.setFont(self._small_font)
        p.setPen(QtGui.QColor("#909090"))
        p.drawText(32, y + 4, f"морг: {status.get('blink_rate', 0):.0f}/мин")

        y += 22
        self._draw_dot(p, 16, y, state == "tension", "tension")
        p.setFont(self._small_font)
        p.setPen(QtGui.QColor("#909090"))
        p.drawText(32, y + 4, f"сутул: {status.get('slouch_score', 0):.2f}")

        y += 22
        self._draw_dot(p, 16, y, state == "flow", "flow")
        p.setFont(self._small_font)
        p.setPen(QtGui.QColor("#909090"))
        p.drawText(32, y + 4, f"взгляд: {status.get('gaze_fixation', 0):.2f}")

        if last_gesture:
            label = GESTURE_RU.get(last_gesture, last_gesture)
            self._gesture_fade = 1.0

        if self._gesture_fade > 0.01:
            p.setFont(self._small_font)
            p.setPen(QtGui.QColor(100, 255, 100, int(255 * self._gesture_fade)))
            label = GESTURE_RU.get(last_gesture, last_gesture or "")
            if label:
                p.drawText(32, y + 22, f"жест: {label}")

    def _draw_dot(self, p: QtGui.QPainter, x: int, y: int, active: bool, kind: str):
        colors = {
            "cam": ("#4ADE80", "#1A4A2E"),
            "pose": ("#4ADE80", "#1A4A2E"),
            "face": ("#4ADE80", "#1A4A2E"),
            "tension": ("#F97316", "#4A1E00"),
            "flow": ("#A78BFA", "#2D1B69"),
        }
        on_c, off_c = colors.get(kind, ("#888", "#333"))
        c = QtGui.QColor(on_c if active else off_c)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(x - 5, y - 2, 10, 10)

    def _color_for(self, state: str) -> QtGui.QColor:
        return {
            "flow": QtGui.QColor("#A78BFA"),
            "stuck": QtGui.QColor("#F97316"),
            "tension": QtGui.QColor("#EF4444"),
        }.get(state, QtGui.QColor("#888888"))

    def _state_label(self, s: dict) -> str:
        ru = {"flow": "поток", "stuck": "завис", "tension": "напряжение"}
        state = s.get("state", "unknown")
        conf = s.get("confidence", 0)
        label = ru.get(state, state)
        return f"{label} ({conf:.2f})"

    def _tick(self):
        self._gesture_fade *= 0.98
        self.update()


def start_hud():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    HudWindow()
    app.exec()
