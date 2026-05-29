import random
import time
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

_OVERLAY_DURATION = 5.0
_FADE_DURATION = 1.0
_ACCENT_COLOR = QtGui.QColor(108, 99, 255)

def _mono_font(size: int, bold: bool = False) -> QtGui.QFont:
    f = QtGui.QFont("SF Mono", size)
    f.setFamilies(["SF Mono", "Menlo", "Monaco", "Courier New", "monospace"])
    if bold:
        f.setWeight(QtGui.QFont.Weight.Bold)
    return f


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, text: str, subtext: str = "", accent_color: QtGui.QColor = _ACCENT_COLOR):
        super().__init__()
        self._start_time = time.time()
        self._accent = accent_color

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QtWidgets.QApplication.primaryScreen()
        self._geo = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 800, 600)
        self.setGeometry(self._geo)

        self._opacity = 0.0
        self._text = text
        self._subtext = subtext

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

        self.setWindowOpacity(0.0)
        self.show()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center = self._geo.center()
        elapsed = time.time() - self._start_time

        if elapsed < 1.0:
            self._opacity = elapsed
        elif elapsed > _OVERLAY_DURATION - _FADE_DURATION:
            self._opacity = max(0.0, (_OVERLAY_DURATION - elapsed) / _FADE_DURATION)
        else:
            self._opacity = 1.0

        self.setWindowOpacity(min(1.0, self._opacity * 1.5))

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(20, 18, 40, int(180 * self._opacity)))
        r = QtCore.QRect(center.x() - 320, center.y() - 100, 640, 200)
        painter.drawRoundedRect(r, 24, 24)

        painter.setPen(QtGui.QPen(self._accent, 2))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(r, 24, 24)

        painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF")))
        title_font = _mono_font(20, bold=True)
        painter.setFont(title_font)
        painter.drawText(r, QtCore.Qt.AlignmentFlag.AlignCenter, self._text)

        if self._subtext:
            painter.setPen(QtGui.QPen(QtGui.QColor("#C0C0C0")))
            sub_font = _mono_font(12)
            painter.setFont(sub_font)
            sub_r = QtCore.QRect(r.x(), r.y() + 55, r.width(), r.height() - 60)
            painter.drawText(sub_r, QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop, self._subtext)

    def _tick(self):
        elapsed = time.time() - self._start_time
        if elapsed > _OVERLAY_DURATION:
            self._timer.stop()
            self.close()
        else:
            self.update()


class ScreenDarkenOverlay(QtWidgets.QWidget):
    def __init__(self, opacity: float = 0.12):
        super().__init__()
        self._target_opacity = opacity
        self._current_opacity = 0.0
        self._fading_out = False

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.availableGeometry())

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(20)
        self.setWindowOpacity(0.0)
        self.show()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        alpha = int(255 * self._current_opacity * 0.55)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, alpha))

        h = self.height()
        w = self.width()
        fade = int(h * 0.12)
        for i in range(fade):
            t = i / fade
            a = int(100 * (1.0 - t) * self._current_opacity)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, a)))
            painter.drawLine(0, i, int(w * 0.12), i)
            painter.drawLine(int(w * 0.88), i, w, i)

    def _tick(self):
        if self._fading_out:
            self._current_opacity = max(0.0, self._current_opacity - 0.02)
            if self._current_opacity <= 0.0:
                self._timer.stop()
                self.close()
        else:
            self._current_opacity = min(self._target_opacity, self._current_opacity + 0.015)
            if self._current_opacity >= self._target_opacity:
                QtCore.QTimer.singleShot(3500, self._fade_out)
                self._timer.stop()
        self.update()

    def _fade_out(self):
        self._fading_out = True
        self._timer.start(20)


def show_intervention(title: str, subtext: str, darken: bool = False):
    if darken:
        ScreenDarkenOverlay()
    OverlayWindow(title, subtext)


def show_slouch_warning():
    ScreenDarkenOverlay()
    OverlayWindow("⟐  straighten  ⟐", "your body is your instrument", accent_color=QtGui.QColor(255, 140, 0))


def show_tension_break():
    movements = [
        "rotate your wrists 3x",
        "clench and release your jaw",
        "roll your shoulders back",
        "look at something 20 feet away",
        "interlock fingers behind your head",
        "shake your hands for 10 seconds",
        "stand up, arms above head, breathe deep",
    ]
    OverlayWindow(f"⏾  micro-break  ⏾", random.choice(movements), accent_color=QtGui.QColor(0, 200, 200))


def show_anchor_prompt(file_path: str):
    short = file_path.split("/")[-1] if "/" in file_path else file_path
    OverlayWindow(f"⚮  physical anchor  ⚮", f"'{short}' — strike a pose and hold it 5s", accent_color=QtGui.QColor(108, 99, 255))


def show_anchor_reminder(anchor: dict):
    pose = anchor.get("pose", "your anchor pose")
    file_name = anchor.get("file", "").split("/")[-1]
    OverlayWindow(f"⌇  recall  ⌇", f"recreate '{pose}' for '{file_name}' (3-5s)", accent_color=QtGui.QColor(200, 100, 255))
