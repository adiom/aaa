"""
PySide6 system tray icon for aaa.

Provides quick access to start/stop, status, calibration, and quit.
Launched from the daemon process in a background thread.
"""

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

_tray_instance: Optional["TrayIcon"] = None


def get_tray() -> Optional["TrayIcon"]:
    return _tray_instance


class TrayIcon(QtWidgets.QSystemTrayIcon):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        global _tray_instance
        _tray_instance = self
        self._setup_icon()
        self._setup_menu()
        self.activated.connect(self._on_activated)
        self.show()

    def _setup_icon(self):
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setIcon(QtGui.QIcon(str(icon_path)))
        else:
            pixmap = QtGui.QPixmap(64, 64)
            pixmap.fill(QtGui.QColor("#6C63FF"))
            self.setIcon(QtGui.QIcon(pixmap))

    def _setup_menu(self):
        menu = QtWidgets.QMenu()

        self.status_action = menu.addAction("Status: unknown")
        self.status_action.setEnabled(False)

        menu.addSeparator()

        self.start_action = menu.addAction("Start")

        self.stop_action = menu.addAction("Stop")
        self.stop_action.setEnabled(False)

        self.calibrate_action = menu.addAction("Calibrate")

        menu.addSeparator()

        self.anchors_action = menu.addAction("Anchors...")

        menu.addSeparator()

        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(QtWidgets.QApplication.quit)

        self.setContextMenu(menu)

    def _on_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason):
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self.status_action.setText("Status: running")

    def update_status(self, text: str):
        self.status_action.setText(f"Status: {text}")
        if text == "running":
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(True)
        elif text == "stopped":
            self.start_action.setEnabled(True)
            self.stop_action.setEnabled(False)

    def notify(self, title: str, message: str, duration_ms: int = 3000):
        icon = QtWidgets.QSystemTrayIcon.MessageIcon.Information
        self.showMessage(title, message, icon, duration_ms)


def start_tray():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    tray = TrayIcon()
    tray.update_status("running")

    app.exec()
