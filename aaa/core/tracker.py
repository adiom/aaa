#!/usr/bin/env python3
"""
Main daemon entry point.

When launched by `aaa start`, this script runs the engine loop
in a background process. It handles signal-based shutdown.

Usage:
    python -m aaa.core.tracker [--headless] [--camera N]
"""

import argparse
import os
import signal
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

_engine = None


def _handle_signal(signum, frame):
    import os
    if _engine:
        _engine.shutdown()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without system tray")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--ip-cam", type=str, default="", help="IP Webcam URL")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    from aaa.core.engine import Engine
    global _engine
    _engine = Engine(camera_index=args.camera, ip_url=args.ip_cam or None)

    if not args.headless:
        from PySide6 import QtWidgets, QtCore
        import threading

        app = QtWidgets.QApplication(sys.argv)

        from aaa.ui.tray import TrayIcon
        from aaa.ui.overlay import HudWindow

        tray = TrayIcon()
        tray.update_status("running")
        hud = HudWindow()

        engine_thread = threading.Thread(target=_engine.run, daemon=True)
        engine_thread.start()

        app.exec()
    else:
        _engine.run()


if __name__ == "__main__":
    main()
