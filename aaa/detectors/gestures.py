"""
Gesture detection using raw facial landmarks.

Detects:
  - Nod: nose_y rises (look down) then falls (look up)
  - Tilt left/right: sustained eye_angle deviation from level
"""

import logging
import time
from collections import deque

from aaa.utils.helpers import smooth_value

log = logging.getLogger("aaa.gestures")


class GestureDetector:
    def __init__(self):
        self._nose_y_history = deque(maxlen=20)
        self._nose_y_avg = 0.5
        self._eye_angle_history = deque(maxlen=10)
        self._smoothed_nose_y = 0.5
        self._smoothed_eye_angle = 0.0

        self._nod_timer = 0.0
        self._nod_cooldown = 2.0
        self._nod_went_down = False
        self._nod_baseline = 0.0

        self._tilt_timer = 0.0
        self._tilt_cooldown = 1.5
        self._tilt_active = "neutral"

        self._frame_count = 0
        self.last_gesture: str | None = None
        self.last_gesture_time: float = 0.0

    def process(self, nose_y: float, eye_angle: float) -> dict:
        now = time.time()
        result = {
            "nod": False,
            "tilt_left": False,
            "tilt_right": False,
            "gesture": None,
        }

        if nose_y <= 0:
            return result

        self._smoothed_nose_y = smooth_value(nose_y, self._smoothed_nose_y, 0.2)
        self._smoothed_eye_angle = smooth_value(eye_angle, self._smoothed_eye_angle, 0.25)

        self._nose_y_history.append(self._smoothed_nose_y)
        self._eye_angle_history.append(self._smoothed_eye_angle)
        self._frame_count += 1

        # --- Nod: nose_y relative to running average ---
        if len(self._nose_y_history) >= 10:
            self._nose_y_avg = sum(self._nose_y_history) / len(self._nose_y_history)

        if now - self._nod_timer > self._nod_cooldown and len(self._nose_y_history) >= 5:
            dev = self._smoothed_nose_y - self._nose_y_avg

            if not self._nod_went_down:
                if dev > 0.02:
                    self._nod_went_down = True
                    self._nod_baseline = self._nose_y_avg
                    log.debug("Nod start: dev=%.4f, nose_y=%.4f, avg=%.4f", dev, self._smoothed_nose_y, self._nose_y_avg)
            else:
                if dev < 0.005:
                    total = self._smoothed_nose_y - self._nod_baseline
                    log.debug("Nod end: dev=%.4f, total=%.4f", dev, total)
                    if total > 0.03:
                        self._nod_timer = now
                        self.last_gesture = "nod"
                        self.last_gesture_time = now
                        result["nod"] = True
                        result["gesture"] = "nod"
                        log.info("Gesture: nod")
                    self._nod_went_down = False

        # --- Tilt: eye_angle sustained deviation ---
        if now - self._tilt_timer > self._tilt_cooldown and len(self._eye_angle_history) >= 5:
            recent = list(self._eye_angle_history)[-5:]
            mean_angle = sum(recent) / len(recent)

            if mean_angle > 8 and all(a > 3 for a in recent) and self._tilt_active != "right":
                self._tilt_active = "right"
                self._tilt_timer = now
                self.last_gesture = "tilt_right"
                self.last_gesture_time = now
                result["tilt_right"] = True
                result["gesture"] = "tilt_right"
                log.info("Gesture: tilt_right (eye_angle=%.1f)", mean_angle)

            elif mean_angle < -8 and all(a < -3 for a in recent) and self._tilt_active != "left":
                self._tilt_active = "left"
                self._tilt_timer = now
                self.last_gesture = "tilt_left"
                self.last_gesture_time = now
                result["tilt_left"] = True
                result["gesture"] = "tilt_left"
                log.info("Gesture: tilt_left (eye_angle=%.1f)", mean_angle)

            elif abs(mean_angle) < 4:
                self._tilt_active = "neutral"

        if self._frame_count % 300 == 0:
            log.debug("nose_y=%.4f avg=%.4f eye_angle=%.1f", self._smoothed_nose_y, self._nose_y_avg, self._smoothed_eye_angle)

        return result
