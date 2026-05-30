"""
Attention state classifier.

Combines posture and face signals to estimate the user's cognitive state:
  - flow: relaxed body, steady breathing, consistent blink rate, moderate head movement
  - stuck: low blink rate, high gaze fixation, neck/face tension
  - tension: high slouch + high mouth tension + irregular movement

Outputs a state string and confidence score.
"""

from typing import Literal

import numpy as np

from aaa.utils.helpers import clamp, smooth_value, lerp

AttentionState = Literal["flow", "stuck", "tension", "unknown"]


class AttentionEngine:
    def __init__(self):
        self._flow_confidence = 0.0
        self._stuck_confidence = 0.0
        self._tension_confidence = 0.0
        self.state: AttentionState = "unknown"
        self.primary_score: float = 0.0

    def update(
        self,
        posture_signals: dict,
        face_signals: dict,
        baseline: dict | None = None,
    ) -> dict:
        slouch = posture_signals.get("slouch_score", 0.0)
        neck_angle = posture_signals.get("neck_forward_angle", 0.0)
        blink_rate = face_signals.get("blink_rate", 15.0)
        gaze_fixation = face_signals.get("gaze_fixation", 0.5)
        mouth_tension = face_signals.get("mouth_tension", 0.3)
        head_yaw = abs(face_signals.get("head_yaw", 0.0))
        pose_detected = posture_signals.get("pose_detected", False)
        face_detected = face_signals.get("face_detected", False)

        if not pose_detected and not face_detected:
            self.state = "unknown"
            return self._build_result()

        b = baseline or {}

        flow_blink = 1.0 - clamp(abs(blink_rate - b.get("blink_rate", 15.0)) / 30.0)
        flow_posture = 1.0 - slouch
        flow_movement = clamp(head_yaw / 20.0)
        flow = flow_blink * 0.3 + flow_posture * 0.5 + flow_movement * 0.2

        stuck_fixation = gaze_fixation
        stuck_still = 1.0 - clamp(head_yaw / 10.0)
        stuck_blink = 1.0 - clamp(blink_rate / 10.0)
        stuck = stuck_fixation * 0.4 + stuck_still * 0.3 + stuck_blink * 0.3

        tension_slouch = slouch
        tension_jaw = mouth_tension
        tension_neck = clamp(neck_angle / 45.0)
        tension = tension_slouch * 0.4 + tension_jaw * 0.3 + tension_neck * 0.3

        self._flow_confidence = smooth_value(flow, self._flow_confidence, 0.15)
        self._stuck_confidence = smooth_value(stuck, self._stuck_confidence, 0.15)
        self._tension_confidence = smooth_value(tension, self._tension_confidence, 0.15)

        scores = {
            "flow": self._flow_confidence,
            "stuck": self._stuck_confidence,
            "tension": self._tension_confidence,
        }
        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score < 0.3:
            self.state = "unknown"
            self.primary_score = 0.0
        else:
            self.state = best
            self.primary_score = best_score

        return self._build_result()

    def _build_result(self) -> dict:
        return {
            "state": self.state,
            "confidence": self.primary_score,
            "flow_score": self._flow_confidence,
            "stuck_score": self._stuck_confidence,
            "tension_score": self._tension_confidence,
        }


def compute_baseline(posture_baseline: dict, face_baseline: dict) -> dict:
    return {**posture_baseline, **face_baseline}


def _check_brightness(frame: np.ndarray) -> tuple[str, float]:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 40:
        return "too_dark", mean_brightness
    elif mean_brightness > 220:
        return "too_bright", mean_brightness
    return "good", mean_brightness


def calibrate_all(duration: float = 15.0, camera_index: int = 0, ip_url: str | None = None) -> dict:
    """Calibrate posture + face in one camera session."""
    import platform
    import time
    import cv2

    if ip_url:
        cap = cv2.VideoCapture(ip_url)
    else:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        return compute_baseline(
            {"slouch_score": 0.3, "neck_forward_angle": 15.0, "spine_angle": 5.0, "shoulder_slope": 3.0},
            {"blink_rate": 15.0, "gaze_fixation": 0.5, "mouth_tension": 0.3},
        )

    from aaa.detectors.posture import PostureDetector
    from aaa.detectors.face import FaceDetector

    posture = PostureDetector()
    face = FaceDetector()

    window_name = "Calibration - sit naturally"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    # --- Pre-check phase: show pose / face / lighting status ---
    precheck_start = time.time()
    precheck_duration = 5.0

    while time.time() - precheck_start < precheck_duration:
        ret, frame = cap.read()
        if not ret:
            break

        p = posture.process(frame)
        f = face.process(frame)

        lighting_status, brightness = _check_brightness(frame)
        pose_ok = p["pose_detected"]
        face_ok = f["face_detected"]

        y = 40
        cv2.putText(frame, "Pre-check — adjust if needed:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y += 35

        if pose_ok:
            cv2.putText(frame, "  Pose: detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
        else:
            cv2.putText(frame, "  Pose: NOT detected — face the camera",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        y += 30

        if face_ok:
            cv2.putText(frame, "  Face: detected", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
        else:
            cv2.putText(frame, "  Face: NOT detected — face the camera",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        y += 30

        if lighting_status == "good":
            cv2.putText(frame, f"  Lighting: good ({brightness:.0f})", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
        elif lighting_status == "too_dark":
            cv2.putText(frame, "  Lighting: too dark — turn on lights",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        else:
            cv2.putText(frame, "  Lighting: too bright — reduce light",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        y += 40

        remaining = precheck_duration - (time.time() - precheck_start)
        cv2.putText(frame, f"Starting calibration in {remaining:.0f}s",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # --- Calibration phase ---
    posture_samples: dict[str, list] = {"slouch_score": [], "neck_forward_angle": [], "spine_angle": [], "shoulder_slope": []}
    face_samples: dict[str, list] = {"blink_rate": [], "gaze_fixation": [], "mouth_tension": [], "head_pitch": [], "head_yaw": [], "head_roll": []}
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Calibrating... sit naturally", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Time remaining: {duration - (time.time() - start):.0f}s",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        p = posture.process(frame)
        if p["pose_detected"]:
            for k in posture_samples:
                posture_samples[k].append(p[k])

        f = face.process(frame)
        if f["face_detected"]:
            for k in face_samples:
                face_samples[k].append(f[k])

    cv2.destroyWindow(window_name)
    posture.close()
    face.close()
    cap.release()

    pb = {k: float(np.mean(v)) if v else 0.0 for k, v in posture_samples.items()}
    fb = {k: float(np.mean(v)) if v else 0.0 for k, v in face_samples.items()}
    return compute_baseline(pb, fb)
