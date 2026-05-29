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


def calibrate_all(duration: float = 15.0, camera_index: int = 0, ip_url: str | None = None) -> dict:
    """Calibrate posture + face in one camera session."""
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
    posture_samples: dict[str, list] = {"slouch_score": [], "neck_forward_angle": [], "spine_angle": [], "shoulder_slope": []}
    face_samples: dict[str, list] = {"blink_rate": [], "gaze_fixation": [], "mouth_tension": [], "head_pitch": [], "head_yaw": [], "head_roll": []}
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        p = posture.process(frame)
        if p["pose_detected"]:
            for k in posture_samples:
                posture_samples[k].append(p[k])

        f = face.process(frame)
        if f["face_detected"]:
            for k in face_samples:
                face_samples[k].append(f[k])

        time.sleep(0.05)

    posture.close()
    face.close()
    cap.release()

    pb = {k: float(np.mean(v)) if v else 0.0 for k, v in posture_samples.items()}
    fb = {k: float(np.mean(v)) if v else 0.0 for k, v in face_samples.items()}
    return compute_baseline(pb, fb)
