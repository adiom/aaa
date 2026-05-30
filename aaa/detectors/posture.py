"""
Posture detection using MediaPipe Pose landmarks.

Landmarks used (COCO topology):
  11 — left shoulder
  12 — right shoulder
  23 — left hip
  24 — right hip
   0 — nose

Computed signals:
  - shoulder_slope: angle of shoulder line relative to horizontal
  - neck_forward_angle: forward head protraction
  - spine_angle: lateral lean of the spine
  - slouch_score: composite 0-1 (0 = perfect, 1 = severe slouch)
  - chest_height: proxy for breathing / chest expansion
"""

import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from aaa.utils.helpers import smooth_value, clamp

_MODEL_PATH = os.path.expanduser("~/.aaa/models/pose_landmarker_heavy.task")


class PostureDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.core import base_options as base_options_lib

        base = base_options_lib.BaseOptions(model_asset_path=_MODEL_PATH)
        options = PoseLandmarkerOptions(
            base_options=base,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.pose = PoseLandmarker.create_from_options(options)
        self.slouch_score: float = 0.0
        self.shoulder_slope: float = 0.0
        self.neck_forward_angle: float = 0.0
        self.spine_angle: float = 0.0
        self.chest_height: float = 0.0
        self._frames_since_detection = 0
        self._max_frames_no_detection = 30

    def process(self, frame: cv2.Mat) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)
        signals = {
            "slouch_score": 0.0,
            "shoulder_slope": 0.0,
            "neck_forward_angle": 0.0,
            "spine_angle": 0.0,
            "chest_height": 0.0,
            "pose_detected": False,
        }

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            self._frames_since_detection += 1
            if self._frames_since_detection > self._max_frames_no_detection:
                signals["slouch_score"] = 1.0
            return signals

        self._frames_since_detection = 0
        lm = result.pose_landmarks[0]

        left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
        right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
        left_hip = np.array([lm[23].x, lm[23].y, lm[23].z])
        right_hip = np.array([lm[24].x, lm[24].y, lm[24].z])
        nose = np.array([lm[0].x, lm[0].y, lm[0].z])

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = (left_hip + right_hip) / 2.0

        shoulder_vector = right_shoulder - left_shoulder
        self.shoulder_slope = float(np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0])))
        signals["shoulder_slope"] = self.shoulder_slope

        spine_vector = shoulder_center - hip_center
        self.spine_angle = float(np.degrees(np.arctan2(spine_vector[0], spine_vector[1])))
        signals["spine_angle"] = abs(self.spine_angle)

        neck_ref = np.array([0.0, -1.0, 0.0])
        neck_vector = nose - shoulder_center
        if np.linalg.norm(neck_vector) > 0.001:
            neck_unit = neck_vector / np.linalg.norm(neck_vector)
            cos_angle = np.clip(np.dot(neck_unit, neck_ref), -1.0, 1.0)
            self.neck_forward_angle = float(np.degrees(np.arccos(cos_angle)))
        signals["neck_forward_angle"] = self.neck_forward_angle

        slouch = clamp(
            (abs(self.spine_angle) / 30.0) * 0.5
            + (self.neck_forward_angle / 45.0) * 0.3
            + (abs(self.shoulder_slope) / 20.0) * 0.2
        )
        self.slouch_score = smooth_value(slouch, self.slouch_score, alpha=0.2)
        signals["slouch_score"] = self.slouch_score

        self.chest_height = smooth_value(
            float(lm[11].y + lm[12].y) / 2.0, self.chest_height, alpha=0.1
        )
        signals["chest_height"] = self.chest_height
        signals["pose_detected"] = True

        return signals

    def close(self):
        self.pose.close()


def calibrate_posture(duration: float = 30.0, camera_index: int = 0) -> dict:
    """Run a calibration, return baseline posture signals."""
    import time
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"slouch_score": 0.3, "neck_forward_angle": 15.0, "spine_angle": 5.0, "shoulder_slope": 3.0}

    detector = PostureDetector()
    samples = {"slouch_score": [], "neck_forward_angle": [], "spine_angle": [], "shoulder_slope": []}
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        signals = detector.process(frame)
        if signals["pose_detected"]:
            for key in samples:
                samples[key].append(signals[key])
        time.sleep(0.05)

    detector.close()
    cap.release()

    baseline = {}
    for key, vals in samples.items():
        baseline[key] = float(np.mean(vals)) if vals else 0.0
    return baseline
