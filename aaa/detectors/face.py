"""
Face and eye tracking using MediaPipe Face Mesh.

Signals:
  - blink_rate: blinks per minute (smoothed)
  - head_pitch / head_yaw / head_roll: head rotation angles
  - gaze_fixation: 0-1 how fixated the gaze is
  - mouth_tension: proxy for jaw clenching
  - ear_left / ear_right: Eye Aspect Ratio for each eye
"""

import collections
import math
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from aaa.utils.helpers import smooth_value, clamp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 133, 157, 158, 159, 160, 161, 173]
RIGHT_EYE = [362, 263, 380, 381, 382, 383, 384, 385]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE_TIP = 1
NOSE_BRIDGE = 168
HEAD_TOP = 10
CHIN = 152


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self._ear_history: collections.deque = collections.deque(maxlen=10)
        self._blink_timestamps: list[float] = []
        self._blink_cooldown = 0.0
        self._prev_ear = 0.0
        self._nose_history: collections.deque = collections.deque(maxlen=30)
        self._frames_no_face = 0

    def _eye_aspect_ratio(self, landmarks, eye_indices) -> float:
        points = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        if horizontal < 1e-6:
            return 1.0
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def _head_pose(self, landmarks, img_w: int, img_h: int) -> tuple[float, float, float]:
        image_points = np.array([
            [landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y],
            [landmarks[CHIN].x, landmarks[CHIN].y],
            [landmarks[LEFT_EYE[0]].x, landmarks[LEFT_EYE[0]].y],
            [landmarks[RIGHT_EYE[0]].x, landmarks[RIGHT_EYE[0]].y],
            [landmarks[NOSE_BRIDGE].x, landmarks[NOSE_BRIDGE].y],
            [landmarks[234].x, landmarks[234].y],
        ], dtype=np.float64)
        image_points[:, 0] *= img_w
        image_points[:, 1] *= img_h

        model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-165.0, 170.0, -135.0],
            [165.0, 170.0, -135.0],
            [0.0, -100.0, 0.0],
            [-220.0, 0.0, -120.0],
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler = cv2.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, np.zeros((3, 1))))
        )[-1]
        pitch, yaw, roll = euler[0, 0], euler[1, 0], euler[2, 0]
        return float(pitch), float(yaw), float(roll)

    def process(self, frame: cv2.Mat, fps: float = 30.0) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.face_mesh.process(rgb)
        signals = {
            "blink_rate": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "gaze_fixation": 0.0,
            "mouth_tension": 0.0,
            "face_detected": False,
        }

        if not results.multi_face_landmarks:
            self._frames_no_face += 1
            return signals

        self._frames_no_face = 0
        landmarks = results.multi_face_landmarks[0].landmark

        ear_left = self._eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_right = self._eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0
        self._ear_history.append(ear)

        blink_threshold = 0.18
        now = time.time()
        if ear < blink_threshold and self._prev_ear >= blink_threshold:
            if now - self._blink_cooldown > 0.1:
                self._blink_timestamps.append(now)
                self._blink_cooldown = now
        self._prev_ear = ear

        blink_window = 30.0
        self._blink_timestamps = [t for t in self._blink_timestamps if now - t < blink_window]
        blink_rate = (len(self._blink_timestamps) / blink_window) * 60.0
        signals["blink_rate"] = blink_rate

        pitch, yaw, roll = self._head_pose(landmarks, w, h)
        signals["head_pitch"] = pitch
        signals["head_yaw"] = yaw
        signals["head_roll"] = roll

        nose_pos = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
        self._nose_history.append(nose_pos)
        if len(self._nose_history) >= 10:
            variance = float(np.var(self._nose_history, axis=0).mean())
            fixation = clamp(1.0 - (variance * 100.0))
            signals["gaze_fixation"] = fixation

        upper_lip = np.array([landmarks[13].x, landmarks[13].y])
        lower_lip = np.array([landmarks[14].x, landmarks[14].y])
        lip_dist = np.linalg.norm(upper_lip - lower_lip)
        tension = clamp(1.0 - (lip_dist * 5.0))
        signals["mouth_tension"] = tension
        signals["face_detected"] = True

        return signals

    def close(self):
        self.face_mesh.close()


def calibrate_face(duration: float = 30.0, camera_index: int = 0) -> dict:
    import time
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"blink_rate": 15.0, "gaze_fixation": 0.5, "mouth_tension": 0.3, "head_pitch": 0.0}

    detector = FaceDetector()
    samples = {"blink_rate": [], "gaze_fixation": [], "mouth_tension": [], "head_pitch": [], "head_yaw": [], "head_roll": []}
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        signals = detector.process(frame)
        if signals["face_detected"]:
            for key in samples:
                samples[key].append(signals[key])
        time.sleep(0.05)

    detector.close()
    cap.release()

    baseline = {}
    for key, vals in samples.items():
        baseline[key] = float(np.mean(vals)) if vals else 0.0
    return baseline
