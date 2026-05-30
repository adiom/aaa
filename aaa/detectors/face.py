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
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from aaa.utils.helpers import smooth_value, clamp

_MODEL_PATH = os.path.expanduser("~/.aaa/models/face_landmarker.task")

LEFT_EYE = [33, 159, 158, 133, 153, 155]
RIGHT_EYE = [362, 385, 386, 263, 374, 380]
NOSE_TIP = 1
NOSE_BRIDGE = 168
HEAD_TOP = 10
CHIN = 152


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.core import base_options as base_options_lib

        base = base_options_lib.BaseOptions(model_asset_path=_MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_mesh = FaceLandmarker.create_from_options(options)
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_mesh.detect(mp_image)
        signals = {
            "blink_rate": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "gaze_fixation": 0.0,
            "mouth_tension": 0.0,
            "face_detected": False,
        }

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            self._frames_no_face += 1
            return signals

        self._frames_no_face = 0
        landmarks = result.face_landmarks[0]

        ear_left = self._eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_right = self._eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0
        self._ear_history.append(ear)

        running_avg = float(np.mean(self._ear_history)) if self._ear_history else 0.5
        blink_threshold = max(0.28, running_avg * 0.60)
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

        nose_tip_pt = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y])
        signals["nose_y"] = float(nose_tip_pt[1])

        left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE], axis=0)
        right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE], axis=0)
        eye_delta = right_eye_center - left_eye_center
        signals["eye_angle"] = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))

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
