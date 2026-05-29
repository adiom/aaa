import json
import logging
import random
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2

from aaa.detectors.posture import PostureDetector
from aaa.detectors.face import FaceDetector
from aaa.detectors.attention import AttentionEngine
from aaa.core.audio import AudioEngine
from aaa.core.anchors import add_anchor, find_anchor_for_file
from aaa.utils.config import load_config
from aaa.utils.helpers import (
    safe_camera_open,
    get_active_window,
    window_is_editor,
    get_file_mod_times,
    detect_file_changes,
    clamp,
)

_LOG_PATH = Path.home() / ".aaa" / "daemon.log"
_STATE_PATH = Path.home() / ".aaa" / "state.json"

logging.basicConfig(
    filename=str(_LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("aaa")


INTERVENTION_MESSAGES = {
    "slouch": [
        ("⟐  straighten  ⟐", "your skeleton is your architecture"),
        ("⟐  unwind  ⟐", "your spine wants to be long"),
        ("⟐  rise  ⟐", "imagine a string pulling you up from the crown of your head"),
        ("⟐  open  ⟐", "roll your shoulders back and down"),
    ],
    "neck": [
        ("⌇  release  ⌇", "your neck is carrying more than it should"),
        ("⌇  lengthen  ⌇", "tuck your chin gently, feel the back of your neck stretch"),
        ("⌇  float  ⌇", "imagine your head is a balloon"),
    ],
    "stare": [
        ("◉  drift  ◉", "your eyes have been fixed too long. look at something 6m away"),
        ("◉  soften  ◉", "unfocus your gaze. let your peripheral vision open"),
        ("◉  distance  ◉", "find the farthest point you can see and rest there"),
    ],
    "blink": [
        ("◈  blink  ◈", "slowly. three times. feel your eyelids"),
        ("◈  hydrate  ◈", "your eyes are dry. blink fully, close for 2 seconds"),
        ("◈  reset  ◈", "close your eyes for 5 seconds"),
    ],
    "jaw": [
        ("⏾  unclench  ⏾", "part your lips. tongue resting on the roof of your mouth"),
        ("⏾  soften  ⏾", "your jaw should not hold tension. let it hang"),
        ("⏾  release  ⏾", "massage your jaw muscles with your fingertips"),
    ],
    "breath": [
        ("〰️  breathe  〰️", "inhale 4s — hold 2s — exhale 6s"),
        ("〰️  slow down  〰️", "your breath is your anchor. feel it"),
        ("〰️  reset  〰️", "three deep breaths. in through nose, out through mouth"),
    ],
    "tension": [
        ("⚡  shake it off  ⚡", "shake your hands for 10 seconds"),
        ("⚡  micro-break  ⚡", "stand up. arms above head. breathe deep"),
        ("⚡  reset  ⚡", "roll your wrists, shrug your shoulders, wiggle your fingers"),
    ],
    "stillness": [
        ("⋯  move  ⋯", "you've been still too long. shift your weight"),
        ("⋯  flow  ⋯", "micro-movement: trace an infinity symbol with your nose"),
        ("⋯  stretch  ⋯", "reach one arm to the ceiling, then the other"),
    ],
}


class Engine:
    def __init__(self, camera_index: int | None = None, ip_url: str | None = None):
        self.ip_url = ip_url
        self.config = load_config()
        if ip_url:
            self.config["camera"]["ip_webcam"]["enabled"] = True
            self.config["camera"]["ip_webcam"]["url"] = ip_url
        elif camera_index is not None:
            self.config["camera"]["device_index"] = camera_index
        self.posture = PostureDetector()
        self.face = FaceDetector()
        self.attention = AttentionEngine()
        self.audio = AudioEngine()
        self.cap: Optional[cv2.VideoCapture] = None
        self.baseline: dict = {}
        self._running = False
        self._prev_file_mtimes: dict[str, float] = {}
        self._focus_session_start: Optional[float] = None
        self._focus_minutes = 0.0
        self._last_intervention_time = 0.0
        self._last_anchor_check_file: Optional[str] = None
        self._session_start = time.time()
        self._reconnect_delay = 1.0

        for key in INTERVENTION_MESSAGES:
            setattr(self, f"_{key}_timer", 0.0)

        self._load_baseline()
        log.info("Engine initialized (ip_url=%s, camera_index=%s)", ip_url, camera_index)

    def _log(self, msg: str):
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info(msg)

    def _load_baseline(self):
        if _STATE_PATH.exists():
            try:
                state = json.loads(_STATE_PATH.read_text())
                self.baseline = state.get("calibration", {})
            except (json.JSONDecodeError, KeyError):
                self.baseline = {}

    def _save_state(self, signals: dict, attention_result: dict):
        state = {
            "current": {
                "slouch_score": round(signals.get("slouch_score", 0.0), 3),
                "blink_rate": round(signals.get("blink_rate", 0.0), 1),
                "gaze_fixation": round(signals.get("gaze_fixation", 0.0), 3),
                "neck_forward_angle": round(signals.get("neck_forward_angle", 0.0), 1),
                "head_yaw": round(signals.get("head_yaw", 0.0), 1),
                "head_roll": round(signals.get("head_roll", 0.0), 1),
                "mouth_tension": round(signals.get("mouth_tension", 0.0), 3),
                "pose_detected": signals.get("pose_detected", False),
                "face_detected": signals.get("face_detected", False),
                "state": attention_result.get("state", "unknown"),
                "confidence": round(attention_result.get("confidence", 0.0), 3),
            },
            "session_started_at": self._session_start,
            "calibration": self.baseline,
        }
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2))

    def _deviation(self, signal_key: str, current_value: float, threshold: float = 0.2) -> float:
        base = self.baseline.get(signal_key, 0.0)
        if base == 0:
            return abs(current_value)
        return abs(current_value - base) / max(abs(base), 0.001)

    def _check_interventions(self, signals: dict, attention_result: dict) -> Optional[str]:
        now = time.time()
        cd = self.config["interventions"]["cooldown_after_intervention_seconds"]
        if now - self._last_intervention_time < cd:
            return None

        slouch = signals.get("slouch_score", 0.0)
        neck = signals.get("neck_forward_angle", 0.0)
        blink = signals.get("blink_rate", 15.0)
        gaze = signals.get("gaze_fixation", 0.0)
        jaw = signals.get("mouth_tension", 0.0)
        head_yaw = abs(signals.get("head_yaw", 0.0))
        pose = signals.get("pose_detected", False)
        face = signals.get("face_detected", False)

        dev_slouch = self._deviation("slouch_score", slouch, 0.3)
        dev_neck = self._deviation("neck_forward_angle", neck, 10.0)
        dev_gaze = gaze
        dev_jaw = jaw
        dev_blink = self._deviation("blink_rate", blink, 5.0)

        candidates = []

        if pose and dev_slouch > 0.5 and now - getattr(self, "_slouch_timer", 0) > 15:
            candidates.append(("slouch", dev_slouch))
            self._slouch_timer = now

        if pose and dev_neck > 0.4 and now - getattr(self, "_neck_timer", 0) > 20:
            candidates.append(("neck", dev_neck))
            self._neck_timer = now

        if face and gaze > 0.85 and now - getattr(self, "_stare_timer", 0) > 25:
            candidates.append(("stare", gaze))
            self._stare_timer = now

        if face and blink < 5 and now - getattr(self, "_blink_timer", 0) > 30:
            candidates.append(("blink", 1.0 - blink / 15.0))
            self._blink_timer = now

        if face and dev_jaw > 0.6 and now - getattr(self, "_jaw_timer", 0) > 20:
            candidates.append(("jaw", dev_jaw))
            self._jaw_timer = now

        if not pose and not face:
            if now - getattr(self, "_stillness_timer", 0) > 60:
                candidates.append(("stillness", 0.5))
                self._stillness_timer = now

        tenor = attention_result.get("tension_score", 0)
        if tenor > 0.7 and now - getattr(self, "_tension_timer", 0) > 45:
            candidates.append(("tension", tenor))
            self._tension_timer = now

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[1])
        chosen = candidates[0][0]
        self._last_intervention_time = now
        log.info("Intervention: %s (dev=%.3f)", chosen, candidates[0][1])
        return chosen

    def _handle_intervention(self, kind: str):
        msgs = INTERVENTION_MESSAGES.get(kind, [("⋆  pause  ⋆", "")])
        title, sub = random.choice(msgs)
        self.audio.play_texture("tension" if kind in ("tension", "slouch") else "stuck")

        from aaa.ui.notifications import show_intervention
        show_intervention(title, sub)

    def _check_anchors(self):
        active = get_active_window()
        if not active or not window_is_editor(active):
            return None
        if active == self._last_anchor_check_file:
            return None
        self._last_anchor_check_file = active
        return find_anchor_for_file(active)

    def _check_file_changes(self):
        for path_str in self.config["anchors"].get("watch_directories", [str(Path.cwd())]):
            p = Path(path_str)
            if p.is_dir():
                current = get_file_mod_times(p)
                changes = detect_file_changes(self._prev_file_mtimes, current)
                self._prev_file_mtimes = current
                if changes:
                    return changes
        return []

    def _track_focus(self):
        active = get_active_window()
        if active and window_is_editor(active):
            if self._focus_session_start is None:
                self._focus_session_start = time.time()
            else:
                from aaa.utils.helpers import smooth_value
                elapsed = (time.time() - self._focus_session_start) / 60.0
                self._focus_minutes = smooth_value(elapsed, self._focus_minutes, alpha=0.1)
        else:
            self._focus_session_start = None

    def _maybe_create_anchor(self, file_changes: list[str]):
        min_focus = self.config["anchors"]["min_focus_minutes"]
        if self._focus_minutes >= min_focus and file_changes:
            for f in file_changes[:1]:
                add_anchor(file_path=f, module_name=f, event="big_refactor")
                from aaa.ui.notifications import show_anchor_prompt
                show_anchor_prompt(f)

    def run(self):
        self._running = True
        cam_cfg = self.config["camera"]
        ip_url = self.ip_url or (cam_cfg.get("ip_webcam", {}).get("url") if cam_cfg.get("ip_webcam", {}).get("enabled") else None)

        while self._running:
            try:
                self.cap = safe_camera_open(cam_cfg["device_index"], ip_url=ip_url)
                if self.cap is None:
                    log.error("Could not open camera, retrying in %.0fs", self._reconnect_delay)
                    time.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(30.0, self._reconnect_delay * 2)
                    continue
                self._reconnect_delay = 1.0
                log.info("Camera opened")

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

                frame_interval = 1.0 / cam_cfg["fps"]
                last_audio_time = 0.0
                frame_count = 0

                while self._running:
                    loop_start = time.time()

                    ret, frame = self.cap.read()
                    if not ret:
                        log.warning("Camera frame read failed, reconnecting...")
                        break

                    frame_count += 1
                    posture_signals = self.posture.process(frame)
                    face_signals = self.face.process(frame, fps=cam_cfg["fps"])
                    attention_result = self.attention.update(posture_signals, face_signals, self.baseline)

                    combined = {**posture_signals, **face_signals}
                    self._save_state(combined, attention_result)

                    state = attention_result.get("state", "unknown")
                    if time.time() - last_audio_time >= 0.5:
                        self.audio.update_state(state)
                        self.audio.play_texture(state)
                        last_audio_time = time.time()

                    intervention = self._check_interventions(combined, attention_result)
                    if intervention:
                        self._handle_intervention(intervention)

                    anchor_reminder = self._check_anchors()
                    if anchor_reminder:
                        from aaa.ui.notifications import show_anchor_reminder
                        show_anchor_reminder(anchor_reminder)

                    self._track_focus()
                    file_changes = self._check_file_changes()
                    if file_changes:
                        self._maybe_create_anchor(file_changes)

                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                log.error("Engine error: %s\n%s", e, traceback.format_exc())
                time.sleep(1.0)
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None

        self.shutdown()

    def shutdown(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.posture.close()
        self.face.close()
        self.audio.close()
        log.info("Engine shut down")
