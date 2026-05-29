"""
Subtle audio cues using sounddevice + numpy.

Produces low-frequency pulses (near-subliminal) that sync with user state:
  - Flow: slow, deep pulses (~40 Hz, 1.5s interval) — grounding
  - Stuck: slightly higher, irregular — gentle nudge
  - Tension: very low, slow — calming
  - Unknown: silence / minimal
"""

import time
from typing import Optional

import numpy as np

from aaa.utils.config import load_config

SAMPLE_RATE = 44100


class AudioEngine:
    def __init__(self):
        self.enabled = True
        self.volume = 0.15
        self._stream: Optional["sounddevice.OutputStream"] = None
        self._current_state = "unknown"
        self._last_pulse_time = 0.0
        self._pulse_interval = 1.5
        self._config = load_config().get("audio", {})
        self._apply_config()

    def _apply_config(self):
        cfg = self._config
        self.enabled = cfg.get("enabled", True)
        self.volume = cfg.get("volume", 0.15)
        self._pulse_interval = cfg.get("pulse_interval_seconds", 1.5)

    def _state_params(self, state: str) -> tuple[float, float]:
        cfg = self._config
        if state == "flow":
            return cfg.get("flow_freq_hz", 40.0), 1.8
        elif state == "stuck":
            return cfg.get("stuck_freq_hz", 70.0), 0.8
        elif state == "tension":
            return cfg.get("base_freq_hz", 55.0), 2.2
        else:
            return 0.0, 0.0

    def _generate_pulse(self, freq: float) -> np.ndarray:
        if freq <= 0:
            return np.array([], dtype=np.float32)
        duration = 0.15
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        wave = np.sin(2.0 * np.pi * freq * t)
        envelope = np.exp(-t * 20.0)
        return (wave * envelope * self.volume).astype(np.float32)

    def _generate_texture(self, state: str, duration_sec: float = 0.1) -> np.ndarray:
        if not self.enabled:
            return np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)
        freq, interval = self._state_params(state)
        if freq <= 0:
            return np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)

        now = time.time()
        samples_needed = int(SAMPLE_RATE * duration_sec)
        buffer = np.zeros(samples_needed, dtype=np.float32)

        if now - self._last_pulse_time >= interval:
            self._last_pulse_time = now
            pulse = self._generate_pulse(freq)
            length = min(len(pulse), len(buffer))
            buffer[:length] += pulse[:length]

        return buffer

    def play_texture(self, state: str, duration_sec: float = 0.1):
        if not self.enabled:
            return
        try:
            import sounddevice as sd
            audio = self._generate_texture(state, duration_sec)
            if len(audio) > 0 and np.max(np.abs(audio)) > 0:
                sd.play(audio, samplerate=SAMPLE_RATE, blocking=False)
        except Exception:
            self.enabled = False

    def update_state(self, state: str):
        if state != self._current_state:
            self._current_state = state
            if state == "tension":
                self.play_calming_tone()

    def play_calming_tone(self, duration_sec: float = 2.0):
        try:
            import sounddevice as sd
            freq = self._config.get("base_freq_hz", 55.0)
            t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
            wave = np.sin(2.0 * np.pi * freq * t) * np.exp(-t * 1.5)
            audio = (wave * self.volume * 0.5).astype(np.float32)
            sd.play(audio, samplerate=SAMPLE_RATE, blocking=False)
        except Exception:
            pass

    def close(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
