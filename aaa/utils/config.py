from pathlib import Path
from typing import Any

_AAA_DIR = Path.home() / ".aaa"
_CONFIG_PATH = _AAA_DIR / "config.toml"

DEFAULT_CONFIG = {
    "camera": {
        "device_index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "ip_webcam": {
            "enabled": False,
            "url": "http://192.168.1.100:8080/video",
        },
    },
    "detection": {
        "posture_check_interval": 0.5,
        "blink_window_seconds": 5.0,
        "gaze_fixation_threshold": 3.0,
        "slouch_angle_threshold": 15.0,
        "tension_threshold": 0.7,
    },
    "audio": {
        "enabled": True,
        "volume": 0.15,
        "base_freq_hz": 55.0,
        "flow_freq_hz": 40.0,
        "stuck_freq_hz": 70.0,
        "pulse_interval_seconds": 1.5,
    },
    "interventions": {
        "slouch_timeout_seconds": 8.0,
        "high_tension_timeout_seconds": 15.0,
        "micro_break_interval_minutes": 45.0,
        "anchor_prompt_seconds": 5.0,
        "darken_opacity": 0.15,
        "cooldown_after_intervention_seconds": 60.0,
    },
    "anchors": {
        "enabled": True,
        "file_change_window_seconds": 300,
        "min_focus_minutes": 15.0,
    },
    "general": {
        "headless": False,
        "auto_calibrate": True,
        "calibration_duration": 30.0,
        "data_dir": str(_AAA_DIR),
        "log_level": "INFO",
    },
}


def load_config() -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if _CONFIG_PATH.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(_CONFIG_PATH, "rb") as f:
            user_config = tomllib.load(f)
        _deep_merge(config, user_config)
    return config


def save_config(config: dict[str, Any]):
    _AAA_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(_to_toml_string(config))


def _deep_merge(base: dict, override: dict):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _to_toml_string(data: dict, prefix: str = "") -> str:
    lines = []
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            lines.append(f"\n[{full_key}]")
            lines.append(_to_toml_string(value, full_key))
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")
    return "\n".join(lines)
