# aaa — Adaptive Attention Anchor

## Commands

```bash
pip install -e .                    # install (setuptools, no lockfile)
aaa start --camera 0                # launch daemon (Qt tray + HUD in main thread, engine in background thread)
aaa stop                            # stop daemon (uses psutil, os.kill broken on Windows)
aaa calibrate                       # 15s calibration with 5s pre-check phase (pose/face/lighting)
aaa status                          # show daemon health + current signals
aaa calibrate --ip-cam URL          # use phone camera via IP Webcam
```

No linter, typechecker, formatter, or test runner configured. No tests exist.

## Architecture

- **`aaa.cli:app`** (Typer) — entrypoint, spawns `aaa.core.tracker` as subprocess
- **`aaa.core.tracker`** — daemon entry: creates `QApplication` + `TrayIcon` + `HudWindow` in main thread, runs `Engine` in daemon thread
- **`aaa.core.engine`** — main loop (~30fps): read frame → posture/face detectors → attention classifier → gesture detector → interventions → save state → sleep
- **UI calls from engine thread** go through `Engine._ui_call()` → `QTimer.singleShot(0, lambda)` — never create Qt widgets from engine thread directly

## Data files (all `~/.aaa/`)

| File | Purpose |
|------|---------|
| `config.toml` | Auto-generated on first run, merged with `aaa.utils.config.DEFAULT_CONFIG` |
| `state.json` | Written every frame by engine: current signals + calibration + last gesture |
| `daemon.pid` | PID file (single-instance guard) |
| `daemon.log` | Python logging (`INFO`) |
| `anchors.json` | Physical anchor list |
| `models/pose_landmarker_heavy.task` | MediaPipe Pose model (auto-downloaded) |
| `models/face_landmarker.task` | MediaPipe Face Mesh model (auto-downloaded) |

## Detection pipeline

Each frame: `PostureDetector.process()` → posture signals → `AttentionEngine.update()` → state (flow/stuck/tension)
                          → `FaceDetector.process()` → face signals ↗
                          → `GestureDetector.process(nose_y, eye_angle)` → gesture (nod/tilt)

- **Nod** → `PageDown` scroll, cooldown 2s
- **Tilt left** → `Alt+Tab`, cooldown 1.5s
- **Tilt right** → `Alt+Shift+Tab`
- Gesture detection uses raw landmark metrics (`nose_y`, `eye_angle`), not solvePnP angles (unreliable at close range)

## Windows quirks

- `os.kill(pid, 0)` raises `OSError` — use `psutil.pid_exists()` instead (already patched in `cli.py`)
- Qt overlay windows (`OverlayWindow`) may not appear on top — gesture feedback uses system tray `showMessage()` instead
- Camera reconnection retries with exponential backoff (1–30s)

## Calibration pre-check

Before 15s data collection, a 5s phase shows on the camera window:
- Pose detected ✅/❌
- Face detected ✅/❌
- Lighting good/too dark/too bright

All status messages are in Russian (UI is Russian-localized).

## Notifications

- `Interventions` → Russian overlay windows (`OverlayWindow` via `show_intervention(kind)`)
- `Gesture feedback` → Windows native tray bubble via `TrayIcon.notify()`
- List of intervention kinds in `INTERVENTION_KINDS` in `engine.py`

## Gesture thresholds (gestures.py)

| Gesture | Signal | Threshold |
|---------|--------|-----------|
| Nod | `nose_y` deviation from running avg | `>0.02` down, `<0.005` up, total `>0.03` |
| Tilt right | `eye_angle` mean over 5 frames | `>8°`, all `>3°` |
| Tilt left | `eye_angle` mean over 5 frames | `< -8°`, all `< -3°` |
