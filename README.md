# aaa — Adaptive Attention Anchor

**Turn your body into a living interface for better code.**

`aaa` is a strange but surprisingly effective tool that watches your posture, micro-movements, breathing, and facial signals while you code — and actively shapes your workflow around them.

---

## What is aaa?

Most coding tools focus on the screen.  
**aaa** focuses on **you**.

It quietly observes your body through your webcam and turns physical states into meaningful feedback and interventions. The goal is simple: make you a better programmer by making your body part of the development process.

## Core Features

### Body-aware Linting
- Detects slouching, neck tension, or hyper-fixation
- Slowly darkens parts of the screen to force natural movement
- Encourages better posture without annoying pop-ups

### Muscle Memory Anchoring
- After important moments (big refactors, complex solutions, etc.), `aaa` asks you to perform a short, unique physical pose or gesture
- Later, when you reopen that file or module, it reminds you to recreate the same pose for a few seconds
- Builds deep somatic connection to your own codebase

### Attention Rhythm Engine
- Monitors blink rate, head movement, and breathing patterns
- Uses subtle audio cues (rhythmic pulses synchronized to your body) to maintain flow state
- Changes the soundscape when you're in deep focus vs. when you're stuck

### Micro-breaks
- Detects signs of tension and reduced movement
- Triggers short, specific physical movements (jaw clenches, wrist rotations, etc.) that actually help

### Personal Anchor Library
- Over time builds a personal library of body states connected to specific projects or code modules

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/aaa.git
cd aaa

# Editable install with all dependencies
pip install -e .

# Or minimal install + manual deps
pip install -e . --no-deps
pip install -r requirements.txt
```

Requires **Python 3.11+** and a webcam.

## Usage

```bash
# Start the daemon (background process)
aaa start

# Stop it
aaa stop

# Calibrate baseline body signals (30s)
aaa calibrate

# Show daemon status and current signals
aaa status

# List your physical anchors
aaa anchors list
```

## Project Structure

```
aaa/
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
└── aaa/
    ├── __init__.py
    ├── cli.py              # Typer CLI (start/stop/calibrate/status/anchors)
    ├── core/
    │   ├── engine.py       # Orchestrator: wires everything together
    │   ├── tracker.py      # Daemon entry point, camera loop
    │   ├── anchors.py      # Physical anchor CRUD + matching
    │   └── audio.py        # Low-frequency subtle audio pulses
    ├── detectors/
    │   ├── posture.py      # MediaPipe Pose: slouch, neck, spine
    │   ├── face.py         # MediaPipe Face Mesh: blink, gaze, head pose
    │   └── attention.py    # Classifier: flow / stuck / tension
    ├── ui/
    │   ├── tray.py         # PySide6 system tray icon
    │   └── notifications.py # Screen overlays (darken, break, anchor)
    └── utils/
        ├── config.py       # TOML config loader + defaults
        └── helpers.py      # Window detection, smoothing, camera helpers
```

## Configuration

Config lives at `~/.aaa/config.toml`. Auto-generated with sensible defaults on first run.

Key settings:

```toml
[camera]
device_index = 0
fps = 30

[detection]
slouch_angle_threshold = 15.0
tension_threshold = 0.7

[audio]
enabled = true
volume = 0.15

[interventions]
slouch_timeout_seconds = 8.0
micro_break_interval_minutes = 45.0
```

## Philosophy

Most productivity tools are about managing your *time*.  
aaa is about managing your *state*.

By creating a feedback loop between your body and your code, aaa helps you:
- Notice tension before it becomes pain
- Find flow more deliberately
- Build muscle memory for your most important work
- Remember that you are a physical being, not just a brain on a stick

## License

MIT
