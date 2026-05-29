import json
import os
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from aaa.utils.config import load_config

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

app = typer.Typer(
    name="aaa",
    help="Adaptive Attention Anchor — turn your body into a living interface for better code.",
    no_args_is_help=True,
)
console = Console()

_AAA_DIR = Path.home() / ".aaa"
_PID_PATH = _AAA_DIR / "daemon.pid"
_STATE_PATH = _AAA_DIR / "state.json"
_ANCHORS_PATH = _AAA_DIR / "anchors.json"
_CONFIG_PATH = _AAA_DIR / "config.toml"
_LOG_PATH = _AAA_DIR / "daemon.log"


def _ensure_dirs():
    _AAA_DIR.mkdir(parents=True, exist_ok=True)


def _is_daemon_running() -> bool:
    if not _PID_PATH.exists():
        return False
    pid = int(_PID_PATH.read_text().strip())
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        _PID_PATH.unlink(missing_ok=True)
        return False


def _read_state() -> dict:
    if _STATE_PATH.exists():
        return json.loads(_STATE_PATH.read_text())
    return {}


def _write_state(state: dict):
    _STATE_PATH.write_text(json.dumps(state, indent=2))


def _read_anchors() -> list[dict]:
    if _ANCHORS_PATH.exists():
        return json.loads(_ANCHORS_PATH.read_text())
    return []


def _write_anchors(anchors: list[dict]):
    _ANCHORS_PATH.write_text(json.dumps(anchors, indent=2))


def _list_cameras_and_exit():
    console.print("[yellow]Camera enumeration skipped (may hang on some systems).[/yellow]")
    console.print("Try these indices manually:\n")
    console.print("  aaa calibrate --camera 0")
    console.print("  aaa calibrate --camera 1")
    console.print("  aaa calibrate --camera 2")
    console.print("\nOr use IP Webcam:")
    console.print('  aaa calibrate --ip-cam "http://192.168.1.100:8080/video"')
    raise typer.Exit()


def _load_config_with_camera(camera_index: int = 255, ip_url: str | None = None) -> dict:
    config = load_config()
    if ip_url:
        config["camera"]["ip_webcam"]["enabled"] = True
        config["camera"]["ip_webcam"]["url"] = ip_url
    else:
        config["camera"]["ip_webcam"]["enabled"] = False
        if camera_index != 255:
            config["camera"]["device_index"] = camera_index
    return config


@app.command()
def start(
    headless: bool = typer.Option(False, "--headless", help="Run without system tray UI"),
    camera: int = typer.Option(255, "--camera", "-c", help="Camera device index (default: 0, use 255 to list)"),
    ip_cam: str = typer.Option("", "--ip-cam", help="IP Webcam URL (e.g. http://192.168.1.100:8080/video)"),
):
    """Start the aaa daemon in background."""
    _ensure_dirs()
    if _is_daemon_running():
        console.print("[yellow]aaa is already running.[/yellow]")
        raise typer.Exit()

    try:
        import cv2
    except ImportError:
        console.print("[red]Missing opencv-python. Run: pip install -e .[/red]")
        raise typer.Exit(1)
    try:
        import mediapipe
    except ImportError:
        console.print("[red]Missing mediapipe. Run: pip install -e .[/red]")
        raise typer.Exit(1)

    if not ip_cam and camera == 255:
        _list_cameras_and_exit()

    tracker_path = Path(__file__).parent / "core" / "tracker.py"
    if not tracker_path.exists():
        console.print("[red]Tracker module not found. Ensure aaa is installed correctly.[/red]")
        raise typer.Exit(1)

    config = _load_config_with_camera(camera, ip_cam or None)

    cmd = [sys.executable, str(tracker_path), "--camera", str(config["camera"]["device_index"])]
    if config["camera"]["ip_webcam"]["enabled"]:
        cmd += ["--ip-cam", config["camera"]["ip_webcam"]["url"]]
    if headless:
        cmd.append("--headless")

    with open(_LOG_PATH, "a") as log_file:
        log_file.write(f"\n--- aaa start at {time.time()} ---\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _PID_PATH.write_text(str(proc.pid))
    console.print(f"[green]aaa started[/green] (pid {proc.pid})")


@app.command()
def stop():
    """Stop the aaa daemon."""
    if not _is_daemon_running():
        console.print("[yellow]aaa is not running.[/yellow]")
        raise typer.Exit()

    pid = int(_PID_PATH.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.3)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    _PID_PATH.unlink(missing_ok=True)
    console.print("[green]aaa stopped.[/green]")


@app.command()
def calibrate(
    camera: int = typer.Option(255, "--camera", "-c", help="Camera device index (default: 0, use 255 to list)"),
    ip_cam: str = typer.Option("", "--ip-cam", help="IP Webcam URL (e.g. http://192.168.1.100:8080/video)"),
):
    """Run calibration to establish baseline body signals."""
    _ensure_dirs()
    if _is_daemon_running():
        console.print("[red]Stop aaa before running calibration.[/red]")
        raise typer.Exit(1)

    if not ip_cam and camera == 255:
        _list_cameras_and_exit()

    console.print("[bold]Calibration[/bold] — sit naturally for 15 seconds.")
    console.print("aaa will measure your resting posture, breathing, and blink rate.\n")

    try:
        from aaa.detectors.attention import calibrate_all
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Run: pip install -e .[/yellow]")
        raise typer.Exit(1)

    with console.status("[cyan]Calibrating...[/cyan]") as status:
        baseline = calibrate_all(duration=15.0, camera_index=camera if camera != 255 else 0, ip_url=ip_cam or None)

    _write_state({"calibration": baseline, "calibrated_at": time.time()})
    console.print("[green]Calibration complete.[/green]")

    table = Table(title="Baseline")
    table.add_column("Signal", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in baseline.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.3f}")
        else:
            table.add_row(key, str(value))
    console.print(table)


@app.command()
def status():
    """Show daemon health and latest signals."""
    _ensure_dirs()
    running = _is_daemon_running()
    state = _read_state()
    anchors = _read_anchors()

    console.print(f"[bold]aaa status[/bold]")
    console.print(f"  Running:    {'[green]yes[/green]' if running else '[red]no[/red]'}")
    console.print(f"  Anchors:    {len(anchors)}")
    console.print(f"  State file: {_STATE_PATH}")
    console.print(f"  PID file:   {_PID_PATH}")

    if state.get("calibration"):
        cal_at = state.get("calibrated_at", "?")
        console.print(f"\n  [bold]Calibration:[/bold] present ({cal_at})")
    if state.get("current"):
        current = state["current"]
        console.print(f"\n  [bold]Current signals:[/bold]")
        for key, value in current.items():
            if isinstance(value, bool):
                console.print(f"    {key}: {'🟢' if value else '🔴'}")
            elif isinstance(value, float):
                console.print(f"    {key}: {value:.3f}")
            else:
                console.print(f"    {key}: {value}")


@app.command()
def anchors():
    """Manage your physical anchors."""
    anchors_list = _read_anchors()

    if not anchors_list:
        console.print("[yellow]No anchors yet. Use aaa for a while to build some.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Anchors ({len(anchors_list)})")
    table.add_column("#", style="dim")
    table.add_column("Pose", style="cyan")
    table.add_column("File / Module", style="green")
    table.add_column("Created", style="magenta")

    for i, anchor in enumerate(anchors_list, 1):
        table.add_row(
            str(i),
            anchor.get("pose", "—"),
            anchor.get("file", "—"),
            anchor.get("timestamp", "—"),
        )
    console.print(table)


@app.command()
def cameras():
    """List available camera devices."""
    _list_cameras_and_exit()


if __name__ == "__main__":
    app()
