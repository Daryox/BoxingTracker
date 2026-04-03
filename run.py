"""
run.py — application entry point.

Prompts the user to select an analysis mode, then starts two processes:
  1. Dash dashboard   (app/dash/dashboard.py)     — background process.
  2. Video pipeline   (logic.models.visual.main)  — main blocking call.

The browser opens automatically at http://127.0.0.1:8050 after a short delay.
When the user closes the video window (presses 'q'), the pipeline exits and
the dashboard process is terminated automatically.
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser

DASHBOARD_URL = "http://127.0.0.1:8050"


def _open_browser() -> None:
    """Wait briefly for the server to start, then open the dashboard in the browser."""
    time.sleep(2)
    webbrowser.open(DASHBOARD_URL)


# ── Mode selection ─────────────────────────────────────────────────────────────

print("=" * 40)
print("  Boxing Tracker")
print("=" * 40)
print("  1 — Real-time (webcam)")
print("  2 — Analyse a video file")
print("=" * 40)

while True:
    _choice = input("Select mode (1 or 2): ").strip()
    if _choice in ("1", "2"):
        break
    print("Please enter 1 or 2.")

_extra_args: list = []

if _choice == "2":
    while True:
        _video_path = input("Enter path to video file: ").strip().strip('"').strip("'")
        if os.path.isfile(_video_path):
            break
        print(f"File not found: {_video_path!r}. Try again.")
    _extra_args = ["--video", _video_path]
    print(f"Using video file: {_video_path}")

# ── Launch dashboard + pipeline ────────────────────────────────────────────────

# Start the dashboard in the background.
dash_proc = subprocess.Popen([sys.executable, "app/dash/dashboard.py"])

# Open the browser once the server has had time to start.
threading.Thread(target=_open_browser, daemon=True).start()

# Block here until the user closes the video window.
subprocess.run([sys.executable, "-m", "logic.models.visual.main"] + _extra_args)

# Video window closed — shut down the dashboard server.
dash_proc.terminate()
dash_proc.wait()
