"""
run.py — application entry point.

Starts two processes concurrently:
  1. Webcam pipeline  (logic.models.visual.main) — runs as a background daemon thread.
  2. Dash dashboard   (app/dash/dashboard.py)     — runs in the main thread.

The dashboard is the main thread so that the process exits cleanly when the
browser tab is closed; the webcam thread dies with it because it is a daemon.

Open http://127.0.0.1:8050 in a browser after launching.
"""

import subprocess
import sys
import threading


def run_webcam() -> None:
    """Launch the webcam pose-tracking pipeline as a subprocess."""
    subprocess.run([sys.executable, "-m", "logic.models.visual.main"])


def run_dashboard() -> None:
    """Launch the Dash live dashboard as a subprocess."""
    subprocess.run([sys.executable, "app/dash/dashboard.py"])


# Start the webcam pipeline in the background; it dies when the main thread exits.
threading.Thread(target=run_webcam, daemon=True).start()

# Block here until the dashboard (main thread) finishes.
run_dashboard()
