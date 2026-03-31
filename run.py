"""
run.py — application entry point.

Starts two processes concurrently:
  1. Dash dashboard   (app/dash/dashboard.py)     — background process.
  2. Webcam pipeline  (logic.models.visual.main)  — main blocking call.

The webcam pipeline is the main thread.  When the user closes the video
window (presses 'q'), the webcam process exits and the dashboard process
is terminated automatically.8

The browser opens automatically at http://127.0.0.1:8050 after a short delay.
"""

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


# Start the dashboard in the background.
dash_proc = subprocess.Popen([sys.executable, "app/dash/dashboard.py"])

# Open the browser once the server has had time to start.
threading.Thread(target=_open_browser, daemon=True).start()

# Block here until the user closes the video window.
subprocess.run([sys.executable, "-m", "logic.models.visual.main"])

# Video window closed — shut down the dashboard server.
dash_proc.terminate()
dash_proc.wait()
