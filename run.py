"""
run.py — application entry point.

Starts two processes concurrently:
  1. Dash dashboard   (app/dash/dashboard.py)     — background process.
  2. Webcam pipeline  (logic.models.visual.main)  — main blocking call.

The webcam pipeline is the main thread.  When the user closes the video
window (presses 'q'), the webcam process exits and the dashboard process
is terminated automatically.

Open http://127.0.0.1:8050 in a browser after launching.
"""

import subprocess
import sys


# Start the dashboard in the background.
dash_proc = subprocess.Popen([sys.executable, "app/dash/dashboard.py"])

# Block here until the user closes the video window.
subprocess.run([sys.executable, "-m", "logic.models.visual.main"])

# Video window closed — shut down the dashboard server.
dash_proc.terminate()
dash_proc.wait()
