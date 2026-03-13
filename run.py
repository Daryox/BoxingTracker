import subprocess
import sys
import threading

def run_webcam():
    subprocess.run([sys.executable, "-m", "logic.models.visual.main"])

def run_dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit/dashboard.py"])

threading.Thread(target=run_webcam, daemon=True).start()
run_dashboard()