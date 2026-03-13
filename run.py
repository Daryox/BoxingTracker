import subprocess
import threading

def run_webcam():
    subprocess.run(["python", "-m", "logic.models.visual.main"])

def run_dashboard():
    subprocess.run(["streamlit", "run", "app/dashboard.py"])

threading.Thread(target=run_webcam).start()
run_dashboard()