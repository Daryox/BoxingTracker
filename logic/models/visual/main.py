"""
main.py — webcam pipeline entry point.

Probes all camera indices 0-9, lets the user pick one if multiple are
available, then starts the YoloPoseWebcam run loop.
"""

from logic.models.visual.yolo.yoloposewcam import YoloPoseWebcam
from logic.models.visual.device import select_pose_model
import cv2


def main() -> None:
    """Detect available cameras, prompt for selection, then start tracking."""
    # Probe each index to find cameras that can actually deliver frames.
    cam_indices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cam_indices.append(i)
        cap.release()

    if len(cam_indices) > 1:
        # Let the user choose when more than one camera is available.
        print("Available camera indices:", cam_indices)
        cam_index = int(input("Enter the camera index to use: "))
        print(f"Using camera index {cam_index}")
    elif len(cam_indices) == 1:
        cam_index = cam_indices[0]
        print(f"Using camera index {cam_index}")
    else:
        print("No cameras found.")
        return

    app = YoloPoseWebcam(
        model_path=select_pose_model(),
        camera_index=cam_index,
        img_size=416,
        conf=0.25,
        show_fps=True,
    )
    app.run()


if __name__ == "__main__":
    main()
