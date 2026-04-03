"""
main.py — webcam pipeline entry point.

When invoked with --video <path>, uses that file as the source.
Otherwise probes cameras and (if needed) prompts the user to pick one.
"""

import argparse
import os

from logic.models.visual.yolo.yoloposewcam import YoloPoseWebcam
from logic.models.visual.device import select_pose_model
import cv2


def main() -> None:
    """Configure the video source, then start tracking."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", default=None)
    args, _ = parser.parse_known_args()

    if args.video is not None:
        video_source: object = args.video
    else:
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
            print("Available camera indices:", cam_indices)
            cam_index = int(input("Enter the camera index to use: "))
            print(f"Using camera index {cam_index}")
        elif len(cam_indices) == 1:
            cam_index = cam_indices[0]
            print(f"Using camera index {cam_index}")
        else:
            print("No cameras found.")
            return
        video_source = cam_index

    app = YoloPoseWebcam(
        model_path=select_pose_model(),
        video_source=video_source,
        img_size=416,
        conf=0.25,
        show_fps=True,
    )
    app.run()


if __name__ == "__main__":
    main()
