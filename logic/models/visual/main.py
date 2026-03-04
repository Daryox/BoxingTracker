from logic.models.visual.yolo.yoloposewcam import YoloPoseWebcam
import cv2


def main():
    camindexlist = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camindexlist.append(i)
        cap.release()
    if len(camindexlist) > 1:
        print("Available camera indices:", camindexlist)
        camindex = int(input("Enter the camera index to use: "))
        print(f"Using camera index {camindex}")
    elif len(camindexlist) == 1:
        camindex = camindexlist[0]
        print(f"Using camera index {camindex}")
    else:
        print("No cameras found.")
        return

    app = YoloPoseWebcam(
        model_path="yolo26n-pose.pt",
        camera_index=camindex,
        img_size=640,
        conf=0.25,
        show_fps=True,
        deepsort_embedder=None, #use mobilenet later for ReID capabilities. I always get pkg_resources not found error when trying to use the embedder, so leaving it out for now.
    )
    app.run()

if __name__ == "__main__":
    main()