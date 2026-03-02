from logic.models.visual.yolo.yoloposewcam import YoloPoseWebcam

def main():
    app = YoloPoseWebcam(
        model_path="yolo26n-pose.pt",
        camera_index=0,
        img_size=640,
        conf=0.25,
        show_fps=True,
        deepsort_embedder=None, #use mobilenet later for ReID capabilities. I always get pkg_resources not found error when trying to use the embedder, so leaving it out for now.
    )
    app.run()

if __name__ == "__main__":
    main()