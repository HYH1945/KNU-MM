from monitoring import run_monitoring
from ptz_controller import PTZCameraController
from settings import configure_opencv_rtsp_tcp, load_settings
from yolo_detector import YoloConfig, YoloDetector


def main():
    settings = load_settings()
    configure_opencv_rtsp_tcp()

    print("Loading YOLO model...")
    detector = YoloDetector(
        YoloConfig(
            model_path=settings.model_path,
            conf_threshold=settings.conf_threshold,
            tracker_cfg_name=settings.tracker_cfg_name,
        )
    )
    print("YOLO model loaded successfully!")
    print(f"YOLO device: {detector.device}")
    if detector.use_tracking:
        print(f"Tracking: {detector.tracker_cfg}")
    else:
        print(f"Tracking disabled: '{settings.tracker_cfg_name}' not found")

    ptz_controller = PTZCameraController(
        settings.camera_ip,
        settings.camera_port,
        settings.camera_user,
        settings.camera_password,
    )

    run_monitoring(
        rtsp_url=settings.rtsp_url,
        detector=detector,
        ptz_controller=ptz_controller,
        window_name=settings.window_name,
        skip_frames=settings.skip_frames,
    )


if __name__ == "__main__":
    main()
