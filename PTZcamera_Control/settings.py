import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    rtsp_url: Optional[str]
    camera_ip: Optional[str]
    camera_port: int
    camera_user: Optional[str]
    camera_password: Optional[str]
    model_path: str
    conf_threshold: float
    tracker_cfg_name: str
    skip_frames: int
    window_name: str


def load_settings() -> Settings:
    return Settings(
        rtsp_url=os.getenv("RTSP_URL"),
        camera_ip=os.getenv("CAMERA_IP"),
        camera_port=int(os.getenv("CAMERA_PORT", "80")),
        camera_user=os.getenv("CAMERA_USER"),
        camera_password=os.getenv("CAMERA_PASSWORD"),
        model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
        conf_threshold=float(os.getenv("YOLO_CONF", "0.5")),
        tracker_cfg_name=os.getenv("YOLO_TRACKER_CFG", "botsort.yaml"),
        skip_frames=int(os.getenv("YOLO_SKIP_FRAMES", "3")),
        window_name=os.getenv("WINDOW_NAME", "CCTV Monitoring + YOLO Detection"),
    )


def configure_opencv_rtsp_tcp() -> None:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
