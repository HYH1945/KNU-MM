import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _env_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

class AppConfig:
    # CCTV INFORMATION
    RTSP_URL = os.getenv("RTSP_URL")
    CAMERA_IP = os.getenv("CAMERA_IP")
    CAMERA_PORT = int(os.getenv("CAMERA_PORT", 80))
    CAMERA_USER = os.getenv("CAMERA_USER")
    CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD") 

    # Control Parameters
    PID_KP: float = 0.4             # TRACKING MODE SPEED
    DEAD_ZONE_PIXELS: int = 50
    PATROL_SPEED: float = 0.2       # PARTROL MODE SPEED

    # AI Parameters
    AI_CONFIDENCE: float = 0.3
    # [변경됨] YOLOv8n -> YOLO26n (Ultralytics 최신 모델 적용)
    # yolo26n.pt: Nano 버전 (M1 Mac CPU 환경에 최적화됨)
    # 실행 시 자동으로 Ultralytics 서버에서 다운로드됩니다.
    AI_MODEL_PATH: str = 'Detection_CCTV/yolo26n.pt'
    AI_DEVICE: str = os.getenv("AI_DEVICE", os.getenv("YOLO_DEVICE", "auto"))
    AI_TRACKER_CFG: str = os.getenv("AI_TRACKER_CFG", os.getenv("YOLO_TRACKER_CFG", ""))
    AI_USE_TRACKING: bool = _env_bool("AI_USE_TRACKING", "1")
    AI_SKIP_FRAMES: int = int(os.getenv("AI_SKIP_FRAMES", os.getenv("YOLO_SKIP_FRAMES", "1")))

    # UI / Monitoring
    WINDOW_NAME: str = os.getenv("WINDOW_NAME", "Smart CCTV")
    SHOW_FPS: bool = _env_bool("SHOW_FPS", "0")
    ENABLE_MANUAL_CONTROL: bool = _env_bool("ENABLE_MANUAL_CONTROL", "1")
    MANUAL_OVERRIDE_SECONDS: float = float(os.getenv("MANUAL_OVERRIDE_SECONDS", "0.0"))

    # PTZ
    PTZ_LOG_ERRORS: bool = _env_bool("PTZ_LOG_ERRORS", "0")
