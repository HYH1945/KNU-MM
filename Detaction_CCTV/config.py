import os
from dotenv import load_dotenv

load_dotenv()

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
    AI_MODEL_PATH: str = 'Detaction_CCTV/yolo26n.pt'