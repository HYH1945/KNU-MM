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
    AI_MODEL_PATH: str = './yolov8n.pt'