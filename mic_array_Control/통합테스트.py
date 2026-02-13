import argparse
import sys
import time
import cv2
import threading
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# PTZ 및 YOLO 관련 라이브러리
from onvif import ONVIFCamera
from ultralytics import YOLO

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent / 'src'))
CONFIG_PATH = Path(__file__).parent / 'config' / 'config.yaml'

load_dotenv()

# --- PTZ 컨트롤러 클래스 (사용자 코드 통합) ---
class PTZCameraController:
    def __init__(self):
        self.ip = os.getenv("CAMERA_IP")
        self.port = int(os.getenv("CAMERA_PORT", 80))
        self.user = os.getenv("CAMERA_USER")
        self.password = os.getenv("CAMERA_PASSWORD")
        self.is_connected = False
        self.connect()

    def connect(self):
        try:
            self.cam = ONVIFCamera(self.ip, self.port, self.user, self.password)
            self.ptz = self.cam.create_ptz_service()
            media = self.cam.create_media_service()
            self.profile = media.GetProfiles()[0].token
            self.move_request = self.ptz.create_type('ContinuousMove')
            self.move_request.ProfileToken = self.profile
            self.is_connected = True
            print("✅ PTZ 카메라 연결 성공")
        except Exception as e:
            print(f"❌ PTZ 연결 실패: {e}")

    def move(self, pan, tilt):
        if not self.is_connected: return
        def _send():
            self.move_request.Velocity = {'PanTilt': {'x': pan, 'y': tilt}, 'Zoom': {'x': 0}}
            self.ptz.ContinuousMove(self.move_request)
        threading.Thread(target=_send).start()

    def stop(self):
        if not self.is_connected: return
        threading.Thread(target=lambda: self.ptz.Stop({'ProfileToken': self.profile})).start()

# --- 실시간 통합 실행 루프 ---
def mode_integrated_realtime(args, config_dict):
    """실시간 모드: 분석(팀원) + 제어/추적(사용자)"""
    print("\n" + "="*60)
    print("🚀 통합 멀티모달 관제 시스템 가동")
    print("="*60)

    # 1. 모델 및 컨트롤러 초기화
    yolo_model = YOLO('yolov8n.pt')
    ptz = PTZCameraController()
    rtsp_url = os.getenv("RTSP_URL")
    cap = cv2.VideoCapture(rtsp_url)

    # 2. 팀원의 분석 시스템 초기화 (백그라운드 스레드용)
    from core.integrated_multimodal_system import IntegratedMultimodalSystem
    analysis_system = IntegratedMultimodalSystem(camera_id=rtsp_url, model=args.model or "gpt-4o-mini")

    # 3. 메인 루프 (OpenCV UI + YOLO 추적 + PTZ 제어)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        # 3프레임마다 YOLO 추론 (성능 최적화)
        if frame_count % 3 == 0:
            results = yolo_model.track(frame, persist=True, verbose=False)
            frame = results[0].plot()

            # [핵심] 특정 조건(예: 응급 상황) 시 분석 시스템 호출 가능
            # if is_emergency_detected: 
            #     threading.Thread(target=analysis_system.analyze_frame, args=(frame,)).start()

        cv2.imshow("Integrated Surveillance", frame)
        
        # 키보드 제어
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('w'): ptz.move(0, 0.5)
        elif key == ord('s'): ptz.move(0, -0.5)
        elif key == ord('a'): ptz.move(-0.5, 0)
        elif key == ord('d'): ptz.move(0.5, 0)
        elif key == ord(' '): ptz.stop()
        
        # 분석 실행 키 (Enter)
        elif key == 13: 
            print("🔍 현재 상황 분석 요청...")
        
            result = analysis_system.analyze_video_only("현재 상황을 분석해줘")
            print(f"결과: {result.get('multimodal_analysis', {}).get('situation')}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    # ... (팀원의 argparse 로직 유지) ...
    parser = argparse.ArgumentParser(description='Integrated Context LLM System')
    parser.add_argument('--mode', choices=['realtime', 'testset', 'file'], default='realtime')
    # ... (기타 인자 생략) ...
    args = parser.parse_args()

    # 설정 로드
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.mode == 'realtime':
        mode_integrated_realtime(args, config)
    # ... (다른 모드들 실행) ...

if __name__ == "__main__":
    main()