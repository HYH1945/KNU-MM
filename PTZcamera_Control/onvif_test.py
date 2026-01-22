import cv2
import threading
import os
from dotenv import load_dotenv
from onvif import ONVIFCamera
from ultralytics import YOLO


load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
CAMERA_IP = os.getenv("CAMERA_IP")
CAMERA_PORT = int(os.getenv("CAMERA_PORT", 80))
CAMERA_USER = os.getenv("CAMERA_USER")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")

# FFmpeg : TCP 전송 강제화
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class PTZCameraController: # PTZ Camera 제어 클래스
    def __init__(self, ip, port, user, password):
        try:
            self.cam = ONVIFCamera(ip, port, user, password)
            self.ptz_service = self.cam.create_ptz_service()
            media_service = self.cam.create_media_service()
            
            self.profile_token = media_service.GetProfiles()[0].token
            
            self.move_request = self.ptz_service.create_type('ContinuousMove')
            self.move_request.ProfileToken = self.profile_token
            
            self.is_connected = True
            print("PTZ Camera Connected Successfully.")
            
        except Exception as e:
            print(f"Connection Error: {e}")
            self.is_connected = False

    def _execute_async(self, func):
        threading.Thread(target=func).start()

    def start_continuous_move(self, pan_velocity, tilt_velocity):
        if not self.is_connected:
            return

        def send_move_command():
            try:
                self.move_request.Velocity = {
                    'PanTilt': {'x': pan_velocity, 'y': tilt_velocity},
                    'Zoom': {'x': 0.0}
                }
                self.ptz_service.ContinuousMove(self.move_request)
            except Exception as e:
                print(f"Move Error: {e}")

        self._execute_async(send_move_command)

    def stop_move(self):
        if not self.is_connected:
            return

        def send_stop_command():
            try:
                self.ptz_service.Stop({'ProfileToken': self.profile_token})
            except Exception as e:
                print(f"Stop Error: {e}")

        self._execute_async(send_stop_command)

def main():
    # YOLO 모델 로드 (yolov8n.pt는 가장 가볍고 빠른 모델)
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("YOLO model loaded successfully!")
    
    ptz_controller = PTZCameraController(CAMERA_IP, CAMERA_PORT, CAMERA_USER, CAMERA_PASSWORD)

    video_capture = cv2.VideoCapture(RTSP_URL)
    window_name = "CCTV Monitoring + YOLO Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if not video_capture.isOpened():
        print("Error: Cannot open video stream.")
        return

    print("Controls: [W/A/S/D] Move, [Space] Stop, [Q] Quit")
    print("YOLO detection: Every 3 frames (for better performance)")

    frame_count = 0
    skip_frames = 3  # 3프레임마다 1번만 YOLO 추론 (성능 최적화)
    last_results = None  # 마지막 추론 결과 저장

    try:
        while True:
            is_frame_read, frame = video_capture.read()
            
            if not is_frame_read:
                print("Video stream disconnected.")
                break
            
            frame_count += 1
            
            # N프레임마다만 YOLO 추론 수행 (성능 최적화)
            if frame_count % skip_frames == 0:
                # YOLO 추론 수행 (conf=0.5는 확신도 50% 이상인 것만 표시)
                results = model(frame, conf=0.5, verbose=False)
                last_results = results
            
            # 마지막 추론 결과를 사용하여 프레임에 박스 그리기
            if last_results is not None:
                annotated_frame = last_results[0].plot()
            else:
                annotated_frame = frame  # 아직 추론 결과가 없으면 원본 프레임 표시
            
            # 박스가 그려진 프레임 표시
            cv2.imshow(window_name, annotated_frame)

            key_code = cv2.waitKey(1) & 0xFF
            
            if key_code == ord('q'):
                break
            elif key_code == ord('w'):
                ptz_controller.start_continuous_move(pan_velocity=0.0, tilt_velocity=0.5)
            elif key_code == ord('s'):
                ptz_controller.start_continuous_move(pan_velocity=0.0, tilt_velocity=-0.5)
            elif key_code == ord('a'):
                ptz_controller.start_continuous_move(pan_velocity=-0.5, tilt_velocity=0.0)
            elif key_code == ord('d'):
                ptz_controller.start_continuous_move(pan_velocity=0.5, tilt_velocity=0.0)
            elif key_code == ord(' '):
                ptz_controller.stop_move()

    finally:
        ptz_controller.stop_move()
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()