import cv2
import threading
import os
from dotenv import load_dotenv
from onvif import ONVIFCamera


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
    ptz_controller = PTZCameraController(CAMERA_IP, CAMERA_PORT, CAMERA_USER, CAMERA_PASSWORD)

    video_capture = cv2.VideoCapture(RTSP_URL)
    window_name = "CCTV Monitoring System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if not video_capture.isOpened():
        print("Error: Cannot open video stream.")
        return

    print("Controls: [W/A/S/D] Move, [Space] Stop, [Q] Quit")

    frame_count = 0

    try:
        while True:
            is_frame_read, frame = video_capture.read()
            
            if not is_frame_read:
                print("Video stream disconnected.")
                break
            
            frame_count += 1
            
            cv2.imshow(window_name, frame)

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