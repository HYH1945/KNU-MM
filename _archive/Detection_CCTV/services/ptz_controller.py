import threading
from onvif import ONVIFCamera
from config import AppConfig

class PTZCameraManager:
    def __init__(self, config: AppConfig):
        self._connected: bool = False
        self._profile_token: str = ""
        self._log_errors: bool = bool(getattr(config, "PTZ_LOG_ERRORS", False))
        
        try:
            self.cam = ONVIFCamera(
                config.CAMERA_IP, config.CAMERA_PORT, 
                config.CAMERA_USER, config.CAMERA_PASSWORD
            )
            self.ptz_service = self.cam.create_ptz_service()
            media_service = self.cam.create_media_service()
            self._profile_token = media_service.GetProfiles()[0].token
            
            self._move_request = self.ptz_service.create_type('ContinuousMove')
            self._move_request.ProfileToken = self._profile_token
            
            self._connected = True
            print("[PTZ] Camera control connected.")
        except Exception as e:
            print(f"[PTZ] Connection failed: {e}")
            if self._log_errors:
                print(f"[PTZ] Exception details: {e}")

    def move_async(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        # MainThread blocking Async PTZ Move Command
        if not self._connected: return
        threading.Thread(
            target=self._send_command, args=(pan, tilt, zoom), daemon=True
        ).start()

    def _send_command(self, pan: float, tilt: float, zoom: float) -> None:
        try:
            self._move_request.Velocity = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }
            self.ptz_service.ContinuousMove(self._move_request)
        except Exception as e:
            if self._log_errors:
                print(f"[PTZ] Move error: {e}")

    def stop(self) -> None:
        if not self._connected: return
        threading.Thread(target=self._stop_routine, daemon=True).start()

    def _stop_routine(self):
        try:
            self.ptz_service.Stop({'ProfileToken': self._profile_token})
        except Exception as e:
            if self._log_errors:
                print(f"[PTZ] Stop error: {e}")

    # --- Compatibility helpers (YOLOv8 controller naming) ---
    def start_continuous_move(self, pan_velocity: float, tilt_velocity: float, zoom: float = 0.0) -> None:
        self.move_async(pan_velocity, tilt_velocity, zoom)

    def stop_move(self) -> None:
        self.stop()
