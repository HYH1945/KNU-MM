import threading

from onvif import ONVIFCamera


class PTZCameraController:
    def __init__(self, ip, port, user, password):
        try:
            self.cam = ONVIFCamera(ip, port, user, password)
            self.ptz_service = self.cam.create_ptz_service()
            media_service = self.cam.create_media_service()

            self.profile_token = media_service.GetProfiles()[0].token

            self.move_request = self.ptz_service.create_type("ContinuousMove")
            self.move_request.ProfileToken = self.profile_token

            self.is_connected = True
            print("PTZ Camera Connected Successfully.")
        except Exception as exc:
            print(f"Connection Error: {exc}")
            self.is_connected = False

    def _execute_async(self, func):
        threading.Thread(target=func).start()

    def start_continuous_move(self, pan_velocity, tilt_velocity):
        if not self.is_connected:
            return

        def send_move_command():
            try:
                self.move_request.Velocity = {
                    "PanTilt": {"x": pan_velocity, "y": tilt_velocity},
                    "Zoom": {"x": 0.0},
                }
                self.ptz_service.ContinuousMove(self.move_request)
            except Exception as exc:
                print(f"Move Error: {exc}")

        self._execute_async(send_move_command)

    def stop_move(self):
        if not self.is_connected:
            return

        def send_stop_command():
            try:
                self.ptz_service.Stop({"ProfileToken": self.profile_token})
            except Exception as exc:
                print(f"Stop Error: {exc}")

        self._execute_async(send_stop_command)
