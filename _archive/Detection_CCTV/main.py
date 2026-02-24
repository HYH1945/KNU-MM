import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import time
from typing import Tuple, List, Dict, Optional

from config import AppConfig
from services import (
    VideoStreamHandler, 
    PTZCameraManager, 
    VisionProcessor, 
    VisualPriorityManager,
    ReIDManager,
    HeatmapOverlay,
)

class SurveillanceSystemController:
    """스마트 CCTV 시스템의 메인 컨트롤러"""

    PATROL_RETURN_DELAY_SECONDS = 3.0
    MODE_PATROL = "PATROL"
    MODE_SEARCHING = "SEARCHING"
    MODE_TRACKING = "TRACKING"
    MODE_MANUAL_SUFFIX = " [MANUAL]"

    def __init__(self):
        print("[System] Initializing...")
        self.config = AppConfig()
        self.skip_frames = max(1, int(self.config.AI_SKIP_FRAMES))
        
        # 서비스 모듈 초기화
        self.stream_handler = VideoStreamHandler(self.config.RTSP_URL).start()
        self.ptz = PTZCameraManager(self.config)
        self.vision = VisionProcessor(
            self.config.AI_MODEL_PATH,
            self.config.AI_CONFIDENCE,
            device=self.config.AI_DEVICE,
            tracker_cfg=self.config.AI_TRACKER_CFG,
            use_tracking=self.config.AI_USE_TRACKING,
        )
        self.priority_manager = VisualPriorityManager()
        self.reid_manager = ReIDManager(similarity_threshold=0.75)
        self.heatmap = HeatmapOverlay(
            alpha=self.config.HEATMAP_ALPHA,
            decay=self.config.HEATMAP_DECAY,
            radius=self.config.HEATMAP_RADIUS,
            intensity=self.config.HEATMAP_INTENSITY,
            downscale=self.config.HEATMAP_DOWNSCALE,
            min_value=self.config.HEATMAP_MIN_VALUE,
        )
        self.heatmap_enabled: bool = bool(self.config.SHOW_HEATMAP)
        
        # 시스템 상태 변수
        self.is_running: bool = True
        self.current_mode: str = self.MODE_PATROL
        self.last_event_time: float = time.time()
        self.tracked_target: Optional[Dict] = None
        
        self.center_x, self.center_y = 0, 0
        self.frame_count: int = 0
        self.last_sorted_objects: List[Dict] = []
        self.last_manual_time: float = 0.0

    def _is_manual_override_active(self) -> bool:
        if not self.config.ENABLE_MANUAL_CONTROL:
            return False
        if self.config.MANUAL_OVERRIDE_SECONDS <= 0:
            return False
        return (time.time() - self.last_manual_time) < self.config.MANUAL_OVERRIDE_SECONDS

    def _handle_key(self, key_code: int) -> bool:
        if key_code == ord('q'):
            return False

        if key_code == ord('h'):
            self.heatmap_enabled = not self.heatmap_enabled
            print(f"[UI] Heatmap {'ON' if self.heatmap_enabled else 'OFF'}")
            return True

        if key_code == ord('r'):
            self.heatmap.clear()
            print("[UI] Heatmap reset")
            return True

        if not self.config.ENABLE_MANUAL_CONTROL:
            return True

        if key_code == ord('w'):
            self.ptz.start_continuous_move(pan_velocity=0.0, tilt_velocity=0.5)
        elif key_code == ord('s'):
            self.ptz.start_continuous_move(pan_velocity=0.0, tilt_velocity=-0.5)
        elif key_code == ord('a'):
            self.ptz.start_continuous_move(pan_velocity=-0.5, tilt_velocity=0.0)
        elif key_code == ord('d'):
            self.ptz.start_continuous_move(pan_velocity=0.5, tilt_velocity=0.0)
        elif key_code == ord(' '):
            self.ptz.stop_move()
        else:
            return True

        self.last_manual_time = time.time()
        return True

    def _calculate_pid_output(self, target_cx: float, target_cy: float) -> Tuple[float, float]:
        """PID 제어를 통해 모터 속도 계산 (P 제어기)"""
        # 0으로 나누기 오류 방지
        if self.center_x == 0 or self.center_y == 0:
            return 0.0, 0.0

        error_x = target_cx - self.center_x
        error_y = target_cy - self.center_y
        
        pan_velocity = 0.0
        tilt_velocity = 0.0

        if abs(error_x) > self.config.DEAD_ZONE_PIXELS:
            pan_velocity = (error_x / self.center_x) * self.config.PID_KP
            
        if abs(error_y) > self.config.DEAD_ZONE_PIXELS:
            tilt_velocity = -(error_y / self.center_y) * self.config.PID_KP
            
        return max(-1.0, min(1.0, pan_velocity)), max(-1.0, min(1.0, tilt_velocity))

    def _draw_overlay(
        self,
        frame,
        all_objects: List[Dict],
        mode_label: str,
        fps: Optional[float] = None,
        heatmap_enabled: bool = False,
    ):
        """프레임에 객체 정보 및 현재 상태를 그리는 함수"""
        target_id = self.tracked_target.get('permanent_id') if self.tracked_target else None

        for obj in all_objects:
            is_target = (target_id is not None) and (obj['permanent_id'] == target_id)
            
            x1, y1, x2, y2 = obj['box']
            name = obj.get('name', 'unknown')
            score = obj.get('priority_score', 0)
            
            color = (0, 0, 255) if is_target else (0, 255, 255)
            thickness = 3 if is_target else 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{obj.get('permanent_id', -1)} {name} ({score:.2f})"
            if is_target:
                label += " [TARGET]"
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self._draw_status_text(frame, mode_label, fps, heatmap_enabled)

    def _draw_status_text(
        self,
        frame,
        mode_label: str,
        fps: Optional[float] = None,
        heatmap_enabled: bool = False,
    ):
        cv2.putText(frame, f"MODE: {mode_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"HEATMAP: {'ON' if heatmap_enabled else 'OFF'} (h:toggle, r:reset)",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


    def run(self):
        """메인 실행 루프"""
        print("[System] System Started.")
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.skip_frames > 1:
            print(f"[System] YOLO inference: every {self.skip_frames} frames")

        fps = 0.0
        frames_since_last = 0
        last_fps_time = time.perf_counter()

        try:
            while self.is_running:
                frame = self.stream_handler.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_h, frame_w = frame.shape[:2]
                self.center_x, self.center_y = frame_w // 2, frame_h // 2

                frames_since_last += 1
                elapsed = time.perf_counter() - last_fps_time
                if elapsed >= 1.0:
                    fps = frames_since_last / elapsed
                    frames_since_last = 0
                    last_fps_time = time.perf_counter()

                self.frame_count += 1
                did_infer = (self.frame_count % self.skip_frames == 0)
                if did_infer:
                    raw_objects, _ = self.vision.process_frame(frame)
                    identified_objects = self.reid_manager.update_ids(frame, raw_objects)
                    self.last_sorted_objects = self.priority_manager.calculate_priorities(
                        identified_objects,
                        frame_w,
                        frame_h,
                    )

                sorted_objects = self.last_sorted_objects
                
                manual_override_active = self._is_manual_override_active()
                if sorted_objects:
                    self.last_event_time = time.time()

                    current_target_still_visible = False
                    if self.tracked_target:
                        for obj in sorted_objects:
                            if obj['permanent_id'] == self.tracked_target['permanent_id']:
                                self.tracked_target = obj
                                current_target_still_visible = True
                                break

                    if not current_target_still_visible:
                        self.tracked_target = sorted_objects[0]

                    self.current_mode = (
                        f"{self.MODE_TRACKING} (ID: {self.tracked_target.get('permanent_id', -1)})"
                    )
                    tx, ty = self.tracked_target['center']
                    pan, tilt = self._calculate_pid_output(tx, ty)
                    if not manual_override_active:
                        self.ptz.move_async(pan, tilt)

                else:
                    if self.current_mode.startswith(self.MODE_TRACKING):
                        self.tracked_target = None
                        self.current_mode = self.MODE_SEARCHING
                        if not manual_override_active:
                            self.ptz.stop()

                    if time.time() - self.last_event_time > self.PATROL_RETURN_DELAY_SECONDS:
                        self.current_mode = self.MODE_PATROL
                        if not manual_override_active:
                            self.ptz.move_async(self.config.PATROL_SPEED, 0.0)

                mode_label = self.current_mode
                if manual_override_active:
                    mode_label = f"{self.current_mode}{self.MODE_MANUAL_SUFFIX}"
                if self.heatmap_enabled:
                    self.heatmap.update(frame, sorted_objects, add_points=did_infer)
                    frame = self.heatmap.apply(frame)
                else:
                    self.heatmap.update(frame, sorted_objects, add_points=False)
                self._draw_overlay(
                    frame,
                    sorted_objects,
                    mode_label,
                    fps if self.config.SHOW_FPS else None,
                    self.heatmap_enabled,
                )
                cv2.imshow(self.config.WINDOW_NAME, frame)
                
                key_code = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key_code):
                    self.is_running = False

        except KeyboardInterrupt:
            print("\n[System] Forced Stop.")
        finally:
            self._shutdown()

    def _shutdown(self):
        print("[System] Shutting down...")
        self.is_running = False
        time.sleep(0.5) 
        
        if hasattr(self, 'stream_handler'): self.stream_handler.release()
        if hasattr(self, 'ptz'): self.ptz.stop()
            
        cv2.destroyAllWindows()
        print("[System] Succeeded to Shutdown")

if __name__ == "__main__":
    app = SurveillanceSystemController()
    app.run()
