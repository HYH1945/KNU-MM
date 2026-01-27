import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import time
import sys
from typing import Tuple

from config import AppConfig
from services import (
    VideoStreamHandler, 
    PTZCameraManager, 
    VisionProcessor, 
    VisualPriorityManager,
    ReIDManager 
)

class SurveillanceSystemController:
    """스마트 CCTV 시스템의 메인 컨트롤러"""

    def __init__(self):
        print("[System] Initializing...")
        self.config = AppConfig()
        
        # 서비스 모듈 초기화
        self.stream_handler = VideoStreamHandler(self.config.RTSP_URL).start()
        self.ptz = PTZCameraManager(self.config)
        self.vision = VisionProcessor(self.config.AI_MODEL_PATH, self.config.AI_CONFIDENCE)
        self.priority_manager = VisualPriorityManager()
        self.reid_manager = ReIDManager(similarity_threshold=0.75)
        
        # 시스템 상태 변수
        self.is_running: bool = True
        
        # [변경] 초기 모드를 바로 순찰로 설정 (IDLE 없음)
        self.current_mode: str = "AUTO PATROL"
        
        # [삭제됨] self.last_action_time 변수는 이제 필요 없어서 지움
        
        self.center_x, self.center_y = 0, 0

    def _calculate_pid_output(self, target_cx: float, target_cy: float) -> Tuple[float, float]:
        """PID 제어를 통해 모터 속도 계산"""
        error_x = target_cx - self.center_x
        error_y = target_cy - self.center_y
        
        pan_velocity = 0.0
        tilt_velocity = 0.0

        if abs(error_x) > self.config.DEAD_ZONE_PIXELS:
            pan_velocity = (error_x / self.center_x) * self.config.PID_KP
            
        if abs(error_y) > self.config.DEAD_ZONE_PIXELS:
            tilt_velocity = -(error_y / self.center_y) * self.config.PID_KP
            
        return max(-1, min(1, pan_velocity)), max(-1, min(1, tilt_velocity))

    def _draw_overlay(self, frame, all_objects, target_obj):
        """그리기 함수"""
        for obj in all_objects:
            is_target = (target_obj is not None) and (obj['permanent_id'] == target_obj['permanent_id'])
            
            x1, y1, x2, y2 = obj['box']
            name = obj['name'] 
            
            color = (0, 0, 255) if is_target else (0, 255, 255)
            thickness = 3 if is_target else 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{name}"
            if is_target: label += " [TARGET]"
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def run(self):
        """메인 실행 루프"""
        print("[System] System Started. (Continuous Patrol Mode)")
        cv2.namedWindow("Smart CCTV", cv2.WINDOW_NORMAL)

        try:
            while self.is_running:
                # 1. 프레임 획득
                frame = self.stream_handler.get_frame()
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                self.center_x, self.center_y = w // 2, h // 2
                
                # 2. AI 인지 (YOLO)
                raw_objects, _ = self.vision.process_frame(frame)
                
                # Re-ID 처리
                identified_objects = self.reid_manager.update_ids(frame, raw_objects)
                
                # 3. 우선순위 결정
                target = self.priority_manager.select_target(identified_objects, w)
                
                # 4. 행동 결정 (State Machine)
                if target:
                    # [상태 A] 타겟 발견 -> 추적 모드
                    self.current_mode = f"TRACKING ({target['name']})"
                    
                    tx, ty = target['center']
                    pan, tilt = self._calculate_pid_output(tx, ty)
                    self.ptz.move_async(pan, tilt)
                    
                else:
                    time.sleep(3)
                    self.current_mode = "PATROL"
                    self.ptz.move_async(self.config.PATROL_SPEED, 0.0)

                # 5. 시각화
                self._draw_overlay(frame, identified_objects, target)

                # 상태 정보 표시
                cv2.putText(frame, f"MODE: {self.current_mode}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 중앙 십자선
                cv2.line(frame, (self.center_x-10, self.center_y), (self.center_x+10, self.center_y), (0,0,255), 1)
                cv2.line(frame, (self.center_x, self.center_y-10), (self.center_x, self.center_y+10), (0,0,255), 1)

                cv2.imshow("Smart CCTV", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
        print("[System] Suceessed to Shutdown")

if __name__ == "__main__":
    app = SurveillanceSystemController()
    app.run()