import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import time
import sys
from typing import Tuple, List, Dict, Optional

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

    # --- 상수 정의 ---
    PATROL_RETURN_DELAY_SECONDS = 3.0  # 마지막 객체 탐지 후 순찰 모드로 복귀하기까지 대기 시간

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
        self.current_mode: str = "PATROL"  # 초기 모드는 순찰
        self.last_event_time: float = time.time() # 마지막 유의미한 이벤트(객체 탐지) 시간
        self.tracked_target: Optional[Dict] = None # 현재 추적 중인 객체 정보
        
        self.center_x, self.center_y = 0, 0

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

    def _draw_overlay(self, frame, all_objects: List[Dict]):
        """프레임에 객체 정보 및 현재 상태를 그리는 함수"""
        # 현재 추적중인 타겟 ID 확인
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
        
        # 전체 시스템 모드 정보 표시
        cv2.putText(frame, f"MODE: {self.current_mode}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    def run(self):
        """메인 실행 루프"""
        print("[System] System Started.")
        cv2.namedWindow("Smart CCTV", cv2.WINDOW_NORMAL)

        try:
            while self.is_running:
                frame = self.stream_handler.get_frame()
                if frame is None:
                    time.sleep(0.01) # 프레임이 없을 경우 CPU 과부하 방지
                    continue
                
                h, w = frame.shape[:2]
                self.center_x, self.center_y = w // 2, h // 2
                
                # 1. AI 인지 (YOLO + Re-ID)
                raw_objects, _ = self.vision.process_frame(frame)
                identified_objects = self.reid_manager.update_ids(frame, raw_objects)
                
                # 2. 우선순위 결정 (점수가 높은 순으로 정렬된 리스트)
                sorted_objects = self.priority_manager.calculate_priorities(identified_objects, w, h)
                
                # 3. 행동 결정 (상태 머신)
                # 3-1. 추적할 객체가 하나 이상 존재하는 경우
                if sorted_objects:
                    self.last_event_time = time.time() # 마지막 객체 탐지 시간 갱신
                    
                    # 현재 추적하던 타겟이 계속 보이는지 확인
                    current_target_still_visible = False
                    if self.tracked_target:
                        for obj in sorted_objects:
                            if obj['permanent_id'] == self.tracked_target['permanent_id']:
                                self.tracked_target = obj # 최신 정보로 업데이트
                                current_target_still_visible = True
                                break
                    
                    # 현재 타겟이 안보이면, 최우선 순위 객체를 새로운 타겟으로 설정
                    if not current_target_still_visible:
                        self.tracked_target = sorted_objects[0]
                    
                    # 타겟팅 및 PTZ 제어
                    target_name = self.tracked_target.get('name', 'Object')
                    self.current_mode = f"TRACKING (ID: {self.tracked_target.get('permanent_id', -1)})"
                    
                    tx, ty = self.tracked_target['center']
                    pan, tilt = self._calculate_pid_output(tx, ty)
                    self.ptz.move_async(pan, tilt)

                # 3-2. 추적할 객체가 아무도 없는 경우
                else:
                    # 타겟을 잃어버린 직후라면 잠시 대기
                    if self.current_mode.startswith("TRACKING"):
                        self.tracked_target = None
                        self.current_mode = "SEARCHING"
                        self.ptz.stop() # 카메라 움직임 정지
                    
                    # 대기 시간(SEARCHING)이 충분히 지났다면 순찰 모드로 전환
                    if time.time() - self.last_event_time > self.PATROL_RETURN_DELAY_SECONDS:
                        self.current_mode = "PATROL"
                        self.ptz.move_async(self.config.PATROL_SPEED, 0.0)

                # 4. 시각화
                self._draw_overlay(frame, sorted_objects)
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
        print("[System] Succeeded to Shutdown")

if __name__ == "__main__":
    app = SurveillanceSystemController()
    app.run()