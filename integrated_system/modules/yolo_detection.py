"""
YOLO 탐지 모듈 - Detaction_CCTV 서비스를 직접 import하여 BaseModule로 래핑

원본 모듈 (직접 import — 수정 시 즉시 반영):
    - Detaction_CCTV/services/vision_processor.py  → VisionProcessor
    - Detaction_CCTV/services/priority_manager.py  → VisualPriorityManager
    - Detaction_CCTV/services/reid_manager.py      → ReIDManager

통합 레이어 (이 파일에만 존재):
    - 상태 머신 (TRACKING / SEARCHING / PATROL)  ← Detaction_CCTV/main.py 참조
    - PID(P) 제어 → PTZ 추적                    ← Detaction_CCTV/main.py 참조
    - EventBus 이벤트 발행 / 구독
    - 시각화 오버레이 (디버그용)
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from integrated_system.core.base_module import BaseModule
from integrated_system.core.event_bus import EventBus, Event
from integrated_system.core.module_loader import DETECT_DIR, import_from_file
from integrated_system.modules.ptz_controller import UnifiedPTZController, PTZPriority

logger = logging.getLogger(__name__)


class YOLODetectionModule(BaseModule):
    """
    YOLO 기반 객체 탐지 + 추적 + 우선순위 모듈

    Detaction_CCTV/services/ 의 원본 서비스 클래스들을 직접 사용합니다.
    원본 파일을 수정하면 이 모듈에 즉시 반영됩니다.

    이벤트 발행:
        - yolo.objects_detected  : 객체 감지 시
        - yolo.person_detected   : 사람 감지 시 (ContextLLM 트리거)
        - yolo.no_objects        : 프레임에 객체 없을 때
    """

    PATROL_RETURN_DELAY = 3.0  # 순찰 복귀 딜레이 (초)

    def __init__(
        self,
        event_bus: EventBus,
        ptz: Optional[UnifiedPTZController] = None,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        pid_kp: float = 0.4,
        dead_zone: int = 50,
        patrol_speed: float = 0.2,
        target_classes: Optional[List[int]] = None,
        camera_fov_deg: float = 90.0,
        doa_boost_weight: float = 0.35,
        doa_memory_sec: float = 1.5,
    ):
        super().__init__(event_bus)
        self.ptz = ptz
        self.model_path = model_path
        self.confidence = confidence
        self.pid_kp = pid_kp
        self.dead_zone = dead_zone
        self.patrol_speed = patrol_speed
        self.target_classes = target_classes
        self.camera_fov_deg = camera_fov_deg
        self.doa_boost_weight = doa_boost_weight
        self.doa_memory_sec = doa_memory_sec

        # ── 원본 서비스 인스턴스 (initialize에서 생성) ──
        self._vision = None           # VisionProcessor   (vision_processor.py)
        self._priority_mgr = None     # VisualPriorityManager (priority_manager.py)
        self._reid_mgr = None         # ReIDManager        (reid_manager.py)

        # ── 통합 상태 머신 ──
        self._tracked_target: Optional[Dict] = None
        self._last_event_time = time.time()
        self._current_mode = "PATROL"
        self._latest_doa_angle: Optional[float] = None
        self._latest_doa_time: float = 0.0

    @property
    def name(self) -> str:
        return "yolo"

    def initialize(self) -> bool:
        """원본 서비스 클래스들을 import하고 인스턴스 생성"""
        try:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

            # ★ 원본 모듈 직접 파일 로드 (services/__init__.py의 onvif 의존성 우회) ★
            _vp = import_from_file("_orig_vision_processor", os.path.join(DETECT_DIR, "services", "vision_processor.py"))
            _pm = import_from_file("_orig_priority_manager", os.path.join(DETECT_DIR, "services", "priority_manager.py"))
            _rm = import_from_file("_orig_reid_manager", os.path.join(DETECT_DIR, "services", "reid_manager.py"))
            VisionProcessor = _vp.VisionProcessor
            VisualPriorityManager = _pm.VisualPriorityManager
            ReIDManager = _rm.ReIDManager

            self._vision = VisionProcessor(self.model_path, self.confidence)
            self._priority_mgr = VisualPriorityManager()
            self._reid_mgr = ReIDManager(similarity_threshold=0.75)
            self._event_bus.subscribe("mic.doa_detected", self._on_doa_detected)

            logger.info("[YOLO] 원본 서비스 로드 완료 (VisionProcessor, VisualPriorityManager, ReIDManager)")
            return True
        except Exception as e:
            logger.error(f"[YOLO] 초기화 실패: {e}")
            return False

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """프레임에서 객체 감지 → 우선순위 → PTZ 추적"""
        frame = shared_data.get("frame")
        if frame is None or self._vision is None:
            return {"objects": [], "person_detected": False, "mode": self._current_mode}

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # 1. YOLO 추론 (★ 원본 VisionProcessor.process_frame)
        raw_objects, annotated_frame = self._vision.process_frame(frame)

        # 원본 VisionProcessor는 'name' 필드를 포함하지 않으므로 YOLO 클래스명 추가
        yolo_names = self._vision.model.names  # {0: 'person', 1: 'bicycle', ...}
        for obj in raw_objects:
            obj['name'] = yolo_names.get(obj.get('cls', -1), 'unknown')

        # 2. Re-ID (★ 원본 ReIDManager.update_ids)
        identified = self._reid_mgr.update_ids(frame, raw_objects)

        # ReIDManager가 'name'을 "Person X"로 덮어쓰므로 YOLO 클래스명을 복원
        for obj in identified:
            obj['display_name'] = obj.get('name', '')  # "Person 1" (UI 표시용)
            obj['name'] = yolo_names.get(obj.get('cls', -1), 'unknown')  # PriorityManager용

        # 3. 우선순위 계산 (★ 원본 VisualPriorityManager.calculate_priorities)
        sorted_objects = self._priority_mgr.calculate_priorities(identified, w, h)
        sorted_objects = self._apply_doa_fusion(sorted_objects, w)

        # 4. 행동 결정 + PTZ 제어 (통합 레이어)
        person_detected = any(obj.get("name", "").lower() == "person" for obj in sorted_objects)
        mode, target = self._decide_action(sorted_objects, center_x, center_y)
        self._current_mode = mode

        # 5. 이벤트 발행
        if sorted_objects:
            self.emit("yolo.objects_detected", {
                "objects": sorted_objects,
                "count": len(sorted_objects),
                "mode": mode,
            })
            if person_detected:
                self.emit("yolo.person_detected", {
                    "objects": [o for o in sorted_objects if o.get("name", "").lower() == "person"],
                    "count": sum(1 for o in sorted_objects if o.get("name", "").lower() == "person"),
                    "target": target,
                }, priority=1)
        else:
            self.emit("yolo.no_objects", {"mode": mode})

        return {
            "objects": sorted_objects,
            "person_detected": person_detected,
            "mode": mode,
            "target": target,
            "priority": "HIGH" if person_detected else "LOW",
        }

    def _on_doa_detected(self, event: Event) -> None:
        sector = event.data.get("sector_angle")
        if sector is not None:
            self._latest_doa_angle = float(sector)
            self._latest_doa_time = time.time()

    def _apply_doa_fusion(self, objects: List[Dict], frame_width: int) -> List[Dict]:
        if not objects or frame_width <= 0:
            return objects

        if self._latest_doa_angle is None:
            return objects
        if time.time() - self._latest_doa_time > self.doa_memory_sec:
            return objects

        half_fov = max(1.0, self.camera_fov_deg / 2.0)
        fused = []
        for obj in objects:
            cx, _ = obj.get("center", (frame_width // 2, 0))
            rel = (float(cx) / float(frame_width)) - 0.5
            obj_angle = rel * self.camera_fov_deg

            doa_error = ((self._latest_doa_angle - obj_angle + 180.0) % 360.0) - 180.0
            doa_error_abs = abs(doa_error)
            alignment = max(0.0, 1.0 - min(doa_error_abs, half_fov) / half_fov)
            bonus = self.doa_boost_weight * alignment

            base = float(obj.get("priority_score", 0.0))
            obj["priority_score"] = base + bonus
            obj["doa_bonus"] = bonus
            fused.append(obj)

        fused.sort(key=lambda o: o.get("priority_score", 0.0), reverse=True)
        return fused

    # ─── 상태 머신 + PTZ (통합 레이어 — Detaction_CCTV/main.py 참조) ───

    def _decide_action(self, sorted_objects: List[Dict], cx: int, cy: int) -> tuple:
        """
        추적 / 탐색 / 순찰 상태 결정
        알고리즘 출처: Detaction_CCTV/main.py → SurveillanceSystemController.run()
        """
        if sorted_objects:
            self._last_event_time = time.time()

            if self._tracked_target:
                for obj in sorted_objects:
                    if obj.get('permanent_id') == self._tracked_target.get('permanent_id'):
                        self._tracked_target = obj
                        break
                else:
                    self._tracked_target = sorted_objects[0]
            else:
                self._tracked_target = sorted_objects[0]

            if self.ptz:
                tx, ty = self._tracked_target['center']
                pan, tilt = self._pid_output(tx, ty, cx, cy)
                self.ptz.request_move(pan, tilt, PTZPriority.YOLO_TRACKING, self.name)

            mode = f"TRACKING (ID:{self._tracked_target.get('permanent_id', -1)})"
            return mode, self._tracked_target
        else:
            if self._current_mode.startswith("TRACKING"):
                self._tracked_target = None
                if self.ptz:
                    self.ptz.stop()
                return "SEARCHING", None

            if time.time() - self._last_event_time > self.PATROL_RETURN_DELAY:
                if self.ptz:
                    self.ptz.request_move(self.patrol_speed, 0.0, PTZPriority.PATROL, self.name)
                return "PATROL", None

            return self._current_mode, None

    def _pid_output(self, tx: float, ty: float, cx: int, cy: int) -> tuple:
        """
        PID(P) 제어로 PTZ 속도 계산
        알고리즘 출처: Detaction_CCTV/main.py → _calculate_pid_output()
        """
        if cx == 0 or cy == 0:
            return 0.0, 0.0
        ex, ey = tx - cx, ty - cy
        pan = (ex / cx) * self.pid_kp if abs(ex) > self.dead_zone else 0.0
        tilt = -(ey / cy) * self.pid_kp if abs(ey) > self.dead_zone else 0.0
        return max(-1, min(1, pan)), max(-1, min(1, tilt))

    def get_annotated_frame(self, frame: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """
        시각화 오버레이 (디버그용)
        알고리즘 출처: Detaction_CCTV/main.py → _draw_overlay()
        """
        target_id = self._tracked_target.get('permanent_id') if self._tracked_target else None

        for obj in objects:
            is_target = target_id is not None and obj.get('permanent_id') == target_id
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            color = (0, 0, 255) if is_target else (0, 255, 255)
            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"ID:{obj.get('permanent_id', -1)} {obj.get('display_name', obj.get('name', ''))} ({obj.get('priority_score', 0):.2f})"
            if is_target:
                label += " [TARGET]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"MODE: {self._current_mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def shutdown(self) -> None:
        self._vision = None
        self._priority_mgr = None
        self._reid_mgr = None
        logger.info("[YOLO] 종료")
