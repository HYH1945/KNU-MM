"""
공간-시간 멀티모달 컨텍스트 트래커

시각(YOLO) + 청각(MicArray/STT) + 카메라(PTZ) 정보를 하나의 상태 객체로
통합 관리하여 진정한 멀티모달 "융합"을 가능하게 합니다.

역할:
    1. 카메라 방위각/앙각 추적 (PTZ 이동 명령 기반)
    2. YOLO 객체의 절대 방위각 추정 (프레임 내 위치 → 세계 좌표)
    3. DOA 방향과 YOLO 객체의 공간 매칭
    4. 시간 이벤트 히스토리 유지 (시간적 상관관계)
    5. LLM에 전달할 구조화된 컨텍스트 생성

사용법:
    ctx = SpatialContext(camera_fov=60.0)
    ctx.update_camera_position(120.0, -15.0)
    ctx.update_yolo_results([...], frame_shape=(480, 640))
    ctx.update_doa(150.0, confidence=0.85)
    ctx.update_stt("도와주세요!")

    match = ctx.check_spatial_match()   # DOA 방향에 사람이 있는가?
    context_str = ctx.build_llm_context()  # LLM에 전달할 텍스트
"""

import time
import threading
import logging
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SpatialContext:
    """
    시각-청각 멀티모달 공간/시간 컨텍스트

    모든 모듈이 이 객체를 공유하여 상태를 갱신하고,
    ContextLLM이 분석 시 통합 컨텍스트를 구성할 때 사용합니다.
    """

    def __init__(
        self,
        camera_fov: float = 60.0,
        spatial_match_threshold: float = 30.0,
        event_history_size: int = 50,
        event_history_duration: float = 60.0,
    ):
        """
        Args:
            camera_fov: 카메라 수평 시야각 (도)
            spatial_match_threshold: DOA↔YOLO 공간 매칭 허용 오차 (도)
            event_history_size: 최대 이벤트 히스토리 크기
            event_history_duration: 이벤트 유지 시간 (초)
        """
        self.camera_fov = camera_fov
        self.spatial_match_threshold = spatial_match_threshold
        self.event_history_duration = event_history_duration

        self._lock = threading.RLock()

        # ── 카메라 상태 ──
        self._camera_azimuth: float = 0.0        # 0~360
        self._camera_elevation: float = 0.0      # -90~90
        self._camera_position_time: float = 0.0

        # ── YOLO 감지 결과 ──
        self._detected_objects: List[Dict] = []
        self._person_detected: bool = False
        self._last_detection_time: float = 0.0
        self._frame_shape: Tuple[int, int] = (0, 0)  # (height, width)

        # ── 청각 상태 ──
        self._last_doa_angle: float = -1.0        # 원시 DOA (마이크 기준)
        self._last_doa_confidence: float = 0.0
        self._last_doa_time: float = 0.0

        self._last_stt_text: Optional[str] = None
        self._last_stt_time: float = 0.0

        # ── 융합 결과 ──
        self._spatial_match: bool = False
        self._match_details: Dict = {}

        # ── 시간 이벤트 히스토리 ──
        self._event_timeline: deque = deque(maxlen=event_history_size)

    # ═══════════════════════════════════════════════════════
    #  상태 갱신 메서드
    # ═══════════════════════════════════════════════════════

    def update_camera_position(self, azimuth: float, elevation: float = 0.0) -> None:
        """
        카메라 방위각/앙각 갱신

        Args:
            azimuth: 수평 방위각 (0~360, absolute 모드 기준)
            elevation: 수직 앙각 (-90~90)
        """
        with self._lock:
            self._camera_azimuth = azimuth % 360
            self._camera_elevation = elevation
            self._camera_position_time = time.time()

        logger.debug(
            f"[SpatialContext] 카메라 위치 갱신: "
            f"방위각={self._camera_azimuth:.1f}°, 앙각={elevation:.1f}°"
        )

    def update_camera_by_velocity(
        self, pan_speed: float, tilt_speed: float, dt: float
    ) -> None:
        """
        ContinuousMove 속도 기반 카메라 위치 추정

        Args:
            pan_speed: 수평 속도 (-1~1)
            tilt_speed: 수직 속도 (-1~1)
            dt: 경과 시간 (초)
        """
        # 경험적 속도 계수: 속도 1.0 = 약 90°/s (Hikvision PTZ 일반적 속도)
        SPEED_SCALE = 90.0
        with self._lock:
            self._camera_azimuth = (
                self._camera_azimuth + pan_speed * SPEED_SCALE * dt
            ) % 360
            self._camera_elevation = max(
                -90.0, min(90.0, self._camera_elevation + tilt_speed * SPEED_SCALE * dt)
            )
            self._camera_position_time = time.time()

    def update_yolo_results(
        self, objects: List[Dict], frame_shape: Tuple[int, int]
    ) -> None:
        """
        YOLO 감지 결과 갱신 + 객체별 절대 방위각 추정

        Args:
            objects: YOLO 감지 객체 리스트
                각 객체는 {name, box, center, priority_score, ...} 포함
            frame_shape: (height, width)
        """
        now = time.time()
        with self._lock:
            self._frame_shape = frame_shape
            h, w = frame_shape

            enriched = []
            for obj in objects:
                obj_copy = dict(obj)
                # 프레임 내 x위치 → 절대 방위각 추정
                if w > 0 and "center" in obj_copy:
                    cx = obj_copy["center"][0]
                    # 프레임 왼쪽 = camera_azimuth - fov/2
                    # 프레임 오른쪽 = camera_azimuth + fov/2
                    normalized_x = cx / w  # 0.0 ~ 1.0
                    world_angle = (
                        self._camera_azimuth
                        - self.camera_fov / 2
                        + normalized_x * self.camera_fov
                    ) % 360
                    obj_copy["world_angle"] = world_angle
                enriched.append(obj_copy)

            self._detected_objects = enriched
            self._person_detected = any(
                obj.get("name", "").lower() == "person" for obj in enriched
            )
            self._last_detection_time = now

            # 공간 매칭 재계산
            self._recalc_spatial_match()

        # 이벤트 기록
        if self._person_detected:
            person_count = sum(
                1 for o in enriched if o.get("name", "").lower() == "person"
            )
            self.add_event("person_detected", {
                "count": person_count,
                "camera_azimuth": self._camera_azimuth,
            })

    def update_doa(self, angle: float, confidence: float) -> None:
        """
        MicArray DOA 결과 갱신

        Args:
            angle: DOA 각도 (마이크 기준 0~360, 마이크와 카메라가 정렬되어 있으므로 = 절대 방위각)
            confidence: 방향 신뢰도 (0~1)
        """
        now = time.time()
        with self._lock:
            self._last_doa_angle = angle
            self._last_doa_confidence = confidence
            self._last_doa_time = now

            # 공간 매칭 재계산
            self._recalc_spatial_match()

        self.add_event("doa_detected", {
            "angle": angle,
            "confidence": confidence,
        })

        logger.debug(
            f"[SpatialContext] DOA 갱신: {angle:.0f}° "
            f"(신뢰도={confidence:.2f}, 매칭={self._spatial_match})"
        )

    def update_stt(self, text: str) -> None:
        """STT 텍스트 갱신"""
        now = time.time()
        with self._lock:
            self._last_stt_text = text
            self._last_stt_time = now

        self.add_event("stt_recognized", {"text": text})
        logger.debug(f'[SpatialContext] STT 갱신: "{text}"')

    # ═══════════════════════════════════════════════════════
    #  공간 매칭 로직
    # ═══════════════════════════════════════════════════════

    def _recalc_spatial_match(self) -> None:
        """DOA 방향에 YOLO 객체가 있는지 재계산 (내부/lock 내에서 호출)"""
        self._spatial_match = False
        self._match_details = {}

        if self._last_doa_angle < 0 or not self._detected_objects:
            return

        # DOA 이벤트가 5초 이내인지 확인
        if time.time() - self._last_doa_time > 5.0:
            return

        doa = self._last_doa_angle
        best_match = None
        best_diff = float("inf")

        for obj in self._detected_objects:
            if "world_angle" not in obj:
                continue

            # 원형 각도 차이 계산 (0~180)
            diff = abs(doa - obj["world_angle"])
            if diff > 180:
                diff = 360 - diff

            if diff < self.spatial_match_threshold and diff < best_diff:
                best_diff = diff
                best_match = obj

        if best_match is not None:
            self._spatial_match = True
            self._match_details = {
                "matched_object": best_match.get("name", "unknown"),
                "object_id": best_match.get("permanent_id", -1),
                "doa_angle": doa,
                "object_angle": best_match.get("world_angle", -1),
                "angle_diff": best_diff,
                "is_person": best_match.get("name", "").lower() == "person",
            }

    def check_spatial_match(self) -> bool:
        """DOA 방향에 YOLO 객체가 있는지 확인"""
        with self._lock:
            return self._spatial_match

    def get_match_details(self) -> Dict:
        """공간 매칭 상세 정보 반환"""
        with self._lock:
            return dict(self._match_details)

    # ═══════════════════════════════════════════════════════
    #  이벤트 타임라인
    # ═══════════════════════════════════════════════════════

    def add_event(self, event_type: str, data: Dict) -> None:
        """타임라인에 이벤트 추가"""
        event = {
            "time": time.time(),
            "type": event_type,
            "data": data,
        }
        with self._lock:
            self._event_timeline.append(event)

    def get_recent_events(self, seconds: float = 30.0) -> List[Dict]:
        """최근 N초 이벤트 조회"""
        cutoff = time.time() - seconds
        with self._lock:
            return [e for e in self._event_timeline if e["time"] > cutoff]

    # ═══════════════════════════════════════════════════════
    #  LLM 컨텍스트 빌더
    # ═══════════════════════════════════════════════════════

    def build_llm_context(self) -> str:
        """
        LLM에 전달할 구조화된 멀티모달 컨텍스트 문자열 생성

        Returns:
            LLM additional_context에 포함할 텍스트
        """
        with self._lock:
            parts = []

            # ── 1. 카메라 상태 ──
            parts.append(
                f"[카메라] 방위각: {self._camera_azimuth:.0f}°, "
                f"앙각: {self._camera_elevation:.0f}°, "
                f"시야각: {self.camera_fov:.0f}° "
                f"(시야 범위: {(self._camera_azimuth - self.camera_fov/2) % 360:.0f}°~"
                f"{(self._camera_azimuth + self.camera_fov/2) % 360:.0f}°)"
            )

            # ── 2. YOLO 감지 결과 ──
            if self._detected_objects:
                obj_descriptions = []
                for obj in self._detected_objects:
                    name = obj.get("name", "unknown")
                    pid = obj.get("permanent_id", "?")
                    score = obj.get("priority_score", 0)
                    world_angle = obj.get("world_angle", -1)
                    desc = f"  - {name} (ID:{pid}, 방위각:{world_angle:.0f}°, 우선도:{score:.2f})"
                    obj_descriptions.append(desc)

                parts.append(
                    f"[시각 감지] {len(self._detected_objects)}개 객체 "
                    f"(사람 감지: {'예' if self._person_detected else '아니오'}):\n"
                    + "\n".join(obj_descriptions)
                )
            else:
                age = time.time() - self._last_detection_time if self._last_detection_time > 0 else -1
                if age > 0:
                    parts.append(f"[시각 감지] 현재 감지된 객체 없음 ({age:.0f}초 전 마지막 감지)")
                else:
                    parts.append("[시각 감지] 감지된 객체 없음")

            # ── 3. 청각 상태 ──
            if self._last_doa_angle >= 0 and (time.time() - self._last_doa_time < 10):
                parts.append(
                    f"[음원 방향] DOA: {self._last_doa_angle:.0f}°, "
                    f"신뢰도: {self._last_doa_confidence:.2f}"
                )

            if self._last_stt_text and (time.time() - self._last_stt_time < 30):
                parts.append(f'[음성 인식] "{self._last_stt_text}"')

            # ── 4. 공간 매칭 결과 ──
            if self._spatial_match and self._match_details:
                md = self._match_details
                parts.append(
                    f"[공간 매칭] ⚠️ 소리 방향({md['doa_angle']:.0f}°)과 "
                    f"{md['matched_object']}(ID:{md['object_id']}, "
                    f"{md['object_angle']:.0f}°)이 일치 "
                    f"(각도 차이: {md['angle_diff']:.0f}°)"
                )
            elif self._last_doa_angle >= 0 and self._detected_objects:
                parts.append(
                    "[공간 매칭] 소리 방향과 감지된 객체가 일치하지 않음 "
                    "(소리와 시각 대상이 서로 다른 위치)"
                )

            # ── 5. 최근 이벤트 요약 ──
            recent = self.get_recent_events(30.0)
            if recent:
                event_summary = []
                for ev in recent[-5:]:  # 최근 5개만
                    elapsed = time.time() - ev["time"]
                    event_summary.append(f"  - {elapsed:.0f}초 전: {ev['type']}")
                parts.append(
                    f"[최근 이벤트] ({len(recent)}개, 최근 30초):\n"
                    + "\n".join(event_summary)
                )

            return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    #  프로퍼티 (읽기 전용)
    # ═══════════════════════════════════════════════════════

    @property
    def camera_azimuth(self) -> float:
        return self._camera_azimuth

    @property
    def camera_elevation(self) -> float:
        return self._camera_elevation

    @property
    def person_detected(self) -> bool:
        with self._lock:
            return self._person_detected

    @property
    def detected_objects(self) -> List[Dict]:
        with self._lock:
            return list(self._detected_objects)

    @property
    def last_doa_angle(self) -> float:
        return self._last_doa_angle

    @property
    def last_stt_text(self) -> Optional[str]:
        with self._lock:
            if self._last_stt_text and (time.time() - self._last_stt_time < 30):
                return self._last_stt_text
            return None

    @property
    def spatial_match(self) -> bool:
        with self._lock:
            return self._spatial_match

    def get_snapshot(self) -> Dict[str, Any]:
        """현재 전체 상태 스냅샷 (디버그/대시보드용)"""
        with self._lock:
            return {
                "camera": {
                    "azimuth": self._camera_azimuth,
                    "elevation": self._camera_elevation,
                    "fov": self.camera_fov,
                },
                "yolo": {
                    "object_count": len(self._detected_objects),
                    "person_detected": self._person_detected,
                    "objects": [
                        {
                            "name": o.get("name"),
                            "id": o.get("permanent_id"),
                            "world_angle": o.get("world_angle"),
                        }
                        for o in self._detected_objects
                    ],
                },
                "audio": {
                    "doa_angle": self._last_doa_angle,
                    "doa_confidence": self._last_doa_confidence,
                    "stt_text": self._last_stt_text,
                },
                "fusion": {
                    "spatial_match": self._spatial_match,
                    "match_details": dict(self._match_details),
                },
                "event_count": len(self._event_timeline),
            }
