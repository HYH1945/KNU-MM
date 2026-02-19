"""
SpatialContext 단위 테스트

하드웨어 없이 실행 가능한 순수 로직 테스트:
    - 공간 매칭 (DOA ↔ YOLO)
    - 월드 좌표 추정 (frame 내 위치 → 방위각)
    - 카메라 위치 추적 (absolute / velocity)
    - 이벤트 히스토리
    - LLM 컨텍스트 빌드

실행:
    pytest integrated_system/tests/test_spatial_context.py -v
"""

import time
import pytest
from unittest.mock import patch

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from integrated_system.core.spatial_context import SpatialContext


# ────────────────────────────────────────────
# 헬퍼: 테스트용 YOLO 객체 생성
# ────────────────────────────────────────────

def make_person(pid: int, center_x: int, center_y: int = 300, score: float = 0.8):
    """사람 객체 생성 (bbox는 center ± 50 기본)"""
    return {
        "name": "person",
        "permanent_id": pid,
        "center": (center_x, center_y),
        "bbox": [center_x - 50, center_y - 100, center_x + 50, center_y + 100],
        "priority_score": score,
    }


def make_car(pid: int, center_x: int, center_y: int = 300, score: float = 0.3):
    """차량 객체 생성"""
    return {
        "name": "car",
        "permanent_id": pid,
        "center": (center_x, center_y),
        "bbox": [center_x - 80, center_y - 40, center_x + 80, center_y + 40],
        "priority_score": score,
    }


# ════════════════════════════════════════════
# 1. 초기 상태 테스트
# ════════════════════════════════════════════

class TestInitialization:
    def test_default_values(self):
        ctx = SpatialContext()
        assert ctx.camera_fov == 60.0
        assert ctx.spatial_match_threshold == 30.0
        assert ctx.camera_azimuth == 0.0
        assert ctx.camera_elevation == 0.0
        assert ctx.person_detected is False
        assert ctx.spatial_match is False
        assert ctx.last_doa_angle == -1.0
        assert ctx.last_stt_text is None
        assert ctx.detected_objects == []

    def test_custom_fov(self):
        ctx = SpatialContext(camera_fov=90.0, spatial_match_threshold=20.0)
        assert ctx.camera_fov == 90.0
        assert ctx.spatial_match_threshold == 20.0


# ════════════════════════════════════════════
# 2. 카메라 위치 추적 테스트
# ════════════════════════════════════════════

class TestCameraPositionTracking:
    def test_absolute_position_update(self):
        ctx = SpatialContext()
        ctx.update_camera_position(180.0, -15.0)
        assert ctx.camera_azimuth == 180.0
        assert ctx.camera_elevation == -15.0

    def test_azimuth_wraps_360(self):
        ctx = SpatialContext()
        ctx.update_camera_position(400.0, 0.0)
        assert ctx.camera_azimuth == 40.0  # 400 % 360 = 40

    def test_negative_azimuth_wraps(self):
        ctx = SpatialContext()
        ctx.update_camera_position(-30.0, 0.0)
        assert ctx.camera_azimuth == 330.0  # -30 % 360 = 330

    def test_velocity_pan_right(self):
        ctx = SpatialContext()
        ctx.update_camera_position(100.0, 0.0)
        # pan_speed=1.0, dt=1.0 → +90° (SPEED_SCALE=90)
        ctx.update_camera_by_velocity(pan_speed=1.0, tilt_speed=0.0, dt=1.0)
        assert ctx.camera_azimuth == pytest.approx(190.0, abs=0.1)

    def test_velocity_pan_left(self):
        ctx = SpatialContext()
        ctx.update_camera_position(100.0, 0.0)
        # pan_speed=-0.5, dt=2.0 → -90° (−0.5 × 90 × 2 = −90)
        ctx.update_camera_by_velocity(pan_speed=-0.5, tilt_speed=0.0, dt=2.0)
        assert ctx.camera_azimuth == pytest.approx(10.0, abs=0.1)

    def test_velocity_tilt_clamped(self):
        ctx = SpatialContext()
        ctx.update_camera_position(0.0, 80.0)
        # tilt_speed=1.0, dt=1.0 → +90 → clamp to 90
        ctx.update_camera_by_velocity(pan_speed=0.0, tilt_speed=1.0, dt=1.0)
        assert ctx.camera_elevation == pytest.approx(90.0, abs=0.1)

    def test_velocity_wraps_around_360(self):
        ctx = SpatialContext()
        ctx.update_camera_position(350.0, 0.0)
        # pan_speed=0.5, dt=1.0 → +45° → 395 % 360 = 35
        ctx.update_camera_by_velocity(pan_speed=0.5, tilt_speed=0.0, dt=1.0)
        assert ctx.camera_azimuth == pytest.approx(35.0, abs=0.1)


# ════════════════════════════════════════════
# 3. 월드 좌표 추정 테스트
# ════════════════════════════════════════════

class TestWorldAngleEstimation:
    """YOLO 바운딩 박스 중심의 프레임 내 x위치 → 절대 방위각 변환 검증"""

    def test_center_of_frame(self):
        """화면 정중앙 객체 → 카메라 방위각과 동일"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results(
            [make_person(1, center_x=320)],  # 640 width의 중앙
            frame_shape=(480, 640),
        )
        objs = ctx.detected_objects
        assert len(objs) == 1
        assert objs[0]["world_angle"] == pytest.approx(180.0, abs=0.5)

    def test_left_edge_of_frame(self):
        """화면 왼쪽 끝 → 카메라 방위각 - FOV/2"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results(
            [make_person(1, center_x=0)],  # 왼쪽 끝
            frame_shape=(480, 640),
        )
        objs = ctx.detected_objects
        # 180 - 30 = 150
        assert objs[0]["world_angle"] == pytest.approx(150.0, abs=0.5)

    def test_right_edge_of_frame(self):
        """화면 오른쪽 끝 → 카메라 방위각 + FOV/2"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results(
            [make_person(1, center_x=640)],  # 오른쪽 끝
            frame_shape=(480, 640),
        )
        objs = ctx.detected_objects
        # 180 + 30 = 210
        assert objs[0]["world_angle"] == pytest.approx(210.0, abs=0.5)

    def test_quarter_position(self):
        """화면 1/4 위치"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results(
            [make_person(1, center_x=160)],  # 640 * 0.25 = 160
            frame_shape=(480, 640),
        )
        objs = ctx.detected_objects
        # 180 - 30 + 0.25 * 60 = 180 - 30 + 15 = 165
        assert objs[0]["world_angle"] == pytest.approx(165.0, abs=0.5)

    def test_wraps_around_360_boundary(self):
        """카메라가 350° 보고 있을 때 오른쪽 끝 → 360° 넘어 20°"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(350.0, 0.0)
        ctx.update_yolo_results(
            [make_person(1, center_x=640)],  # 오른쪽 끝
            frame_shape=(480, 640),
        )
        objs = ctx.detected_objects
        # 350 + 30 = 380 → 380 % 360 = 20
        assert objs[0]["world_angle"] == pytest.approx(20.0, abs=0.5)

    def test_multiple_objects(self):
        """여러 객체 동시 추정"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results(
            [
                make_person(1, center_x=320),  # 중앙 → 180°
                make_car(2, center_x=0),       # 왼쪽 → 150°
            ],
            frame_shape=(480, 640),
        )
        objs = ctx.detected_objects
        assert len(objs) == 2
        assert objs[0]["world_angle"] == pytest.approx(180.0, abs=0.5)
        assert objs[1]["world_angle"] == pytest.approx(150.0, abs=0.5)

    def test_person_detected_flag(self):
        """person 객체 존재 시 person_detected 플래그"""
        ctx = SpatialContext()
        ctx.update_yolo_results([make_car(1, 320)], (480, 640))
        assert ctx.person_detected is False

        ctx.update_yolo_results([make_person(1, 320)], (480, 640))
        assert ctx.person_detected is True


# ════════════════════════════════════════════
# 4. 공간 매칭 테스트 (핵심)
# ════════════════════════════════════════════

class TestSpatialMatching:
    """DOA 방향과 YOLO 객체의 공간 매칭 검증"""

    def _setup_person_at_angle(self, ctx, target_angle, pid=1):
        """지정된 월드 각도에 사람을 배치하는 헬퍼"""
        # FOV=60, 카메라=target_angle이면 중앙 객체가 target_angle에 배치됨
        ctx.update_camera_position(target_angle, 0.0)
        ctx.update_yolo_results(
            [make_person(pid, center_x=320)],  # 중앙
            frame_shape=(480, 640),
        )

    def test_match_same_direction(self):
        """DOA와 사람이 같은 방향 → 매칭 성공"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 180.0)
        ctx.update_doa(180.0, confidence=0.9)

        assert ctx.spatial_match is True
        details = ctx.get_match_details()
        assert details["matched_object"] == "person"
        assert details["angle_diff"] == pytest.approx(0.0, abs=1.0)

    def test_match_within_threshold(self):
        """DOA와 사람이 임계값 이내 → 매칭 성공"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 180.0)
        ctx.update_doa(175.0, confidence=0.8)

        assert ctx.spatial_match is True
        details = ctx.get_match_details()
        assert details["angle_diff"] == pytest.approx(5.0, abs=1.0)

    def test_no_match_outside_threshold(self):
        """DOA와 사람이 임계값 밖 → 매칭 실패"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 180.0)
        ctx.update_doa(90.0, confidence=0.9)

        assert ctx.spatial_match is False

    def test_360_boundary_wrap(self):
        """360° 경계 처리: DOA=355°, 사람=5° → 차이=10°"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 5.0)
        ctx.update_doa(355.0, confidence=0.9)

        assert ctx.spatial_match is True
        details = ctx.get_match_details()
        assert details["angle_diff"] == pytest.approx(10.0, abs=1.0)

    def test_360_boundary_wrap_reverse(self):
        """역방향 360° 경계: DOA=5°, 사람=355°"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 355.0)
        ctx.update_doa(5.0, confidence=0.9)

        assert ctx.spatial_match is True
        details = ctx.get_match_details()
        assert details["angle_diff"] == pytest.approx(10.0, abs=1.0)

    def test_car_also_matches(self):
        """차량도 매칭 대상 (is_person 플래그로 구분)"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results([make_car(1, center_x=320)], (480, 640))
        ctx.update_doa(180.0, confidence=0.9)

        assert ctx.spatial_match is True
        details = ctx.get_match_details()
        assert details["matched_object"] == "car"
        assert details["is_person"] is False

    def test_best_match_closest_angle(self):
        """여러 객체 중 가장 가까운 각도의 객체와 매칭"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results(
            [
                make_person(1, center_x=160),   # 165°
                make_person(2, center_x=480),   # 195°
            ],
            frame_shape=(480, 640),
        )
        ctx.update_doa(170.0, confidence=0.9)

        assert ctx.spatial_match is True
        details = ctx.get_match_details()
        assert details["object_id"] == 1  # 165°에 가까운 person 1

    def test_no_doa_no_match(self):
        """DOA 없으면 매칭 불가"""
        ctx = SpatialContext()
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results([make_person(1, 320)], (480, 640))
        # DOA 미갱신 (기본 -1)
        assert ctx.spatial_match is False

    def test_no_yolo_no_match(self):
        """YOLO 결과 없으면 매칭 불가"""
        ctx = SpatialContext()
        ctx.update_doa(180.0, confidence=0.9)
        assert ctx.spatial_match is False

    def test_stale_doa_expires(self):
        """5초 이상 지난 DOA는 매칭에서 제외"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 180.0)

        # DOA를 설정하되 시간을 과거로 조작
        ctx.update_doa(180.0, confidence=0.9)
        # 시간을 6초 전으로 조작
        ctx._last_doa_time = time.time() - 6.0
        # 재계산 트리거
        ctx.update_yolo_results([make_person(1, 320)], (480, 640))

        assert ctx.spatial_match is False

    def test_custom_threshold(self):
        """좁은 임계값에서는 매칭 범위가 축소"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=5.0)
        self._setup_person_at_angle(ctx, 180.0)
        ctx.update_doa(170.0, confidence=0.9)  # 10° 차이

        assert ctx.spatial_match is False  # 5° 임계값 초과

    def test_recalc_on_yolo_update(self):
        """YOLO 결과 업데이트 시 매칭 재계산"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        ctx.update_doa(180.0, confidence=0.9)

        # 처음: 객체 없음 → 매칭 안 됨
        assert ctx.spatial_match is False

        # 사람 추가 → 자동 재계산
        self._setup_person_at_angle(ctx, 180.0)
        assert ctx.spatial_match is True

    def test_recalc_on_doa_update(self):
        """DOA 업데이트 시 매칭 재계산"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        self._setup_person_at_angle(ctx, 180.0)

        # 다른 방향 DOA → 매칭 안 됨
        ctx.update_doa(90.0, confidence=0.9)
        assert ctx.spatial_match is False

        # 같은 방향 DOA → 매칭 됨
        ctx.update_doa(180.0, confidence=0.9)
        assert ctx.spatial_match is True


# ════════════════════════════════════════════
# 5. STT 갱신 테스트
# ════════════════════════════════════════════

class TestSTTUpdate:
    def test_stt_text_stored(self):
        ctx = SpatialContext()
        ctx.update_stt("도와주세요")
        assert ctx.last_stt_text == "도와주세요"

    def test_stt_text_expires_after_30s(self):
        ctx = SpatialContext()
        ctx.update_stt("테스트")
        ctx._last_stt_time = time.time() - 31.0  # 31초 전으로 조작
        assert ctx.last_stt_text is None

    def test_stt_overwrite(self):
        ctx = SpatialContext()
        ctx.update_stt("첫번째")
        ctx.update_stt("두번째")
        assert ctx.last_stt_text == "두번째"


# ════════════════════════════════════════════
# 6. 이벤트 히스토리 테스트
# ════════════════════════════════════════════

class TestEventHistory:
    def test_add_and_retrieve(self):
        ctx = SpatialContext()
        ctx.add_event("test_event", {"key": "value"})
        events = ctx.get_recent_events(30.0)
        assert len(events) == 1
        assert events[0]["type"] == "test_event"
        assert events[0]["data"]["key"] == "value"

    def test_max_size_limit(self):
        ctx = SpatialContext(event_history_size=5)
        for i in range(10):
            ctx.add_event(f"event_{i}", {})
        # deque maxlen=5이므로 최대 5개만 유지
        events = ctx.get_recent_events(30.0)
        assert len(events) == 5
        assert events[0]["type"] == "event_5"  # 가장 오래된 것 = event_5

    def test_time_based_filtering(self):
        ctx = SpatialContext()
        # 오래된 이벤트
        ctx.add_event("old_event", {})
        ctx._event_timeline[-1]["time"] = time.time() - 60.0

        # 최근 이벤트
        ctx.add_event("new_event", {})

        events = ctx.get_recent_events(30.0)
        assert len(events) == 1
        assert events[0]["type"] == "new_event"

    def test_yolo_person_adds_event(self):
        """사람 감지 시 자동 이벤트 추가"""
        ctx = SpatialContext()
        ctx.update_yolo_results([make_person(1, 320)], (480, 640))
        events = ctx.get_recent_events(5.0)
        types = [e["type"] for e in events]
        assert "person_detected" in types

    def test_doa_adds_event(self):
        """DOA 감지 시 자동 이벤트 추가"""
        ctx = SpatialContext()
        ctx.update_doa(180.0, 0.9)
        events = ctx.get_recent_events(5.0)
        types = [e["type"] for e in events]
        assert "doa_detected" in types

    def test_stt_adds_event(self):
        """STT 인식 시 자동 이벤트 추가"""
        ctx = SpatialContext()
        ctx.update_stt("테스트")
        events = ctx.get_recent_events(5.0)
        types = [e["type"] for e in events]
        assert "stt_recognized" in types


# ════════════════════════════════════════════
# 7. LLM 컨텍스트 빌드 테스트
# ════════════════════════════════════════════

class TestLLMContextBuild:
    def test_empty_context_has_camera(self):
        """기본 상태에서도 카메라 정보 포함"""
        ctx = SpatialContext()
        text = ctx.build_llm_context()
        assert "[카메라]" in text
        assert "방위각: 0°" in text

    def test_yolo_objects_included(self):
        """YOLO 결과가 컨텍스트에 포함"""
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results([make_person(1, 320)], (480, 640))
        text = ctx.build_llm_context()
        assert "[시각 감지]" in text
        assert "1개 객체" in text
        assert "사람 감지: 예" in text

    def test_doa_included(self):
        """DOA 정보가 컨텍스트에 포함"""
        ctx = SpatialContext()
        ctx.update_doa(175.0, 0.9)
        text = ctx.build_llm_context()
        assert "[음원 방향]" in text
        assert "175°" in text

    def test_stt_included(self):
        """STT 텍스트가 컨텍스트에 포함"""
        ctx = SpatialContext()
        ctx.update_stt("도와주세요")
        text = ctx.build_llm_context()
        assert "[음성 인식]" in text
        assert "도와주세요" in text

    def test_spatial_match_warning(self):
        """매칭 시 경고 포함"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results([make_person(1, 320)], (480, 640))
        ctx.update_doa(180.0, 0.9)
        text = ctx.build_llm_context()
        assert "[공간 매칭]" in text
        assert "⚠️" in text

    def test_no_match_message(self):
        """매칭 안 될 때 메시지"""
        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        ctx.update_camera_position(180.0, 0.0)
        ctx.update_yolo_results([make_person(1, 320)], (480, 640))
        ctx.update_doa(90.0, 0.9)  # 다른 방향
        text = ctx.build_llm_context()
        assert "일치하지 않음" in text

    def test_no_objects_message(self):
        """객체 없을 때 메시지"""
        ctx = SpatialContext()
        text = ctx.build_llm_context()
        assert "감지된 객체 없음" in text


# ════════════════════════════════════════════
# 8. 스냅샷 테스트
# ════════════════════════════════════════════

class TestSnapshot:
    def test_snapshot_structure(self):
        ctx = SpatialContext()
        snap = ctx.get_snapshot()
        assert "camera" in snap
        assert "yolo" in snap
        assert "audio" in snap
        assert "fusion" in snap
        assert "event_count" in snap

    def test_snapshot_reflects_state(self):
        ctx = SpatialContext(camera_fov=60.0)
        ctx.update_camera_position(123.0, -10.0)
        ctx.update_doa(150.0, 0.8)
        snap = ctx.get_snapshot()
        assert snap["camera"]["azimuth"] == 123.0
        assert snap["camera"]["elevation"] == -10.0
        assert snap["audio"]["doa_angle"] == 150.0
        assert snap["audio"]["doa_confidence"] == 0.8


# ════════════════════════════════════════════
# 9. 스레드 안전성 기본 테스트
# ════════════════════════════════════════════

class TestThreadSafety:
    def test_concurrent_updates(self):
        """다수의 동시 업데이트에서 에러 없이 동작"""
        import threading

        ctx = SpatialContext(camera_fov=60.0, spatial_match_threshold=30.0)
        errors = []

        def update_yolo():
            try:
                for i in range(100):
                    ctx.update_yolo_results(
                        [make_person(1, center_x=320)], (480, 640)
                    )
            except Exception as e:
                errors.append(e)

        def update_doa():
            try:
                for i in range(100):
                    ctx.update_doa(float(i * 3.6), 0.8)
            except Exception as e:
                errors.append(e)

        def read_context():
            try:
                for i in range(100):
                    ctx.build_llm_context()
                    ctx.check_spatial_match()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_yolo),
            threading.Thread(target=update_doa),
            threading.Thread(target=read_context),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
