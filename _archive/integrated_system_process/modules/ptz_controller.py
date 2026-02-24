"""
통합 PTZ 컨트롤러 - 우선순위 기반 PTZ 제어 중재

원본 모듈 (직접 import — 수정 시 즉시 반영):
    - Detaction_CCTV/services/ptz_controller.py → PTZCameraManager (ONVIF ContinuousMove)

통합 레이어 (이 파일에만 존재):
    - PTZPriority 우선순위 시스템
    - Hikvision HTTP AbsoluteMove 어댑터 (mic_array_Control/test.py의 control_ptz_absolute 참조)
    - 우선순위 기반 중재 (여러 모듈의 PTZ 요청 관리)
"""

import threading
import time
import logging
from typing import Optional, Dict, Any
from types import SimpleNamespace
from enum import IntEnum

from integrated_system_process.core.module_loader import DETECT_DIR, PTZ_DIR, import_from_file

logger = logging.getLogger(__name__)


class PTZPriority(IntEnum):
    """PTZ 제어 우선순위 (높을수록 우선)"""
    PATROL = 0           # 순찰 모드
    MIC_DOA = 1          # 마이크 DOA 방향 이동
    YOLO_TRACKING = 2    # YOLO 객체 추적
    EMERGENCY = 3        # 긴급 상황


class UnifiedPTZController:
    """
    통합 PTZ 컨트롤러

    ONVIF ContinuousMove: ★ 원본 PTZCameraManager에 위임 ★
    Hikvision HTTP Absolute: mic_array_Control/test.py의 control_ptz_absolute() 참조

    여러 모듈이 동시에 PTZ를 제어하려 할 때 우선순위로 중재합니다.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: PTZ 설정 dict
                - camera_ip, camera_port, camera_user, camera_password
                - control_mode: "onvif" | "hikvision_http" | "both"
        """
        self.config = config
        self.control_mode = str(config.get("control_mode", "onvif")).strip().lower()
        self._lock = threading.Lock()
        self._current_priority = PTZPriority.PATROL
        self._current_owner = ""
        self._last_move_time = 0.0

        # ★ 원본 PTZCameraManager 인스턴스 (ONVIF용) ★
        self._onvif_mgr = None
        self._hikvision_auth = None
        self._connected = False
        self._last_abs_pan = 0.0
        self._last_abs_tilt = 0.0

    def initialize(self) -> bool:
        """PTZ 연결 초기화"""
        control_mode = self.control_mode

        if control_mode in ("onvif", "both"):
            self._connected = self._init_onvif()

        if control_mode in ("hikvision_http", "both"):
            self._init_hikvision_http()

        return self._connected
    
    def _init_onvif(self) -> bool:
        """
        ONVIF PTZ 초기화
        우선순위:
        1) PTZcamera_Control/ptz_controller.py (사용자 검증 경로)
        2) Detection_CCTV/services/ptz_controller.py (레거시 폴백)
        """
        # 1) PTZcamera_Control 경로 우선
        if self._init_onvif_from_ptzcamera_control():
            return True

        # 2) 레거시 Detection_CCTV 경로 폴백
        return self._init_onvif_from_detection_cctv()

    def _init_onvif_from_ptzcamera_control(self) -> bool:
        """PTZcamera_Control 기반 ONVIF 초기화."""
        try:
            import os
            _ptz_mod = import_from_file(
                "_orig_ptzcamera_control_controller",
                os.path.join(PTZ_DIR, "ptz_controller.py"),
            )
            PTZCameraController = _ptz_mod.PTZCameraController
            self._onvif_mgr = PTZCameraController(
                self.config.get("camera_ip", ""),
                int(self.config.get("camera_port", 80)),
                self.config.get("camera_user", ""),
                self.config.get("camera_password", ""),
            )
            self._connected = bool(getattr(self._onvif_mgr, "is_connected", False))
            if self._connected:
                logger.info("[PTZ] ONVIF 연결 성공 (PTZcamera_Control)")
            return self._connected
        except Exception as e:
            logger.warning(f"[PTZ] PTZcamera_Control ONVIF 초기화 실패, 레거시 경로로 폴백: {e}")
            return False

    def _init_onvif_from_detection_cctv(self) -> bool:
        """Detection_CCTV/services/ptz_controller.py 기반 ONVIF 초기화(하위 호환)."""
        try:
            import os
            import sys
            from types import ModuleType

            if 'config' not in sys.modules:
                mock_mod = ModuleType('config')

                class AppConfig:
                    CAMERA_IP = self.config.get("camera_ip", "")
                    CAMERA_PORT = self.config.get("camera_port", 80)
                    CAMERA_USER = self.config.get("camera_user", "")
                    CAMERA_PASSWORD = self.config.get("camera_password", "")

                mock_mod.AppConfig = AppConfig
                sys.modules['config'] = mock_mod

            _ptz_mod = import_from_file("_orig_ptz_controller", os.path.join(DETECT_DIR, "services", "ptz_controller.py"))
            PTZCameraManager = _ptz_mod.PTZCameraManager

            from config import AppConfig
            config_obj = AppConfig()
            self._onvif_mgr = PTZCameraManager(config_obj)
            self._connected = bool(getattr(self._onvif_mgr, '_connected', False))
            if self._connected:
                logger.info("[PTZ] ONVIF 연결 성공 (Detection_CCTV fallback)")
            return self._connected
        except Exception as e:
            logger.error(f"[PTZ] ONVIF 연결 실패: {e}")
            return False

    def _init_hikvision_http(self):
        """
        Hikvision HTTP API 초기화
        인증 정보만 설정 (실제 이동은 _absolute_move에서 처리)
        """
        try:
            from requests.auth import HTTPDigestAuth
            self._hikvision_auth = HTTPDigestAuth(
                self.config.get("camera_user", ""),
                self.config.get("camera_password", ""),
            )
            logger.info("[PTZ] Hikvision HTTP 인증 설정 완료")
        except Exception as e:
            logger.error(f"[PTZ] Hikvision HTTP 설정 실패: {e}")

    # ─── 우선순위 기반 이동 요청 (통합 레이어) ───

    def request_move(
        self,
        pan: float,
        tilt: float,
        priority: PTZPriority,
        owner: str,
        move_type: str = "continuous",
        zoom: float = 0.0,
    ) -> bool:
        """
        우선순위 기반 PTZ 이동 요청

        Args:
            pan: 수평 이동 (continuous: -1~1 속도, absolute: 0~360 각도)
            tilt: 수직 이동 (continuous: -1~1 속도, absolute: -90~90 각도)
            priority: 요청 우선순위
            owner: 요청자 (모듈 이름)
            move_type: "continuous" | "absolute"
            zoom: 줌 값
        """
        with self._lock:
            if priority < self._current_priority:
                if time.time() - self._last_move_time < 2.0:
                    logger.debug(f"[PTZ] 요청 거절: {owner}({priority.name}) < {self._current_owner}({self._current_priority.name})")
                    return False

            self._current_priority = priority
            self._current_owner = owner
            self._last_move_time = time.time()

        if move_type == "absolute":
            if self._hikvision_auth:
                threading.Thread(target=self._absolute_move, args=(pan, tilt, zoom), daemon=True).start()
            elif self._onvif_mgr:
                # onvif 전용 환경에서는 절대이동 API가 없어, 연속이동 펄스로 근사 이동
                threading.Thread(target=self._absolute_move_via_onvif, args=(pan, tilt, zoom), daemon=True).start()
            else:
                logger.warning("[PTZ] AbsoluteMove 요청 무시: 사용 가능한 PTZ 백엔드가 없습니다.")
        else:
            threading.Thread(target=self._continuous_move, args=(pan, tilt, zoom), daemon=True).start()

        return True

    def _continuous_move(self, pan: float, tilt: float, zoom: float):
        """
        ONVIF ContinuousMove
        ★ 원본 컨트롤러 메서드에 위임 ★
        """
        self._onvif_start_move(pan, tilt, zoom)

    def _absolute_move(self, pan: float, tilt: float, zoom: float):
        """
        Hikvision HTTP AbsoluteMove

        알고리즘 출처: mic_array_Control/test.py → control_ptz_absolute()
        ※ test.py는 전역 변수 부작용으로 직접 import 불가하여 동일 로직 유지
        ※ test.py의 XML 구조 및 API URL을 수정하면 이쪽도 동기화 필요
        """
        if not self._hikvision_auth:
            return
        try:
            import requests
            url = f"http://{self.config.get('camera_ip')}/ISAPI/PTZCtrl/channels/1/absolute"

            azimuth = int(pan * 10) if pan is not None else 0
            elevation = int(tilt * 10)
            absolute_zoom = int(zoom * 10) if zoom else 10

            xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
            <PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
                <AbsoluteHigh>
                    <elevation>{elevation}</elevation>
                    <azimuth>{azimuth}</azimuth>
                    <absoluteZoom>{absolute_zoom}</absoluteZoom>
                </AbsoluteHigh>
            </PTZData>"""

            requests.put(url, data=xml_data, auth=self._hikvision_auth, timeout=1)
        except Exception as e:
            logger.error(f"[PTZ] AbsoluteMove 오류: {e}")

    def _absolute_move_via_onvif(self, pan: float, tilt: float, zoom: float):
        """
        onvif-only 환경용 pseudo absolute move.
        절대 좌표 API 부재를 보완하기 위해, 목표 각도까지 연속이동을 짧게 펄스 제어한다.
        """
        if not self._onvif_mgr:
            return

        try:
            target_pan = float(pan) % 360.0 if pan is not None else self._last_abs_pan
            target_tilt = float(tilt) if tilt is not None else self._last_abs_tilt
            target_tilt = max(-90.0, min(90.0, target_tilt))

            pan_delta = ((target_pan - self._last_abs_pan + 540.0) % 360.0) - 180.0
            tilt_delta = target_tilt - self._last_abs_tilt

            pan_vel = 0.0 if abs(pan_delta) < 2.0 else max(-1.0, min(1.0, pan_delta / 90.0))
            tilt_vel = 0.0 if abs(tilt_delta) < 2.0 else max(-1.0, min(1.0, tilt_delta / 45.0))
            if pan_vel == 0.0 and tilt_vel == 0.0:
                return

            pan_time = abs(pan_delta) / 90.0
            tilt_time = abs(tilt_delta) / 45.0
            duration = max(pan_time, tilt_time)
            duration = max(0.1, min(2.0, duration))

            self._onvif_start_move(pan_vel, tilt_vel, zoom)
            time.sleep(duration)
            self._onvif_stop()

            self._last_abs_pan = target_pan
            self._last_abs_tilt = target_tilt
            logger.info(
                "[PTZ] ONVIF pseudo-absolute move: pan=%.1f tilt=%.1f (duration=%.2fs)",
                target_pan,
                target_tilt,
                duration,
            )
        except Exception as e:
            logger.error(f"[PTZ] ONVIF pseudo-absolute move 오류: {e}")

    def _onvif_start_move(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        if not self._onvif_mgr:
            return
        try:
            if hasattr(self._onvif_mgr, "move_async"):
                self._onvif_mgr.move_async(pan, tilt, zoom)
            elif hasattr(self._onvif_mgr, "start_continuous_move"):
                self._onvif_mgr.start_continuous_move(pan, tilt)
            else:
                logger.warning("[PTZ] ONVIF 컨트롤러에 연속 이동 메서드가 없습니다.")
        except Exception as e:
            logger.error(f"[PTZ] ONVIF 연속 이동 오류: {e}")

    def _onvif_stop(self) -> None:
        if not self._onvif_mgr:
            return
        try:
            if hasattr(self._onvif_mgr, "stop"):
                self._onvif_mgr.stop()
            elif hasattr(self._onvif_mgr, "stop_move"):
                self._onvif_mgr.stop_move()
        except Exception as e:
            logger.error(f"[PTZ] ONVIF 정지 오류: {e}")

    def stop(self) -> None:
        """PTZ 정지"""
        with self._lock:
            self._current_priority = PTZPriority.PATROL
            self._current_owner = ""

        self._onvif_stop()

    def release_control(self, owner: str) -> None:
        """PTZ 제어권 반환"""
        with self._lock:
            if self._current_owner == owner:
                self._current_priority = PTZPriority.PATROL
                self._current_owner = ""

    @property
    def current_owner(self) -> str:
        return self._current_owner

    @property
    def current_priority(self) -> PTZPriority:
        return self._current_priority

    def shutdown(self) -> None:
        """종료"""
        self.stop()
        logger.info("[PTZ] 종료 완료")
