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

from integrated_system.core.module_loader import DETECT_DIR, import_from_file

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
        self._lock = threading.Lock()
        self._current_priority = PTZPriority.PATROL
        self._current_owner = ""
        self._last_move_time = 0.0

        # ★ 원본 PTZCameraManager 인스턴스 (ONVIF용) ★
        self._onvif_mgr = None
        self._hikvision_auth = None
        self._connected = False

    def initialize(self) -> bool:
        """PTZ 연결 초기화"""
        control_mode = self.config.get("control_mode", "onvif")

        if control_mode in ("onvif", "both"):
            self._connected = self._init_onvif()

        if control_mode in ("hikvision_http", "both"):
            self._init_hikvision_http()

        return self._connected

    def _init_onvif(self) -> bool:
        """
        ONVIF PTZ 초기화
        ★ 원본 PTZCameraManager에 위임 ★
        원본 수정 시 즉시 반영됩니다.
        """
        try:
            import os
            _ptz_mod = import_from_file("_orig_ptz_controller", os.path.join(DETECT_DIR, "services", "ptz_controller.py"))
            PTZCameraManager = _ptz_mod.PTZCameraManager

            # PTZCameraManager는 AppConfig 객체를 파라미터로 받으므로 호환 객체 생성
            config_obj = SimpleNamespace(
                CAMERA_IP=self.config.get("camera_ip", ""),
                CAMERA_PORT=self.config.get("camera_port", 80),
                CAMERA_USER=self.config.get("camera_user", ""),
                CAMERA_PASSWORD=self.config.get("camera_password", ""),
            )

            self._onvif_mgr = PTZCameraManager(config_obj)
            self._connected = self._onvif_mgr._connected

            if self._connected:
                logger.info("[PTZ] ONVIF 연결 성공 (원본 PTZCameraManager 사용)")
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
            threading.Thread(target=self._absolute_move, args=(pan, tilt, zoom), daemon=True).start()
        else:
            threading.Thread(target=self._continuous_move, args=(pan, tilt, zoom), daemon=True).start()

        return True

    def _continuous_move(self, pan: float, tilt: float, zoom: float):
        """
        ONVIF ContinuousMove
        ★ 원본 PTZCameraManager.move_async() 에 위임 ★
        """
        if self._onvif_mgr:
            self._onvif_mgr.move_async(pan, tilt, zoom)

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

    def stop(self) -> None:
        """PTZ 정지"""
        with self._lock:
            self._current_priority = PTZPriority.PATROL
            self._current_owner = ""

        # ★ 원본 PTZCameraManager.stop() 에 위임 ★
        if self._onvif_mgr:
            self._onvif_mgr.stop()

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
