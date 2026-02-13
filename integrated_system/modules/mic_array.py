"""
마이크 어레이 모듈 - ReSpeaker DOA(음원 방향) 감지

원본 모듈 (직접 import — 수정 시 즉시 반영):
    - mic_array_Control/tuning.py → Tuning 클래스 (USB 디바이스 제어)

통합 레이어 (이 파일에만 존재):
    - DOA 각도 처리 (원형 평균 + 신뢰도)  ← mic_array_Control/test.py 참조
    - 섹터 양자화 (30도 단위)              ← mic_array_Control/test.py → get_sector_angle()
    - 천정 사각지대 감지                   ← mic_array_Control/test.py → mic_monitoring_thread()
    - EventBus 이벤트 발행
    - PTZ 카메라 연동

※ mic_array_Control/test.py는 전역 USB 디바이스 초기화 부작용으로 직접 import 불가.
   알고리즘만 참조하여 동일 로직을 유지합니다. test.py의 알고리즘을 수정하면
   아래 해당 메서드도 동기화해주세요:
   - get_sector_angle()     → _get_sector()
   - mic_monitoring_thread() → _monitor_loop() + _process_doa()
   - control_ptz_absolute() → (UnifiedPTZController._absolute_move로 대체됨)
"""

import time
import threading
import collections
import logging
from typing import Dict, Any, Optional

import numpy as np

from integrated_system.core.base_module import BaseModule
from integrated_system.core.event_bus import EventBus, Event
from integrated_system.modules.ptz_controller import UnifiedPTZController, PTZPriority

logger = logging.getLogger(__name__)


class MicArrayModule(BaseModule):
    """
    ReSpeaker v2 마이크 어레이 기반 DOA 감지 모듈

    ★ Tuning 클래스는 원본 tuning.py에서 직접 import합니다.
    tuning.py를 수정하면 즉시 반영됩니다.

    DOA 알고리즘은 test.py의 mic_monitoring_thread()를 참조하되,
    EventBus 연동 + 백그라운드 스레드 구조를 추가한 버전입니다.
    """

    def __init__(
        self,
        event_bus: EventBus,
        ptz: Optional[UnifiedPTZController] = None,
        agc_max_gain: float = 15.0,
        vad_threshold: float = 10.0,
        confidence_threshold: float = 0.6,
        zenith_confidence: float = 0.4,
        zenith_gain: float = 10.0,
        history_size: int = 10,
    ):
        super().__init__(event_bus)
        self.ptz = ptz
        self.agc_max_gain = agc_max_gain
        self.vad_threshold = vad_threshold
        self.confidence_threshold = confidence_threshold
        self.zenith_confidence = zenith_confidence
        self.zenith_gain = zenith_gain
        self.history_size = history_size

        # 런타임 상태
        self._mic_tuning = None  # ★ 원본 Tuning 인스턴스
        self._angle_history = collections.deque(maxlen=history_size)
        self._last_sector = -1
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def name(self) -> str:
        return "mic_array"

    def initialize(self) -> bool:
        """ReSpeaker USB 디바이스 연결"""
        try:
            import usb.core
            import usb.backend.libusb1

            # libusb 백엔드 확인
            backend = usb.backend.libusb1.get_backend()
            if backend is None:
                logger.warning(
                    "[MicArray] libusb 백엔드를 찾을 수 없습니다.\n"
                    "         macOS: brew install libusb\n"
                    "         Ubuntu: sudo apt install libusb-1.0-0-dev\n"
                    "         → 비활성화 상태로 계속 실행합니다."
                )
                return False

            # ★ 원본 Tuning 클래스 import ★
            import sys
            from pathlib import Path
            from integrated_system.core.module_loader import MIC_DIR, ensure_path
            ensure_path(MIC_DIR)
            from tuning import Tuning

            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018, backend=backend)
            if dev is None:
                logger.warning("[MicArray] ReSpeaker 디바이스를 찾을 수 없습니다 (USB 미연결, 비활성화)")
                return False

            self._mic_tuning = Tuning(dev)

            # 민감도 설정 — 원본 test.py → initialize_system() 참조
            self._mic_tuning.write('AGCMAXGAIN', self.agc_max_gain)
            self._mic_tuning.write('GAMMAVAD_SR', self.vad_threshold)

            logger.info("[MicArray] ReSpeaker 초기화 완료 (원본 Tuning 사용)")
            return True

        except ImportError as e:
            logger.warning(
                f"[MicArray] pyusb 패키지 미설치 (비활성화): {e}\n"
                "         설치: pip install pyusb"
            )
            return False
        except Exception as e:
            logger.error(f"[MicArray] 초기화 실패: {e}")
            return False

    def start_monitoring(self) -> None:
        """백그라운드 DOA 모니터링 시작"""
        if not self._initialized or self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[MicArray] 모니터링 시작")

    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

    def _monitor_loop(self) -> None:
        """
        DOA 모니터링 루프 (백그라운드 스레드)
        알고리즘 출처: mic_array_Control/test.py → mic_monitoring_thread()
        """
        while self._running:
            try:
                if self._mic_tuning.read('SPEECHDETECTED') == 1:
                    raw_angle = self._mic_tuning.read('DOAANGLE')
                    gain = self._mic_tuning.read('AGCGAIN')
                    self._angle_history.append(raw_angle)

                    # 음성 감지 이벤트
                    self.emit("mic.speech_detected", {
                        "raw_angle": raw_angle,
                        "gain": gain,
                    })

                    if len(self._angle_history) >= 5:
                        self._process_doa(gain)

                time.sleep(0.05)

            except Exception as e:
                logger.error(f"[MicArray] 모니터링 오류: {e}")
                time.sleep(0.5)

    def _process_doa(self, gain: float) -> None:
        """
        DOA 각도 처리 (원형 평균 + 신뢰도)
        알고리즘 출처: mic_array_Control/test.py → mic_monitoring_thread() 내부 로직
        """
        rad_angles = np.deg2rad(list(self._angle_history))
        sin_mean = np.mean(np.sin(rad_angles))
        cos_mean = np.mean(np.cos(rad_angles))
        confidence = np.sqrt(sin_mean ** 2 + cos_mean ** 2)

        # 1. 천정 사각지대 — test.py: confidence < 0.4 and gain < 10.0
        if confidence < self.zenith_confidence and gain < self.zenith_gain:
            self.emit("mic.zenith_detected", {"confidence": confidence})
            if self.ptz:
                self.ptz.request_move(0, -90, PTZPriority.MIC_DOA, self.name, move_type="absolute")
            return

        # 2. 유효한 방향 — test.py: confidence > 0.6
        if confidence > self.confidence_threshold:
            smooth_angle = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
            sector = self._get_sector(smooth_angle)

            if sector != self._last_sector:
                self._last_sector = sector

                self.emit("mic.doa_detected", {
                    "sector_angle": sector,
                    "smooth_angle": float(smooth_angle),
                    "confidence": float(confidence),
                }, priority=1)

                if self.ptz:
                    self.ptz.request_move(
                        sector, -15,
                        PTZPriority.MIC_DOA,
                        self.name,
                        move_type="absolute",
                    )

    @staticmethod
    def _get_sector(raw_angle: float) -> int:
        """
        360도를 12섹터(30도 단위)로 양자화
        알고리즘 출처: mic_array_Control/test.py → get_sector_angle()
        """
        return int(((raw_angle + 15) // 30) * 30) % 360

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        오케스트레이터 파이프라인에서 호출 시 최근 상태 반환.
        MicArray는 자체 백그라운드 스레드에서 동작합니다.
        """
        return {
            "last_sector": self._last_sector,
            "monitoring": self._running,
            "history_length": len(self._angle_history),
        }

    def shutdown(self) -> None:
        self.stop_monitoring()
        if self._mic_tuning:
            try:
                self._mic_tuning.close()
            except Exception:
                pass
        logger.info("[MicArray] 종료")
