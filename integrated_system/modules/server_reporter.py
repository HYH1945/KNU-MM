import time
import logging
import requests
import json
import datetime
from typing import Dict, Any

from integrated_system.core.base_module import BaseModule
from integrated_system.core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)

class ServerReporterModule(BaseModule):
    """
    서버 전송 모듈
    이벤트 빈도 조절(Throttling) 기능이 포함된 통합 버전
    """

    def __init__(
        self,
        event_bus: EventBus,
        server_url: str = "",
        timeout: float = 2.0,
        enabled: bool = True,
    ):
        super().__init__(event_bus)
        self.server_url = server_url
        self.timeout = timeout
        self._enabled = enabled
        self._send_count = 0
        self._fail_count = 0

        # --- 전송 빈도 조절(Throttling) 설정 ---
        self.last_sent_time = {
            "PERSON_DETECTED": 0,
            "CAMERA_MOVE": 0,
            "AUDIO_DETECTION": 0,
            "ANALYSIS": 0,
            "EMERGENCY": 0
        }
        # 초 단위 쿨다운 설정 (필요에 따라 조절)
        self.cooldowns = {
            "PERSON_DETECTED": 3.0,  # 사람 감지는 3초마다 1번만
            "CAMERA_MOVE": 0.5,      # PTZ 이동 로그는 0.5초마다
            "AUDIO_DETECTION": 0.2,   # 소리 방향은 0.2초마다
            "ANALYSIS": 2.0,         # LLM 일반 분석은 2초마다
            "EMERGENCY": 0.0         # 긴급 상황은 즉시 전송
        }

    @property
    def name(self) -> str:
        """BaseModule 필수 구현: 모듈 이름"""
        return "server_reporter"

    def initialize(self) -> bool:
        """BaseModule 필수 구현: 초기화 및 이벤트 구독"""
        if not self.server_url:
            logger.warning("[ServerReporter] 서버 URL이 설정되지 않았습니다.")
            return False

        # 이벤트 버스 구독 등록
        self._event_bus.subscribe("llm.emergency", self._on_emergency)
        self._event_bus.subscribe("llm.analysis_complete", self._on_analysis_complete)
        self._event_bus.subscribe("yolo.person_detected", self._on_person_detected)
        self._event_bus.subscribe("mic.doa_detected", self._on_doa_detected)

        logger.info(f"[ServerReporter] 초기화 완료 -> {self.server_url}")
        return True

    def _should_send(self, event_type: str) -> bool:
        """전송 빈도(쿨다운)를 체크하는 내부 함수"""
        now = time.time()
        last_time = self.last_sent_time.get(event_type, 0)
        cooldown = self.cooldowns.get(event_type, 1.0)

        if now - last_time >= cooldown:
            self.last_sent_time[event_type] = now
            return True
        return False

    # --- 이벤트 핸들러들 ---

    def _on_emergency(self, event: Event) -> None:
        if self._should_send("EMERGENCY"):
            self._send_to_server({
                "source": "ContextLLM",
                "type": "EMERGENCY",
                "data": {
                    "urgency": event.data.get("urgency", "N/A"),
                    "situation": event.data.get("situation", "N/A"),
                    "angle": "-" # 로그 표시용
                }
            })

    def _on_analysis_complete(self, event: Event) -> None:
        if self._should_send("ANALYSIS"):
            result = event.data.get("result", {})
            self._send_to_server({
                "source": "ContextLLM",
                "type": "ANALYSIS",
                "data": {
                    "priority": event.data.get("priority", "LOW"),
                    "situation_type": result.get("situation_type", "N/A"),
                    "angle": "-" 
                }
            })

    def _on_person_detected(self, event: Event) -> None:
        if self._should_send("PERSON_DETECTED"):
            self._send_to_server({
                "source": "YOLO",
                "type": "PERSON_DETECTED",
                "data": {
                    "count": event.data.get("count", 0),
                    "angle": "-" 
                }
            })

    def _on_doa_detected(self, event: Event) -> None:
        if self._should_send("AUDIO_DETECTION"):
            self._send_to_server({
                "source": "MicArray",
                "type": "AUDIO_DETECTION",
                "data": {
                    "angle": event.data.get("sector_angle", 0)
                }
            })

    def _send_to_server(self, payload: dict) -> bool:
        """실제 HTTP 전송 수행"""
        if not self._enabled or not self.server_url:
            return False

        try:
            # timestamp 자동 추가
            payload["timestamp"] = time.time()
            
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                self._send_count += 1
                return True
            else:
                self._fail_count += 1
                return False
        except Exception as e:
            self._fail_count += 1
            return False

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """BaseModule 필수 구현: 파이프라인 처리 시 상태 반환"""
        return {
            "send_count": self._send_count,
            "fail_count": self._fail_count
        }

    def shutdown(self) -> None:
        """BaseModule 필수 구현: 종료 시 처리"""
        logger.info(f"[ServerReporter] 종료 (성공: {self._send_count}, 실패: {self._fail_count})")