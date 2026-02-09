"""
서버 리포터 모듈 - 분석 결과를 외부 대시보드 서버에 전송

원본 모듈 참조:
    - 서버전송예시.py → send_angle() 함수의 HTTP POST 패턴

    ※ 서버전송예시.py는 모듈 하단에서 send_angle()을 호출하므로
      직접 import 시 부작용이 발생합니다.
      동일한 HTTP POST 패턴을 유지하되, EventBus 기반으로 확장합니다.
    ※ 서버전송예시.py의 SERVER_URL 또는 payload 구조를 변경하면
      아래 _send_to_server / _on_doa_detected 도 동기화해주세요.

통합 레이어 (이 파일에만 존재):
    - 4종 이벤트 구독 (emergency, analysis, person, doa)
    - 이벤트별 payload 구성
    - 전송 통계 (성공/실패 카운트)

이벤트 구독:
    - llm.emergency          : 긴급 상황 → 서버 전송
    - llm.analysis_complete  : 분석 완료 → 서버 전송
    - yolo.person_detected   : 사람 감지 → 서버 전송
    - mic.doa_detected       : DOA 감지 → 카메라 각도 전송 (서버전송예시.py와 동일 구조)
"""

import time
import logging
from typing import Dict, Any, Optional

import requests

from integrated_system.core.base_module import BaseModule
from integrated_system.core.event_bus import EventBus, Event

logger = logging.getLogger(__name__)


class ServerReporterModule(BaseModule):
    """
    서버 전송 모듈

    서버전송예시.py의 send_angle() 패턴을 기반으로,
    EventBus 이벤트를 구독하여 자동 전송합니다.
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

    @property
    def name(self) -> str:
        return "server_reporter"

    def initialize(self) -> bool:
        """이벤트 구독"""
        if not self.server_url:
            logger.warning("[ServerReporter] 서버 URL이 설정되지 않았습니다 (비활성화)")
            return False

        self._event_bus.subscribe("llm.emergency", self._on_emergency)
        self._event_bus.subscribe("llm.analysis_complete", self._on_analysis_complete)
        self._event_bus.subscribe("yolo.person_detected", self._on_person_detected)
        self._event_bus.subscribe("mic.doa_detected", self._on_doa_detected)

        logger.info(f"[ServerReporter] 초기화 완료 → {self.server_url}")
        return True

    def _on_emergency(self, event: Event) -> None:
        """긴급 상황 서버 전송"""
        self._send_to_server({
            "source": "ContextLLM",
            "type": "EMERGENCY",
            "data": {
                "urgency": event.data.get("urgency", "N/A"),
                "situation": event.data.get("situation", "N/A"),
                "timestamp": event.timestamp,
            }
        })

    def _on_analysis_complete(self, event: Event) -> None:
        """분석 결과 서버 전송"""
        result = event.data.get("result", {})
        self._send_to_server({
            "source": "ContextLLM",
            "type": "ANALYSIS",
            "data": {
                "priority": event.data.get("priority", "LOW"),
                "is_emergency": event.data.get("is_emergency", False),
                "situation_type": result.get("situation_type", "N/A"),
                "urgency": result.get("urgency", "N/A"),
                "timestamp": event.timestamp,
            }
        })

    def _on_person_detected(self, event: Event) -> None:
        """사람 감지 서버 전송"""
        self._send_to_server({
            "source": "YOLO",
            "type": "PERSON_DETECTED",
            "data": {
                "count": event.data.get("count", 0),
                "timestamp": event.timestamp,
            }
        })

    def _on_doa_detected(self, event: Event) -> None:
        """
        DOA 각도 → 카메라 이동 서버 전송
        payload 구조: 서버전송예시.py → send_angle() 참조
        """
        self._send_to_server({
            "source": "MicArray",
            "type": "CAMERA_MOVE",  # 서버전송예시.py 와 동일 타입
            "data": {
                "angle": event.data.get("sector_angle", 0),
                "confidence": event.data.get("confidence", 0),
                "timestamp": event.timestamp,
            }
        })

    def _send_to_server(self, payload: dict) -> bool:
        """
        HTTP POST로 서버에 전송
        패턴 출처: 서버전송예시.py → send_angle()
        """
        if not self._enabled or not self.server_url:
            return False

        try:
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                self._send_count += 1
                logger.debug(f"[ServerReporter] 전송 성공: {payload.get('type')}")
                return True
            else:
                self._fail_count += 1
                logger.warning(f"[ServerReporter] 전송 실패 ({response.status_code}): {payload.get('type')}")
                return False
        except requests.exceptions.Timeout:
            self._fail_count += 1
            logger.debug(f"[ServerReporter] 전송 타임아웃: {payload.get('type')}")
            return False
        except Exception as e:
            self._fail_count += 1
            logger.debug(f"[ServerReporter] 전송 오류: {e}")
            return False

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인에서 호출 시 최근 전송 상태 반환"""
        return {
            "send_count": self._send_count,
            "fail_count": self._fail_count,
            "server_url": self.server_url,
        }

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "send_count": self._send_count,
            "fail_count": self._fail_count,
        })
        return status

    def shutdown(self) -> None:
        logger.info(f"[ServerReporter] 종료 (전송: {self._send_count}, 실패: {self._fail_count})")
