"""
기본 모듈 인터페이스 - 모든 분석/제어 모듈이 구현해야 하는 추상 클래스

각 모듈(YOLO, ContextLLM, MicArray 등)은 이 인터페이스를 구현하여
오케스트레이터에 등록됩니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time

from .event_bus import EventBus, Event


class BaseModule(ABC):
    """
    모든 분석/제어 모듈의 기본 클래스
    
    구현 필수:
        - name (property)    : 모듈 이름
        - initialize()       : 초기화
        - process()          : 메인 처리 (프레임 단위)
        - shutdown()         : 종료
    
    선택 구현:
        - get_status()       : 모듈 상태 조회
        - on_event()         : 이벤트 수신 핸들러
    """

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._initialized = False
        self._enabled = True
        self._last_result: Dict[str, Any] = {}
        self._process_count = 0
        self._error_count = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    # ─── 추상 메서드 (반드시 구현) ───

    @property
    @abstractmethod
    def name(self) -> str:
        """모듈 이름 (예: 'yolo', 'context_llm', 'mic_array')"""
        ...

    @abstractmethod
    def initialize(self) -> bool:
        """
        모듈 초기화
        
        Returns:
            성공 여부 (False 반환 시 해당 모듈은 비활성화)
        """
        ...

    @abstractmethod
    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        메인 처리 로직 (프레임 단위로 호출)
        
        Args:
            shared_data: 공유 데이터 (프레임, 오디오, 이전 모듈 결과 등)
                - "frame"      : 현재 프레임 (np.ndarray)
                - "timestamp"  : 현재 타임스탬프
                - "results"    : 이전 모듈들의 결과 dict
                
        Returns:
            처리 결과 dict
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """안전한 종료"""
        ...

    # ─── 선택 메서드 (오버라이드 가능) ───

    def on_event(self, event: Event) -> None:
        """
        이벤트 수신 핸들러 (오버라이드하여 사용)
        
        initialize() 내에서 self._event_bus.subscribe()로 원하는 이벤트를 구독하고,
        이 메서드에서 처리합니다.
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """모듈 상태 조회"""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "enabled": self._enabled,
            "process_count": self._process_count,
            "error_count": self._error_count,
            "last_result": bool(self._last_result),
        }

    # ─── 유틸리티 메서드 ───

    def emit(self, event_type: str, data: dict, priority: int = 0) -> None:
        """이벤트 발행 (간편 메서드)"""
        self._event_bus.publish(Event(
            type=event_type,
            data=data,
            source=self.name,
            priority=priority,
        ))

    def safe_process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        예외 안전 process() 래퍼 - 오케스트레이터가 호출
        한 모듈의 실패가 다른 모듈에 영향 주지 않도록 보호
        """
        if not self._enabled or not self._initialized:
            return {}

        try:
            result = self.process(shared_data)
            self._last_result = result or {}
            self._process_count += 1
            return self._last_result
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"[{self.name}] 처리 오류: {e}")
            return {"error": str(e)}

    @property
    def is_ready(self) -> bool:
        """모듈 사용 가능 여부"""
        return self._initialized and self._enabled

    def enable(self) -> None:
        self._enabled = True
        self.logger.info(f"[{self.name}] 활성화")

    def disable(self) -> None:
        self._enabled = False
        self.logger.info(f"[{self.name}] 비활성화")
