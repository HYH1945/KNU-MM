"""
이벤트 버스 - 모듈 간 느슨한 결합을 위한 Pub/Sub 시스템

모듈들이 직접 서로를 참조하지 않고, 이벤트를 통해 통신합니다.

이벤트 타입 규약:
    - "mic.doa_detected"       : 마이크 어레이에서 음원 방향 감지
    - "mic.speech_detected"    : 마이크 어레이에서 음성 감지
    - "yolo.objects_detected"  : YOLO에서 객체 감지
    - "yolo.person_detected"   : YOLO에서 사람 감지 (우선순위 높은 이벤트)
    - "yolo.no_objects"        : YOLO에서 객체 미감지 (순찰 모드 전환)
    - "llm.analysis_complete"  : ContextLLM 분석 완료
    - "llm.emergency"          : ContextLLM 긴급 상황 판정
    - "ptz.move_requested"     : PTZ 이동 요청
    - "ptz.moved"              : PTZ 이동 완료
    - "system.mode_changed"    : 시스템 모드 변경
    - "system.shutdown"        : 시스템 종료 신호
"""

import threading
import time
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """이벤트 객체"""
    type: str                          # 이벤트 타입 (예: "yolo.person_detected")
    data: Dict[str, Any]               # 이벤트 데이터
    source: str = ""                   # 발행 모듈 이름
    timestamp: float = field(default_factory=time.time)
    priority: int = 0                  # 0=일반, 1=높음, 2=긴급

    def __repr__(self):
        return f"Event({self.type}, src={self.source}, priority={self.priority})"


class EventBus:
    """
    스레드 안전 이벤트 버스
    
    사용법:
        bus = EventBus()
        
        # 구독
        bus.subscribe("yolo.person_detected", handler_func)
        
        # 발행
        bus.publish(Event(type="yolo.person_detected", data={"count": 3}, source="yolo"))
        
        # 패턴 구독 (와일드카드)
        bus.subscribe("yolo.*", on_any_yolo_event)
    """

    def __init__(self, max_workers: int = 4, async_mode: bool = True):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._async_mode = async_mode
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if async_mode else None
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._running = True

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        이벤트 구독
        
        Args:
            event_type: 이벤트 타입 (예: "yolo.person_detected", "yolo.*")
            callback: 콜백 함수 (Event 객체를 인자로 받음)
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug(f"구독 추가: {event_type} -> {callback.__name__}")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """이벤트 구독 해제"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def publish(self, event: Event) -> None:
        """
        이벤트 발행 - 모든 구독자에게 전달
        
        Args:
            event: 발행할 이벤트 객체
        """
        if not self._running:
            return

        # 이벤트 히스토리 저장
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # 해당 이벤트 타입의 구독자 + 와일드카드 구독자 수집
        callbacks = []
        with self._lock:
            # 정확한 매칭
            callbacks.extend(self._subscribers.get(event.type, []))
            
            # 와일드카드 매칭 (예: "yolo.*"는 "yolo.person_detected"에 매칭)
            prefix = event.type.split(".")[0] + ".*"
            callbacks.extend(self._subscribers.get(prefix, []))
            
            # 전체 와일드카드
            callbacks.extend(self._subscribers.get("*", []))

        # 콜백 실행
        for callback in callbacks:
            try:
                if self._async_mode and self._executor:
                    self._executor.submit(self._safe_call, callback, event)
                else:
                    self._safe_call(callback, event)
            except Exception as e:
                logger.error(f"이벤트 발행 오류 [{event.type}]: {e}")

    def publish_simple(self, event_type: str, data: dict, source: str = "", priority: int = 0) -> None:
        """간편 이벤트 발행 (Event 객체를 직접 만들지 않아도 됨)"""
        self.publish(Event(
            type=event_type,
            data=data,
            source=source,
            priority=priority,
        ))

    def _safe_call(self, callback: Callable, event: Event) -> None:
        """예외 안전 콜백 실행"""
        try:
            callback(event)
        except Exception as e:
            logger.error(f"콜백 실행 오류 [{callback.__name__}] for {event.type}: {e}")

    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 50) -> List[Event]:
        """최근 이벤트 조회"""
        with self._lock:
            events = self._event_history
            if event_type:
                events = [e for e in events if e.type == event_type or e.type.startswith(event_type.rstrip("*"))]
            return events[-limit:]

    def shutdown(self) -> None:
        """이벤트 버스 종료"""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("EventBus 종료")
