"""
공유 스트림 매니저 - 원본 VideoStreamHandler를 래핑하여 공유 스트림 제공

원본 모듈 (직접 import — 수정 시 즉시 반영):
    - Detaction_CCTV/services/stream_handler.py → VideoStreamHandler

통합 레이어 (이 파일에만 존재):
    - 싱글톤 패턴 (여러 모듈이 동일 스트림 공유)
    - 테스트 모드 (test:// prefix → 더미 프레임)
    - 프레임 메타데이터 (timestamp, count)
"""

import os
import cv2
import threading
import time
import logging
from typing import Optional

import numpy as np

from integrated_system.core.module_loader import DETECT_DIR, import_from_file

logger = logging.getLogger(__name__)

# ★ 원본 VideoStreamHandler를 직접 파일 로드 (services/__init__.py의 onvif 의존성 우회) ★
_stream_mod = import_from_file(
    "_orig_stream_handler",
    os.path.join(DETECT_DIR, "services", "stream_handler.py")
)
VideoStreamHandler = _stream_mod.VideoStreamHandler


class SharedStreamManager:
    """
    싱글톤 영상 스트림 관리자

    내부적으로 원본 VideoStreamHandler에 위임합니다.
    원본 stream_handler.py를 수정하면 즉시 반영됩니다.

    지원 소스:
        - 정수(0, 1, ...) : 웹캠
        - RTSP URL 문자열  : IP 카메라
        - "test://..."     : 더미 프레임 (테스트용)
    """

    _instance: Optional['SharedStreamManager'] = None
    _lock_cls = threading.Lock()

    @classmethod
    def get_instance(cls, source_url=None) -> 'SharedStreamManager':
        """싱글톤 인스턴스 반환"""
        with cls._lock_cls:
            if cls._instance is None:
                if source_url is None:
                    raise ValueError("첫 번째 호출 시 source_url이 필요합니다")
                cls._instance = cls(source_url)
            return cls._instance

    @classmethod
    def reset(cls):
        """싱글톤 리셋 (테스트용)"""
        with cls._lock_cls:
            if cls._instance:
                cls._instance.release()
            cls._instance = None

    def __init__(self, source_url):
        self.source_url = source_url
        self.frame_count: int = 0
        self.frame_timestamp: float = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # 테스트 모드 판정
        self._test_mode = isinstance(source_url, str) and source_url.startswith("test://")

        # ★ 원본 VideoStreamHandler (테스트 모드가 아닌 경우) ★
        self._handler: Optional[VideoStreamHandler] = None
        self.current_frame: Optional[np.ndarray] = None

        if self._test_mode:
            self.is_running = True
            logger.info("[Stream] 테스트 모드 활성화 (더미 프레임)")
        else:
            self._handler = VideoStreamHandler(source_url)
            self.is_running = self._handler.is_running

    def start(self) -> 'SharedStreamManager':
        """비동기 프레임 업데이트 시작"""
        if self._test_mode:
            if self._thread is None:
                self._thread = threading.Thread(target=self._test_loop, daemon=True)
                self._thread.start()
                logger.info("[Stream] 테스트 프레임 생성 스레드 시작")
        elif self._handler:
            # ★ 원본 VideoStreamHandler.start() 에 위임 ★
            self._handler.start()
            logger.info("[Stream] 원본 VideoStreamHandler 프레임 수신 시작")
        return self

    def _test_loop(self) -> None:
        """테스트 모드: 더미 프레임 생성"""
        while self.is_running:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, "TEST MODE", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            cv2.putText(dummy, f"Frame: {self.frame_count}", (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            with self._lock:
                self.current_frame = dummy
                self.frame_timestamp = time.time()
                self.frame_count += 1
            time.sleep(0.033)  # ~30 FPS

    def get_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 복사본 반환"""
        if self._test_mode:
            with self._lock:
                return self.current_frame.copy() if self.current_frame is not None else None
        elif self._handler:
            # ★ 원본 VideoStreamHandler.get_frame() 에 위임 ★
            frame = self._handler.get_frame()
            if frame is not None:
                with self._lock:
                    self.frame_count += 1
                    self.frame_timestamp = time.time()
            return frame
        return None

    def get_frame_with_info(self) -> tuple:
        """프레임 + 메타 정보 반환"""
        frame = self.get_frame()
        if frame is not None:
            return frame, self.frame_timestamp, self.frame_count
        return None, 0.0, 0

    def get_resolution(self) -> tuple:
        """해상도 (width, height) 반환"""
        frame = self.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            return w, h
        return 0, 0

    def release(self) -> None:
        """안전한 종료"""
        self.is_running = False
        if self._handler:
            # ★ 원본 VideoStreamHandler.release() 에 위임 ★
            self._handler.release()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("[Stream] 리소스 해제 완료")
