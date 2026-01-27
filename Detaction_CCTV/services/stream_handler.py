import cv2
import threading
import os
import time
from typing import Optional
import numpy as np

class VideoStreamHandler:
    def __init__(self, source_url: str):
        # TCP 전송 강제 (패킷 손실 방지)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.source_url = source_url
        self.capture = cv2.VideoCapture(self.source_url)

        self.is_running: bool = False
        self.current_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.thread = None  # [추가] 스레드 객체를 저장할 변수
        
        if self.capture.isOpened():
            self.is_running = True
            print("[Stream] Video connection established.")
        else:
            print("[Stream] Failed to open video stream.")

    def start(self) -> 'VideoStreamHandler':
        # 비동기 Frame Update start
        if self.is_running:
            # [수정] 스레드 객체를 self.thread에 저장
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        return self

    def _update_loop(self) -> None:
        # Frame update loop
        while self.is_running:
            # 1. 캡처 객체 상태 확인
            if not self.capture.isOpened():
                self._reconnect()
                continue
                
            # 2. 프레임 읽기
            grabbed, frame = self.capture.read()
            
            if grabbed:
                with self._lock:
                    self.current_frame = frame
            else:
                print("[Stream] Signal lost. Reconnecting...")
                self._reconnect()
            
            # [추가] CPU 점유율 폭주 방지 (Mac 과부하 방지)
            time.sleep(0.01)

    def _reconnect(self):
        self.capture.release()
        time.sleep(1)
        self.capture = cv2.VideoCapture(self.source_url)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def release(self) -> None:
        """안전한 종료: 스레드가 멈출 때까지 기다린 후 자원 해제"""
        self.is_running = False  # 1. 루프 정지 신호 보냄
        
        # 2. 스레드가 현재 작업을 마칠 때까지 대기 (Join)
        # 이 부분이 없으면 'Double free' 오류 발생
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

        # 3. 안전하게 카메라 해제
        if self.capture.isOpened():
            self.capture.release()
        print("[Stream] Resources released.")