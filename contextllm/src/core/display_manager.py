#!/usr/bin/env python3
"""
OpenCV 디스플레이 매니저
라이브 영상 모드(webcam, network)에서 실시간 카메라 화면과 분석 결과를 오버레이로 표시합니다.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class OverlayResult:
    """오버레이에 표시할 분석 결과"""
    text: str
    situation: str
    emotion: str
    urgency: str
    is_emergency: bool
    timestamp: float


class OpenCVDisplay:
    """OpenCV 기반 실시간 디스플레이"""
    
    def __init__(self, window_name: str = "ContextLLM - Live View"):
        self.window_name = window_name
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.display_thread = None
        
        # 결과 오버레이
        self.current_result: Optional[OverlayResult] = None
        self.result_display_time = 5.0  # 결과 표시 시간 (초)
        
        # 색상 정의 (BGR)
        self.colors = {
            'critical': (0, 0, 220),      # 빨강
            'high': (0, 127, 255),        # 주황
            'medium': (0, 200, 255),      # 노랑
            'low': (0, 200, 0),           # 초록
            'text': (255, 255, 255),      # 흰색
            'bg': (30, 30, 30),           # 어두운 배경
        }
        
        # 한글 폰트 (시스템에 따라 다름)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # 윈도우 생성 플래그
        self.window_created = False
    
    def start(self, video_source=None):
        """디스플레이 시작"""
        if self.running:
            return
        
        self.running = True
        self.video_source = video_source
        
        print(f"🖥️  OpenCV 디스플레이 시작: '{self.window_name}'")
        print("   ESC 또는 'q' 키로 종료할 수 있습니다.\n")
    
    def is_running(self):
        """디스플레이 실행 상태 반환"""
        return self.running
    
    def stop(self):
        """디스플레이 중지"""
        self.running = False
        if self.window_created:
            cv2.destroyAllWindows()
            self.window_created = False
    
    def update_frame(self, frame: np.ndarray):
        """프레임 업데이트"""
        with self.frame_lock:
            self.frame = frame.copy() if frame is not None else None
    
    def update_result(self, result: Dict[str, Any]):
        """분석 결과 업데이트"""
        analysis = result.get('multimodal_analysis', {})
        
        # 긴급도 레벨 결정
        is_emergency = analysis.get('is_emergency', False)
        urgency = analysis.get('urgency_level', 'LOW')
        
        if is_emergency:
            level = 'critical'
        elif urgency in ['HIGH', '높음']:
            level = 'high'
        elif urgency in ['MEDIUM', '중간']:
            level = 'medium'
        else:
            level = 'low'
        
        self.current_result = OverlayResult(
            text=result.get('transcribed_text', ''),
            situation=analysis.get('situation_type', 'N/A'),
            emotion=analysis.get('emotion', 'N/A'),
            urgency=level,
            is_emergency=is_emergency,
            timestamp=time.time()
        )
    
    def render(self):
        """메인 스레드에서 호출 - 프레임 렌더링 및 키 입력 처리"""
        # 프레임 가져오기
        with self.frame_lock:
            if self.frame is not None:
                display_frame = self.frame.copy()
            else:
                # 빈 프레임 (대기 화면)
                display_frame = self._create_waiting_frame()
        
        # 오버레이 추가
        display_frame = self._add_overlay(display_frame)
        
        # 첫 렌더링 시 윈도우 생성
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            self.window_created = True
        
        # 화면에 표시
        cv2.imshow(self.window_name, display_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            self.running = False
            return False
        
        return self.running
    
    def _create_waiting_frame(self) -> np.ndarray:
        """대기 화면 생성"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = self.colors['bg']
        
        # 중앙에 텍스트
        text = "Waiting for video..."
        text_size = cv2.getTextSize(text, self.font, 1, 2)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (x, y), self.font, 1, (100, 100, 100), 2)
        
        return frame
    
    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """분석 결과 오버레이 추가"""
        h, w = frame.shape[:2]
        
        # 상단 상태바
        cv2.rectangle(frame, (0, 0), (w, 40), self.colors['bg'], -1)
        cv2.putText(frame, "ContextLLM Live", (10, 28), self.font, 0.7, self.colors['text'], 2)
        
        # 현재 시간 표시
        current_time = time.strftime("%H:%M:%S")
        time_text_size = cv2.getTextSize(current_time, self.font, 0.6, 1)[0]
        cv2.putText(frame, current_time, (w - time_text_size[0] - 10, 28), 
                   self.font, 0.6, self.colors['text'], 1)
        
        # 분석 결과 오버레이
        if self.current_result:
            elapsed = time.time() - self.current_result.timestamp
            
            if elapsed < self.result_display_time:
                self._draw_result_overlay(frame, self.current_result)
            else:
                # 시간이 지나면 결과 제거
                pass
        
        return frame
    
    def _draw_result_overlay(self, frame: np.ndarray, result: OverlayResult):
        """분석 결과 박스 그리기"""
        h, w = frame.shape[:2]
        
        # 색상 결정
        color = self.colors.get(result.urgency, self.colors['low'])
        
        # 긴급 상황이면 테두리 깜빡임 효과
        if result.is_emergency:
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (5, 5), (w-5, h-5), self.colors['critical'], 4)
        
        # 하단 결과 박스 배경
        box_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - box_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 좌측 색상 바
        cv2.rectangle(frame, (0, h - box_height), (8, h), color, -1)
        
        # 텍스트 표시
        y_offset = h - box_height + 25
        line_height = 25
        
        # 음성 텍스트
        text_line = f"Voice: {result.text[:40]}..." if len(result.text) > 40 else f"Voice: {result.text}"
        cv2.putText(frame, text_line, (15, y_offset), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        y_offset += line_height
        
        # 상황 유형
        cv2.putText(frame, f"Situation: {result.situation}", (15, y_offset),
                   self.font, self.font_scale, color, self.font_thickness)
        y_offset += line_height
        
        # 감정 상태
        cv2.putText(frame, f"Emotion: {result.emotion}", (15, y_offset),
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        # 우측 상단에 긴급도 배지
        if result.is_emergency:
            badge_text = "EMERGENCY"
            badge_color = self.colors['critical']
        else:
            badge_text = result.urgency.upper()
            badge_color = color
        
        badge_size = cv2.getTextSize(badge_text, self.font, 0.8, 2)[0]
        badge_x = w - badge_size[0] - 20
        badge_y = h - box_height + 30
        
        # 배지 배경
        cv2.rectangle(frame, 
                     (badge_x - 10, badge_y - badge_size[1] - 5),
                     (badge_x + badge_size[0] + 10, badge_y + 5),
                     badge_color, -1)
        cv2.putText(frame, badge_text, (badge_x, badge_y),
                   self.font, 0.8, (255, 255, 255), 2)
    
    def is_running(self) -> bool:
        """실행 중인지 확인"""
        return self.running


# 전역 인스턴스
_display_instance: Optional[OpenCVDisplay] = None


def get_display() -> OpenCVDisplay:
    """싱글톤 디스플레이 인스턴스 반환"""
    global _display_instance
    if _display_instance is None:
        _display_instance = OpenCVDisplay()
    return _display_instance


def start_display():
    """디스플레이 시작"""
    get_display().start()


def stop_display():
    """디스플레이 중지"""
    if _display_instance:
        _display_instance.stop()


def update_frame(frame: np.ndarray):
    """프레임 업데이트"""
    if _display_instance and _display_instance.is_running():
        _display_instance.update_frame(frame)


def update_result(result: Dict[str, Any]):
    """결과 업데이트"""
    if _display_instance and _display_instance.is_running():
        _display_instance.update_result(result)


if __name__ == '__main__':
    # 테스트: 웹캠으로 테스트
    print("OpenCV 디스플레이 테스트")
    
    display = OpenCVDisplay()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다")
        exit(1)
    
    display.start()
    
    # 테스트 결과
    test_result = {
        'transcribed_text': '테스트 음성입니다',
        'multimodal_analysis': {
            'situation_type': '정상',
            'emotion': '중립',
            'urgency_level': 'LOW',
            'is_emergency': False
        }
    }
    
    try:
        while display.is_running():
            ret, frame = cap.read()
            if ret:
                display.update_frame(frame)
            
            # 5초마다 결과 업데이트
            if int(time.time()) % 5 == 0:
                display.update_result(test_result)
            
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        display.stop()
        cap.release()
