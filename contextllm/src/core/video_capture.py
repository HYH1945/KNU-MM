#!/usr/bin/env python3
"""
비디오 캡처 및 프레임 추출 모듈
실시간 비디오 스트림에서 프레임을 추출하여 멀티모달 분석에 활용

사용법:
    # 웹캠에서 실시간 모니터링
    monitor = VideoMonitor()
    monitor.start_monitoring(on_frame_callback=your_callback)
    
    # 특정 간격으로 프레임 추출
    extractor = VideoFrameExtractor()
    frames = extractor.extract_frames("video.mp4", interval=2.0)
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import threading
import time


class VideoFrameExtractor:
    """비디오 파일에서 프레임 추출"""
    
    def __init__(self):
        """프레임 추출기 초기화"""
        pass
    
    def extract_frames(
        self, 
        video_path: str, 
        interval: float = 1.0,
        max_frames: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        비디오 파일에서 일정 간격으로 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            interval: 프레임 추출 간격 (초)
            max_frames: 최대 추출 프레임 수 (None이면 제한 없음)
            save_dir: 프레임 저장 디렉토리 (None이면 저장하지 않음)
        
        Returns:
            [(timestamp, frame), ...] 리스트
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없음: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없음: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📹 비디오 분석 시작: {video_path}")
        print(f"   FPS: {fps}, 추출 간격: {interval}초 ({frame_interval} 프레임마다)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격마다 프레임 추출
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
                
                if save_dir:
                    filename = f"frame_{extracted_count:04d}_{timestamp:.2f}s.jpg"
                    filepath = Path(save_dir) / filename
                    cv2.imwrite(str(filepath), frame)
                
                extracted_count += 1
                print(f"   ✅ 프레임 {extracted_count} 추출 (시간: {timestamp:.2f}초)")
                
                # 최대 프레임 수 도달 시 중단
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"✅ 총 {extracted_count}개 프레임 추출 완료")
        
        return frames
    
    def extract_key_frames(
        self,
        video_path: str,
        threshold: float = 30.0,
        max_frames: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        장면 변화가 큰 키프레임만 추출
        
        Args:
            video_path: 비디오 파일 경로
            threshold: 장면 변화 임계값 (높을수록 변화가 큰 프레임만 추출)
            max_frames: 최대 추출 프레임 수
            save_dir: 프레임 저장 디렉토리
        
        Returns:
            [(timestamp, frame), ...] 리스트
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없음: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없음: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = 0
        extracted_count = 0
        prev_frame_gray = None
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📹 키프레임 추출 시작: {video_path}")
        print(f"   임계값: {threshold}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 그레이스케일로 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 첫 프레임은 무조건 추가
            if prev_frame_gray is None:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
                extracted_count += 1
                
                if save_dir:
                    filename = f"keyframe_{extracted_count:04d}_{timestamp:.2f}s.jpg"
                    filepath = Path(save_dir) / filename
                    cv2.imwrite(str(filepath), frame)
                
                print(f"   ✅ 키프레임 {extracted_count} 추출 (시간: {timestamp:.2f}초)")
            else:
                # 이전 프레임과의 차이 계산
                diff = cv2.absdiff(prev_frame_gray, gray)
                mean_diff = np.mean(diff)
                
                # 임계값을 넘으면 키프레임으로 추출
                if mean_diff > threshold:
                    timestamp = frame_count / fps
                    frames.append((timestamp, frame))
                    extracted_count += 1
                    
                    if save_dir:
                        filename = f"keyframe_{extracted_count:04d}_{timestamp:.2f}s.jpg"
                        filepath = Path(save_dir) / filename
                        cv2.imwrite(str(filepath), frame)
                    
                    print(f"   ✅ 키프레임 {extracted_count} 추출 (시간: {timestamp:.2f}초, 변화도: {mean_diff:.2f})")
                    
                    # 최대 프레임 수 도달 시 중단
                    if max_frames and extracted_count >= max_frames:
                        break
            
            prev_frame_gray = gray
            frame_count += 1
        
        cap.release()
        print(f"✅ 총 {extracted_count}개 키프레임 추출 완료")
        
        return frames


class VideoMonitor:
    """실시간 비디오 모니터링"""
    
    def __init__(self, camera_id: int = 0):
        """
        비디오 모니터 초기화
        
        Args:
            camera_id: 카메라 ID (기본값: 0)
        """
        self.camera_id = camera_id
        self.is_monitoring = False
        self.cap = None
        self.monitoring_thread = None
        self.frame_callback = None
        self.frame_interval = 1.0  # 콜백 호출 간격 (초)
    
    def start_monitoring(
        self,
        on_frame_callback: Callable[[np.ndarray, float], None],
        frame_interval: float = 1.0,
        show_preview: bool = False
    ):
        """
        실시간 모니터링 시작
        
        Args:
            on_frame_callback: 프레임 처리 콜백 함수 callback(frame, timestamp)
            frame_interval: 콜백 호출 간격 (초)
            show_preview: 프리뷰 창 표시 여부
        """
        if self.is_monitoring:
            print("⚠️  이미 모니터링 중입니다")
            return
        
        self.frame_callback = on_frame_callback
        self.frame_interval = frame_interval
        self.is_monitoring = True
        
        # 별도 스레드에서 모니터링 실행
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(show_preview,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        print(f"✅ 비디오 모니터링 시작 (카메라 {self.camera_id})")
    
    def _monitoring_loop(self, show_preview: bool):
        """모니터링 루프 (내부 메서드)"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ 카메라 {self.camera_id}를 열 수 없습니다")
            self.is_monitoring = False
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = int(fps * self.frame_interval)
        frame_count = 0
        start_time = time.time()
        
        print(f"📹 모니터링 시작 (FPS: {fps}, 간격: {self.frame_interval}초)")
        
        while self.is_monitoring:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️  프레임 읽기 실패")
                break
            
            # 지정된 간격마다 콜백 호출
            if frame_count % frame_skip == 0:
                timestamp = time.time() - start_time
                
                try:
                    if self.frame_callback:
                        self.frame_callback(frame, timestamp)
                except Exception as e:
                    print(f"❌ 콜백 오류: {e}")
            
            # 프리뷰 창 표시
            if show_preview:
                cv2.imshow(f'Video Monitor (Camera {self.camera_id})', frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 사용자가 모니터링을 종료했습니다")
                    break
            
            frame_count += 1
        
        self.cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print("✅ 모니터링 종료")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if not self.is_monitoring:
            print("⚠️  모니터링 중이 아닙니다")
            return
        
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=3.0)
        
        print("✅ 모니터링 중지됨")
    
    def capture_current_frame(self) -> Optional[np.ndarray]:
        """
        현재 프레임 캡처 (모니터링 중일 때)
        
        Returns:
            현재 프레임 (numpy array) 또는 None
        """
        if not self.is_monitoring or not self.cap or not self.cap.isOpened():
            print("❌ 모니터링 중이 아니거나 카메라가 열려있지 않습니다")
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("❌ 프레임 캡처 실패")
            return None


class MotionDetector:
    """움직임 감지기"""
    
    def __init__(self, threshold: float = 25.0, min_area: int = 500):
        """
        움직임 감지기 초기화
        
        Args:
            threshold: 움직임 감지 임계값
            min_area: 최소 움직임 영역 크기
        """
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
    
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        """
        프레임에서 움직임 감지
        
        Args:
            frame: 현재 프레임
        
        Returns:
            (움직임 감지 여부, 움직임 영역 리스트 [(x, y, w, h), ...])
        """
        # 그레이스케일 변환 및 블러 처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 첫 프레임 저장
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, []
        
        # 프레임 차이 계산
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_areas.append((x, y, w, h))
        
        self.prev_frame = gray
        
        has_motion = len(motion_areas) > 0
        return has_motion, motion_areas
    
    def draw_motion_boxes(self, frame: np.ndarray, motion_areas: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        움직임 영역에 박스 그리기
        
        Args:
            frame: 원본 프레임
            motion_areas: 움직임 영역 리스트
        
        Returns:
            박스가 그려진 프레임
        """
        result = frame.copy()
        
        for (x, y, w, h) in motion_areas:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 움직임 감지 텍스트
        if motion_areas:
            cv2.putText(result, f"Motion Detected ({len(motion_areas)} areas)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result


if __name__ == "__main__":
    # 테스트
    print("=" * 70)
    print("📹 비디오 캡처 모듈 테스트")
    print("=" * 70)
    
    # 1. 웹캠 모니터링 테스트
    print("\n1️⃣ 웹캠 모니터링 테스트 (5초)")
    
    def on_frame(frame, timestamp):
        print(f"   프레임 수신: {timestamp:.2f}초, 크기: {frame.shape}")
    
    monitor = VideoMonitor(camera_id=0)
    monitor.start_monitoring(on_frame_callback=on_frame, frame_interval=1.0)
    
    time.sleep(5)
    monitor.stop_monitoring()
    
    print("\n✅ 테스트 완료")
