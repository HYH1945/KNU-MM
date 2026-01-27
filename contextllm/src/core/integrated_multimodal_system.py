#!/usr/bin/env python3
"""
통합 멀티모달 시스템
음성 감지 기반으로 작동하며, 음성이 감지되면 동시에:
1. 음성 특성 분석 (피치, 에너지, 속도 등)
2. 영상 캡처 및 분석 (다운샘플링 적용)

사용법:
    system = IntegratedMultimodalSystem()
    
    # 음성 감지 시 자동으로 영상 분석
    system.start_monitoring()
    
    # 또는 단발성 분석
    result = system.analyze_once()
"""

import os
import sys
import json
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️  speech_recognition이 설치되지 않았습니다: pip install SpeechRecognition")

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI가 설치되지 않았습니다: pip install openai python-dotenv")

# 내부 모듈 임포트
try:
    from core.voice_characteristics import VoiceCharacteristicsAnalyzer
    VOICE_CHARACTERISTICS_AVAILABLE = True
except ImportError:
    try:
        from voice_characteristics import VoiceCharacteristicsAnalyzer
        VOICE_CHARACTERISTICS_AVAILABLE = True
    except ImportError:
        VOICE_CHARACTERISTICS_AVAILABLE = False
        print("⚠️  음성 특성 분석 모듈을 불러올 수 없습니다")

try:
    from core.multimodal_analyzer import MultimodalAnalyzer
    MULTIMODAL_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from multimodal_analyzer import MultimodalAnalyzer
        MULTIMODAL_ANALYZER_AVAILABLE = True
    except ImportError:
        MULTIMODAL_ANALYZER_AVAILABLE = False
        print("⚠️  멀티모달 분석 모듈을 불러올 수 없습니다")


@dataclass
class DownsamplingConfig:
    """다운샘플링 설정"""
    # 이미지 다운샘플링
    max_image_size: int = 640  # 최대 이미지 크기 (픽셀)
    jpeg_quality: int = 75  # JPEG 품질 (1-100)
    
    # 비디오 다운샘플링
    video_fps: float = 2.0  # 분석용 FPS (원본에서 샘플링)
    max_video_frames: int = 10  # 최대 분석 프레임 수 (5초 * 2fps = 10)
    video_resolution_scale: float = 0.5  # 비디오 해상도 스케일 (0.5 = 50%)
    
    # 비디오 캡처 시간
    video_capture_duration: float = 5.0  # 캡처할 비디오 길이 (초)


class VideoSourceType:
    """비디오 소스 타입"""
    WEBCAM = "webcam"           # 웹캠
    FILE = "file"               # 파일 (이미지/비디오)
    NETWORK = "network"         # 네트워크 카메라 (RTSP/HTTP)
    TESTSET = "testset"         # 테스트셋 폴더


class VideoDownsampler:
    """비디오/이미지 다운샘플링 유틸리티"""
    
    def __init__(self, config: DownsamplingConfig = None):
        self.config = config or DownsamplingConfig()
    
    def downsample_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 다운샘플링
        
        Args:
            image: 원본 이미지 (numpy array, BGR format)
        
        Returns:
            다운샘플링된 이미지
        """
        if image is None:
            return None
        
        height, width = image.shape[:2]
        max_size = self.config.max_image_size
        
        # 크기가 max_size보다 크면 리사이징
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # INTER_AREA: 축소에 적합한 보간법
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)
            
        return image
    
    def downsample_video_frames(
        self, 
        frames: List[np.ndarray], 
        timestamps: List[float] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        비디오 프레임 리스트 다운샘플링
        
        Args:
            frames: 프레임 리스트
            timestamps: 타임스탬프 리스트
        
        Returns:
            (다운샘플링된 프레임 리스트, 타임스탬프 리스트)
        """
        if not frames:
            return [], []
        
        # 최대 프레임 수로 제한
        max_frames = self.config.max_video_frames
        if len(frames) > max_frames:
            # 균등하게 샘플링
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            if timestamps:
                timestamps = [timestamps[i] for i in indices]
        
        # 각 프레임 다운샘플링
        downsampled_frames = []
        scale = self.config.video_resolution_scale
        
        for frame in frames:
            if scale < 1.0:
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            
            # 추가로 max_image_size 적용
            frame = self.downsample_image(frame)
            downsampled_frames.append(frame)
        
        return downsampled_frames, timestamps or []
    
    def encode_frame_to_jpeg(self, frame: np.ndarray) -> bytes:
        """프레임을 JPEG 바이트로 인코딩"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()


class BaseVideoSource:
    """비디오 소스 기본 클래스 (추상)"""
    
    def __init__(self):
        self.is_opened = False
        self.lock = threading.Lock()
        self.source_type = None
    
    def open(self) -> bool:
        """소스 열기"""
        raise NotImplementedError
    
    def close(self):
        """소스 닫기"""
        raise NotImplementedError
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """단일 프레임 캡처"""
        raise NotImplementedError
    
    def capture_video_segment(
        self, 
        duration: float = 5.0, 
        target_fps: float = 2.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """비디오 세그먼트 캡처"""
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Any]:
        """소스 정보 반환"""
        return {
            "source_type": self.source_type,
            "is_opened": self.is_opened
        }


class WebcamVideoSource(BaseVideoSource):
    """웹캠 비디오 소스"""
    
    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        self.source_type = VideoSourceType.WEBCAM
    
    def open(self) -> bool:
        """웹캠 열기"""
        with self.lock:
            if self.is_opened:
                return True
            
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                self.is_opened = True
                print(f"✅ 웹캠 {self.camera_id} 열림")
                return True
            else:
                print(f"❌ 웹캠 {self.camera_id}를 열 수 없습니다")
                return False
    
    def close(self):
        """웹캠 닫기"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.is_opened = False
                print("✅ 웹캠 닫힘")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """단일 프레임 캡처"""
        with self.lock:
            if not self.is_opened or not self.cap:
                return None
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
    
    def capture_video_segment(
        self, 
        duration: float = 5.0, 
        target_fps: float = 2.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """비디오 세그먼트 캡처"""
        frames = []
        timestamps = []
        
        with self.lock:
            if not self.is_opened or not self.cap:
                print("❌ 웹캠이 열려있지 않습니다")
                return frames, timestamps
            
            original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
            
            start_time = time.time()
            frame_count = 0
            
            print(f"📹 웹캠 캡처 중 ({duration}초, {target_fps}fps)...")
            
            while (time.time() - start_time) < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = time.time() - start_time
                    frames.append(frame.copy())
                    timestamps.append(timestamp)
                
                frame_count += 1
            
            print(f"   ✅ {len(frames)}개 프레임 캡처 완료")
        
        return frames, timestamps
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["camera_id"] = self.camera_id
        if self.cap and self.is_opened:
            info["fps"] = self.cap.get(cv2.CAP_PROP_FPS)
            info["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return info


class NetworkVideoSource(BaseVideoSource):
    """네트워크 카메라 비디오 소스 (RTSP/HTTP)"""
    
    def __init__(self, url: str):
        """
        Args:
            url: 카메라 URL (예: rtsp://192.168.1.100:554/stream, http://192.168.1.100:8080/video)
        """
        super().__init__()
        self.url = url
        self.cap = None
        self.source_type = VideoSourceType.NETWORK
    
    def open(self) -> bool:
        """네트워크 카메라 연결"""
        with self.lock:
            if self.is_opened:
                return True
            
            print(f"🌐 네트워크 카메라 연결 중: {self.url}")
            self.cap = cv2.VideoCapture(self.url)
            
            # 버퍼 크기 줄이기 (지연 감소)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_opened = True
                print(f"✅ 네트워크 카메라 연결됨: {self.url}")
                return True
            else:
                print(f"❌ 네트워크 카메라 연결 실패: {self.url}")
                return False
    
    def close(self):
        """네트워크 카메라 연결 종료"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.is_opened = False
                print("✅ 네트워크 카메라 연결 종료")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """단일 프레임 캡처"""
        with self.lock:
            if not self.is_opened or not self.cap:
                return None
            
            # 버퍼 비우기 (최신 프레임 가져오기)
            for _ in range(3):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
    
    def capture_video_segment(
        self, 
        duration: float = 5.0, 
        target_fps: float = 2.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """비디오 세그먼트 캡처"""
        frames = []
        timestamps = []
        
        with self.lock:
            if not self.is_opened or not self.cap:
                print("❌ 네트워크 카메라가 연결되어 있지 않습니다")
                return frames, timestamps
            
            frame_interval = 1.0 / target_fps
            start_time = time.time()
            last_capture_time = 0
            
            print(f"📹 네트워크 카메라 캡처 중 ({duration}초, {target_fps}fps)...")
            
            while (time.time() - start_time) < duration:
                current_time = time.time() - start_time
                
                if current_time - last_capture_time >= frame_interval:
                    ret, frame = self.cap.read()
                    if ret:
                        frames.append(frame.copy())
                        timestamps.append(current_time)
                        last_capture_time = current_time
                
                time.sleep(0.01)  # CPU 사용량 줄이기
            
            print(f"   ✅ {len(frames)}개 프레임 캡처 완료")
        
        return frames, timestamps
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["url"] = self.url
        if self.cap and self.is_opened:
            info["fps"] = self.cap.get(cv2.CAP_PROP_FPS)
            info["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return info


class FileVideoSource(BaseVideoSource):
    """파일 기반 비디오 소스 (이미지 또는 비디오 파일)"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: 이미지 또는 비디오 파일 경로
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.cap = None
        self.is_video = False
        self.is_image = False
        self.image = None
        self.source_type = VideoSourceType.FILE
        
        # 파일 타입 확인
        self._detect_file_type()
    
    def _detect_file_type(self):
        """파일 타입 감지"""
        suffix = self.file_path.suffix.lower()
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        if suffix in video_extensions:
            self.is_video = True
        elif suffix in image_extensions:
            self.is_image = True
    
    def open(self) -> bool:
        """파일 열기"""
        with self.lock:
            if self.is_opened:
                return True
            
            if not self.file_path.exists():
                print(f"❌ 파일을 찾을 수 없습니다: {self.file_path}")
                return False
            
            if self.is_video:
                self.cap = cv2.VideoCapture(str(self.file_path))
                if self.cap.isOpened():
                    self.is_opened = True
                    print(f"✅ 비디오 파일 열림: {self.file_path.name}")
                    return True
                else:
                    print(f"❌ 비디오 파일을 열 수 없습니다: {self.file_path}")
                    return False
            
            elif self.is_image:
                self.image = cv2.imread(str(self.file_path))
                if self.image is not None:
                    self.is_opened = True
                    print(f"✅ 이미지 파일 열림: {self.file_path.name}")
                    return True
                else:
                    print(f"❌ 이미지 파일을 열 수 없습니다: {self.file_path}")
                    return False
            
            else:
                print(f"❌ 지원하지 않는 파일 형식: {self.file_path.suffix}")
                return False
    
    def close(self):
        """파일 닫기"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.image = None
            self.is_opened = False
            print(f"✅ 파일 닫힘: {self.file_path.name}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """프레임/이미지 가져오기"""
        with self.lock:
            if not self.is_opened:
                return None
            
            if self.is_image:
                return self.image.copy() if self.image is not None else None
            
            elif self.is_video and self.cap:
                ret, frame = self.cap.read()
                if ret:
                    return frame
                else:
                    # 비디오 끝에 도달하면 처음으로 되감기
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    return frame if ret else None
            
            return None
    
    def capture_video_segment(
        self, 
        duration: float = 5.0, 
        target_fps: float = 2.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """비디오 세그먼트 캡처"""
        frames = []
        timestamps = []
        
        with self.lock:
            if not self.is_opened:
                print("❌ 파일이 열려있지 않습니다")
                return frames, timestamps
            
            if self.is_image:
                # 이미지인 경우 같은 이미지를 여러 번 반환
                num_frames = int(duration * target_fps)
                print(f"📷 이미지에서 {num_frames}개 프레임 생성...")
                
                for i in range(num_frames):
                    frames.append(self.image.copy())
                    timestamps.append(i / target_fps)
                
                print(f"   ✅ {len(frames)}개 프레임 생성 완료")
                return frames, timestamps
            
            elif self.is_video and self.cap:
                original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_frames / original_fps
                
                # 요청된 시간이 비디오 길이보다 길면 비디오 길이로 제한
                actual_duration = min(duration, video_duration)
                
                frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
                frame_count = 0
                
                print(f"📹 비디오 파일에서 캡처 중 ({actual_duration:.1f}초, {target_fps}fps)...")
                
                # 비디오 처음으로 되감기
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    current_time = frame_count / original_fps
                    if current_time > actual_duration:
                        break
                    
                    if frame_count % frame_interval == 0:
                        frames.append(frame.copy())
                        timestamps.append(current_time)
                    
                    frame_count += 1
                
                print(f"   ✅ {len(frames)}개 프레임 캡처 완료")
        
        return frames, timestamps
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["file_path"] = str(self.file_path)
        info["is_video"] = self.is_video
        info["is_image"] = self.is_image
        
        if self.is_video and self.cap and self.is_opened:
            info["fps"] = self.cap.get(cv2.CAP_PROP_FPS)
            info["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["total_frames"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
        
        elif self.is_image and self.image is not None:
            info["height"], info["width"] = self.image.shape[:2]
        
        return info
    
    def seek(self, position: float):
        """비디오에서 특정 위치로 이동 (초 단위)"""
        if self.is_video and self.cap and self.is_opened:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_num = int(position * fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


class TestsetVideoSource(BaseVideoSource):
    """테스트셋 폴더 비디오 소스 (폴더 내 이미지/비디오 순차 재생)"""
    
    def __init__(self, folder_path: str, loop: bool = True):
        """
        Args:
            folder_path: 테스트셋 폴더 경로
            loop: 파일 끝에 도달하면 처음부터 다시 시작할지 여부
        """
        super().__init__()
        self.folder_path = Path(folder_path)
        self.loop = loop
        self.source_type = VideoSourceType.TESTSET
        
        self.files: List[Path] = []
        self.current_index = 0
        self.current_source: Optional[FileVideoSource] = None
    
    def _scan_files(self):
        """폴더 내 미디어 파일 스캔"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        all_extensions = video_extensions | image_extensions
        
        self.files = sorted([
            f for f in self.folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in all_extensions
        ])
        
        print(f"📁 테스트셋 폴더 스캔 완료: {len(self.files)}개 파일")
        for f in self.files:
            print(f"   - {f.name}")
    
    def open(self) -> bool:
        """테스트셋 폴더 열기"""
        with self.lock:
            if self.is_opened:
                return True
            
            if not self.folder_path.exists():
                print(f"❌ 폴더를 찾을 수 없습니다: {self.folder_path}")
                return False
            
            if not self.folder_path.is_dir():
                print(f"❌ 디렉토리가 아닙니다: {self.folder_path}")
                return False
            
            self._scan_files()
            
            if not self.files:
                print(f"❌ 폴더에 미디어 파일이 없습니다: {self.folder_path}")
                return False
            
            # 첫 번째 파일 열기
            self.current_index = 0
            self.current_source = FileVideoSource(str(self.files[0]))
            
            if self.current_source.open():
                self.is_opened = True
                print(f"✅ 테스트셋 준비 완료: {self.folder_path.name}")
                return True
            else:
                return False
    
    def close(self):
        """테스트셋 닫기"""
        with self.lock:
            if self.current_source:
                self.current_source.close()
                self.current_source = None
            self.is_opened = False
            self.current_index = 0
            print("✅ 테스트셋 닫힘")
    
    def _next_file(self) -> bool:
        """다음 파일로 이동"""
        if self.current_source:
            self.current_source.close()
        
        self.current_index += 1
        
        if self.current_index >= len(self.files):
            if self.loop:
                self.current_index = 0
                print("🔄 테스트셋 처음부터 다시 시작")
            else:
                print("📁 테스트셋 끝에 도달")
                return False
        
        self.current_source = FileVideoSource(str(self.files[self.current_index]))
        return self.current_source.open()
    
    def select_file(self, index: int) -> bool:
        """특정 인덱스의 파일 선택"""
        with self.lock:
            if index < 0 or index >= len(self.files):
                print(f"❌ 잘못된 인덱스: {index} (0-{len(self.files)-1})")
                return False
            
            if self.current_source:
                self.current_source.close()
            
            self.current_index = index
            self.current_source = FileVideoSource(str(self.files[index]))
            success = self.current_source.open()
            
            if success:
                print(f"📂 파일 선택: {self.files[index].name}")
            
            return success
    
    def select_file_by_name(self, filename: str) -> bool:
        """파일 이름으로 선택"""
        for i, f in enumerate(self.files):
            if f.name == filename or f.stem == filename:
                return self.select_file(i)
        
        print(f"❌ 파일을 찾을 수 없습니다: {filename}")
        return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """현재 파일에서 프레임 캡처"""
        with self.lock:
            if not self.is_opened or not self.current_source:
                return None
            
            return self.current_source.capture_frame()
    
    def capture_video_segment(
        self, 
        duration: float = 5.0, 
        target_fps: float = 2.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """현재 파일에서 비디오 세그먼트 캡처"""
        with self.lock:
            if not self.is_opened or not self.current_source:
                print("❌ 테스트셋이 열려있지 않습니다")
                return [], []
            
            return self.current_source.capture_video_segment(duration, target_fps)
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["folder_path"] = str(self.folder_path)
        info["total_files"] = len(self.files)
        info["current_index"] = self.current_index
        info["current_file"] = self.files[self.current_index].name if self.files else None
        info["loop"] = self.loop
        
        if self.current_source:
            info["current_source_info"] = self.current_source.get_info()
        
        return info
    
    def list_files(self) -> List[str]:
        """파일 목록 반환"""
        return [f.name for f in self.files]


def create_video_source(
    source_type: str,
    **kwargs
) -> BaseVideoSource:
    """
    비디오 소스 팩토리 함수
    
    Args:
        source_type: 소스 타입 (webcam, file, network, testset)
        **kwargs: 소스별 추가 인자
            - webcam: camera_id (int, 기본값 0)
            - file: file_path (str)
            - network: url (str)
            - testset: folder_path (str), loop (bool, 기본값 True)
    
    Returns:
        BaseVideoSource 인스턴스
    
    Examples:
        # 웹캠
        source = create_video_source("webcam", camera_id=0)
        
        # 파일
        source = create_video_source("file", file_path="testsets/violence.mp4")
        
        # 네트워크 카메라
        source = create_video_source("network", url="rtsp://192.168.1.100:554/stream")
        
        # 테스트셋 폴더
        source = create_video_source("testset", folder_path="testsets/")
    """
    if source_type == VideoSourceType.WEBCAM or source_type == "webcam":
        camera_id = kwargs.get("camera_id", 0)
        return WebcamVideoSource(camera_id)
    
    elif source_type == VideoSourceType.FILE or source_type == "file":
        file_path = kwargs.get("file_path")
        if not file_path:
            raise ValueError("file_path가 필요합니다")
        return FileVideoSource(file_path)
    
    elif source_type == VideoSourceType.NETWORK or source_type == "network":
        url = kwargs.get("url")
        if not url:
            raise ValueError("url이 필요합니다")
        return NetworkVideoSource(url)
    
    elif source_type == VideoSourceType.TESTSET or source_type == "testset":
        folder_path = kwargs.get("folder_path")
        if not folder_path:
            raise ValueError("folder_path가 필요합니다")
        loop = kwargs.get("loop", True)
        return TestsetVideoSource(folder_path, loop)
    
    else:
        raise ValueError(f"알 수 없는 소스 타입: {source_type}")


class VideoCaptureManager:
    """비디오 캡처 관리자 (레거시 호환 + 멀티 소스 지원)"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False
        self.lock = threading.Lock()
        
        # 새로운 비디오 소스 시스템
        self._video_source: Optional[BaseVideoSource] = None
    
    def set_source(self, source: BaseVideoSource):
        """비디오 소스 설정"""
        if self._video_source and self._video_source.is_opened:
            self._video_source.close()
        self._video_source = source
        print(f"✅ 비디오 소스 설정됨: {source.source_type}")
    
    def get_source(self) -> Optional[BaseVideoSource]:
        """현재 비디오 소스 반환"""
        return self._video_source
    
    def open(self) -> bool:
        """카메라/소스 열기"""
        # 새로운 소스가 설정되어 있으면 그것을 사용
        if self._video_source:
            result = self._video_source.open()
            self.is_opened = result
            return result
        
        # 레거시 모드: 웹캠
        with self.lock:
            if self.is_opened:
                return True
            
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                self.is_opened = True
                print(f"✅ 카메라 {self.camera_id} 열림")
                return True
            else:
                print(f"❌ 카메라 {self.camera_id}를 열 수 없습니다")
                return False
    
    def close(self):
        """카메라/소스 닫기"""
        # 새로운 소스 시스템
        if self._video_source:
            self._video_source.close()
            self.is_opened = False
            return
        
        # 레거시 모드
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.is_opened = False
                print("✅ 카메라 닫힘")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """단일 프레임 캡처"""
        # 새로운 소스 시스템
        if self._video_source:
            return self._video_source.capture_frame()
        
        # 레거시 모드
        with self.lock:
            if not self.is_opened or not self.cap:
                return None
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
    
    def capture_video_segment(
        self, 
        duration: float = 5.0, 
        target_fps: float = 2.0
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        지정된 시간 동안 비디오 세그먼트 캡처
        
        Args:
            duration: 캡처 시간 (초)
            target_fps: 타겟 FPS (다운샘플링용)
        
        Returns:
            (프레임 리스트, 타임스탬프 리스트)
        """
        # 새로운 소스 시스템
        if self._video_source:
            return self._video_source.capture_video_segment(duration, target_fps)
        
        # 레거시 모드
        frames = []
        timestamps = []
        
        with self.lock:
            if not self.is_opened or not self.cap:
                print("❌ 카메라가 열려있지 않습니다")
                return frames, timestamps
            
            # 원본 FPS 가져오기
            original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
            
            start_time = time.time()
            frame_count = 0
            
            print(f"📹 비디오 캡처 중 ({duration}초, {target_fps}fps)...")
            
            while (time.time() - start_time) < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 프레임 간격에 따라 샘플링
                if frame_count % frame_interval == 0:
                    timestamp = time.time() - start_time
                    frames.append(frame.copy())
                    timestamps.append(timestamp)
                
                frame_count += 1
            
            print(f"   ✅ {len(frames)}개 프레임 캡처 완료")
        
        return frames, timestamps


class SpeechDetector:
    """음성 감지 및 인식"""
    
    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.8):
        """
        Args:
            energy_threshold: 음성 감지 에너지 임계값
            pause_threshold: 문장 끝 판단 대기 시간 (초)
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("speech_recognition이 필요합니다: pip install SpeechRecognition")
        
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True
        
        # 마이크 초기화
        self.microphone = sr.Microphone()
        
        # 주변 소음 조정
        with self.microphone as source:
            print("🎤 주변 소음 조정 중...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"   ✅ 에너지 임계값: {self.recognizer.energy_threshold}")
    
    def listen_for_speech(self, timeout: float = None, phrase_time_limit: float = None) -> Tuple[Optional[sr.AudioData], bool]:
        """
        음성이 감지될 때까지 대기
        
        Args:
            timeout: 최대 대기 시간 (초)
            phrase_time_limit: 최대 발화 시간 (초)
        
        Returns:
            (AudioData 또는 None, 음성 감지 여부)
        """
        try:
            with self.microphone as source:
                print("👂 음성 대기 중...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                print("✅ 음성 감지됨!")
                return audio, True
        
        except sr.WaitTimeoutError:
            return None, False
        except Exception as e:
            print(f"❌ 음성 감지 오류: {e}")
            return None, False
    
    def recognize_speech(self, audio: sr.AudioData, language: str = "ko-KR") -> Optional[str]:
        """
        음성을 텍스트로 변환 (Google Speech Recognition)
        
        Args:
            audio: 음성 데이터
            language: 인식 언어
        
        Returns:
            인식된 텍스트 또는 None
        """
        try:
            text = self.recognizer.recognize_google(audio, language=language)
            print(f"📝 인식된 텍스트: {text}")
            return text
        except sr.UnknownValueError:
            print("⚠️  음성을 인식할 수 없습니다")
            return None
        except sr.RequestError as e:
            print(f"❌ 음성 인식 API 오류: {e}")
            return None
    
    def save_audio_to_wav(self, audio: sr.AudioData, output_path: str) -> str:
        """
        AudioData를 WAV 파일로 저장
        
        Args:
            audio: 음성 데이터
            output_path: 저장 경로
        
        Returns:
            저장된 파일 경로
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(audio.get_wav_data())
        
        return output_path


class IntegratedMultimodalSystem:
    """통합 멀티모달 시스템"""
    
    def __init__(
        self, 
        camera_id: int = 0,
        model: str = "gpt-4o-mini",
        downsampling_config: DownsamplingConfig = None,
        log_dir: str = None
    ):
        """
        통합 멀티모달 시스템 초기화
        
        Args:
            camera_id: 웹캠 카메라 ID
            model: OpenAI 모델명
            downsampling_config: 다운샘플링 설정
            log_dir: 로그 저장 디렉토리
        """
        self.camera_id = camera_id
        self.model = model
        self.downsampling_config = downsampling_config or DownsamplingConfig()
        
        # 컴포넌트 초기화
        self.video_manager = VideoCaptureManager(camera_id)
        self.downsampler = VideoDownsampler(self.downsampling_config)
        self.speech_detector = SpeechDetector()
        
        # 멀티모달 분석기
        if MULTIMODAL_ANALYZER_AVAILABLE:
            self.multimodal_analyzer = MultimodalAnalyzer(model=model)
        else:
            self.multimodal_analyzer = None
            print("⚠️  멀티모달 분석기를 사용할 수 없습니다")
        
        # 음성 특성 분석기
        if VOICE_CHARACTERISTICS_AVAILABLE:
            self.voice_characteristics_analyzer = VoiceCharacteristicsAnalyzer()
        else:
            self.voice_characteristics_analyzer = None
            print("⚠️  음성 특성 분석기를 사용할 수 없습니다")
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # 로그 설정
        self.log_dir = Path(log_dir) if log_dir else Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 녹음 파일 저장 디렉토리
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        
        # 콜백 함수
        self.on_result_callback: Optional[Callable[[Dict], None]] = None
        
        print("✅ 통합 멀티모달 시스템 초기화 완료")
    
    # ==================== 비디오 소스 설정 메서드 ====================
    
    def set_video_source(self, source: BaseVideoSource):
        """
        비디오 소스 직접 설정
        
        Args:
            source: BaseVideoSource 인스턴스
        """
        self.video_manager.set_source(source)
    
    def use_webcam(self, camera_id: int = 0):
        """
        웹캠을 비디오 소스로 사용
        
        Args:
            camera_id: 카메라 ID (기본값: 0)
        """
        source = WebcamVideoSource(camera_id)
        self.video_manager.set_source(source)
        print(f"📹 비디오 소스: 웹캠 (ID: {camera_id})")
    
    def use_file(self, file_path: str):
        """
        파일(이미지/비디오)을 비디오 소스로 사용
        
        Args:
            file_path: 파일 경로
        """
        source = FileVideoSource(file_path)
        self.video_manager.set_source(source)
        print(f"📹 비디오 소스: 파일 ({file_path})")
    
    def use_network_camera(self, url: str):
        """
        네트워크 카메라(RTSP/HTTP)를 비디오 소스로 사용
        
        Args:
            url: 카메라 URL
                - RTSP: rtsp://username:password@192.168.1.100:554/stream
                - HTTP: http://192.168.1.100:8080/video
        """
        source = NetworkVideoSource(url)
        self.video_manager.set_source(source)
        print(f"📹 비디오 소스: 네트워크 카메라 ({url})")
    
    def use_testset(self, folder_path: str, loop: bool = True):
        """
        테스트셋 폴더를 비디오 소스로 사용
        
        Args:
            folder_path: 테스트셋 폴더 경로
            loop: 파일 끝에 도달하면 처음부터 다시 시작할지 여부
        """
        source = TestsetVideoSource(folder_path, loop)
        self.video_manager.set_source(source)
        print(f"📹 비디오 소스: 테스트셋 ({folder_path})")
    
    def get_testset_files(self) -> List[str]:
        """테스트셋의 파일 목록 반환"""
        source = self.video_manager.get_source()
        if isinstance(source, TestsetVideoSource):
            return source.list_files()
        return []
    
    def select_testset_file(self, index_or_name) -> bool:
        """
        테스트셋에서 특정 파일 선택
        
        Args:
            index_or_name: 파일 인덱스(int) 또는 파일명(str)
        """
        source = self.video_manager.get_source()
        if not isinstance(source, TestsetVideoSource):
            print("❌ 현재 테스트셋 모드가 아닙니다")
            return False
        
        if isinstance(index_or_name, int):
            return source.select_file(index_or_name)
        else:
            return source.select_file_by_name(index_or_name)
    
    def get_video_source_info(self) -> Dict[str, Any]:
        """현재 비디오 소스 정보 반환"""
        source = self.video_manager.get_source()
        if source:
            return source.get_info()
        return {"source_type": "legacy_webcam", "camera_id": self.camera_id}
    
    # ==================== 테스트 모드 (음성 입력 없이 영상만 분석) ====================
    
    def analyze_video_only(self, text_input: str = None) -> Dict[str, Any]:
        """
        음성 입력 없이 영상만 분석 (테스트용)
        
        Args:
            text_input: 음성 대신 사용할 텍스트 (없으면 기본 텍스트 사용)
        
        Returns:
            분석 결과 딕셔너리
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "mode": "video_only",
            "text_input": text_input or "(테스트 모드 - 음성 입력 없음)",
            "video_analysis": None,
            "multimodal_analysis": None,
            "error": None
        }
        
        try:
            # 1. 비디오 소스 열기
            if not self.video_manager.open():
                result["error"] = "비디오 소스를 열 수 없습니다"
                return result
            
            print("\n" + "=" * 60)
            print("🎬 영상만 분석 모드 (테스트용)")
            print("=" * 60)
            
            # 2. 영상 캡처
            print("\n📹 영상 캡처 중...")
            frames, timestamps = self._capture_and_process_video()
            
            if not frames:
                result["error"] = "프레임을 캡처할 수 없습니다"
                return result
            
            result["video_analysis"] = {
                "frame_count": len(frames),
                "timestamps": timestamps
            }
            
            # 3. 멀티모달 분석 (영상 + 텍스트 입력)
            if self.multimodal_analyzer and frames:
                print("\n🔍 영상 분석 수행 중...")
                
                # 대표 프레임 선택 (중간 프레임)
                representative_frame = frames[len(frames) // 2]
                
                # 분석할 텍스트 (기본값: 영상 분석 요청)
                # 기존 시스템 프롬프트는 "음성 입력"을 기대하므로, 
                # 테스트 모드에서는 상황 설명 요청으로 대체
                default_text = "현재 상황을 분석해 주세요. 위험하거나 긴급한 상황인지 판단해 주세요."
                analysis_text = text_input or default_text
                
                multimodal_result = self.multimodal_analyzer.analyze_with_image(
                    audio_text=analysis_text,
                    image_source=representative_frame,
                    additional_context="[테스트 모드] 실제 음성 입력 없이 영상만 분석. 영상에서 보이는 상황을 객관적으로 분석하세요."
                )
                
                result["multimodal_analysis"] = multimodal_result
            
            result["success"] = True
            
            # 로그 저장
            self._save_result_log(result)
            
            return result
        
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 분석 오류: {e}")
            return result
    
    def analyze_testset_all(self, text_input: str = None) -> List[Dict[str, Any]]:
        """
        테스트셋의 모든 파일을 순차적으로 분석
        
        Args:
            text_input: 각 파일 분석 시 사용할 텍스트
        
        Returns:
            각 파일의 분석 결과 리스트
        """
        source = self.video_manager.get_source()
        if not isinstance(source, TestsetVideoSource):
            print("❌ 현재 테스트셋 모드가 아닙니다. use_testset()을 먼저 호출하세요.")
            return []
        
        results = []
        files = source.list_files()
        
        print(f"\n📁 테스트셋 전체 분석 시작 ({len(files)}개 파일)")
        print("=" * 60)
        
        for i, filename in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] 📂 {filename}")
            print("-" * 40)
            
            # 파일 선택
            if not source.select_file(i):
                print(f"   ❌ 파일 열기 실패")
                continue
            
            # 분석
            result = self.analyze_video_only(text_input)
            result["file_index"] = i
            result["filename"] = filename
            
            results.append(result)
            
            # 결과 요약
            if result.get("success"):
                analysis = result.get("multimodal_analysis", {})
                print(f"   ✅ 성공")
                print(f"      상황: {analysis.get('situation_type', 'N/A')}")
                print(f"      위급도: {analysis.get('urgency', 'N/A')}")
                print(f"      우선순위: {analysis.get('priority', 'N/A')}")
            else:
                print(f"   ❌ 실패: {result.get('error', '알 수 없는 오류')}")
        
        print("\n" + "=" * 60)
        print(f"📊 테스트셋 분석 완료")
        print(f"   총 {len(files)}개 파일, 성공 {sum(1 for r in results if r.get('success'))}개")
        print("=" * 60)
        
        return results
    
    # ==================== 기존 메서드 ====================
    
    def analyze_once(self, phrase_time_limit: float = 30.0) -> Dict[str, Any]:
        """
        단발성 분석 수행
        음성이 감지되면 음성 + 영상을 함께 분석
        
        Args:
            phrase_time_limit: 최대 발화 시간 (초)
        
        Returns:
            분석 결과 딕셔너리
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "speech_detected": False,
            "transcribed_text": None,
            "voice_characteristics": None,
            "video_analysis": None,
            "multimodal_analysis": None,
            "error": None
        }
        
        try:
            # 1. 카메라 열기 (백그라운드에서 대기)
            if not self.video_manager.open():
                result["error"] = "카메라를 열 수 없습니다"
                return result
            
            # 2. 음성 감지 대기 (음성 감지 전까지는 음성만 대기)
            print("\n" + "=" * 60)
            print("🎙️  음성 감지 대기 중... (말씀해 주세요)")
            print("=" * 60)
            
            audio, detected = self.speech_detector.listen_for_speech(
                phrase_time_limit=phrase_time_limit
            )
            
            if not detected or audio is None:
                result["error"] = "음성이 감지되지 않았습니다"
                return result
            
            result["speech_detected"] = True
            
            # 3. 음성 감지됨! 동시에 영상 캡처 시작
            print("\n🚀 음성 감지! 멀티모달 분석 시작...")
            
            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                # 3-1. 음성 인식 (텍스트 변환)
                futures["speech_recognition"] = executor.submit(
                    self.speech_detector.recognize_speech, 
                    audio
                )
                
                # 3-2. 음성 특성 분석 (오디오 파일 저장 후 분석)
                audio_path = self.recordings_dir / f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                self.speech_detector.save_audio_to_wav(audio, str(audio_path))
                
                if self.voice_characteristics_analyzer:
                    futures["voice_characteristics"] = executor.submit(
                        self._analyze_voice_characteristics,
                        str(audio_path)
                    )
                
                # 3-3. 영상 캡처 및 다운샘플링
                futures["video_capture"] = executor.submit(
                    self._capture_and_process_video
                )
                
                # 결과 수집
                transcribed_text = None
                voice_features = None
                video_frames = []
                
                for name, future in futures.items():
                    try:
                        if name == "speech_recognition":
                            transcribed_text = future.result(timeout=30)
                            result["transcribed_text"] = transcribed_text
                        
                        elif name == "voice_characteristics":
                            voice_features = future.result(timeout=30)
                            result["voice_characteristics"] = voice_features
                        
                        elif name == "video_capture":
                            video_frames, timestamps = future.result(timeout=30)
                            result["video_analysis"] = {
                                "frame_count": len(video_frames),
                                "timestamps": timestamps
                            }
                    
                    except Exception as e:
                        print(f"   ⚠️  {name} 오류: {e}")
            
            # 4. 멀티모달 분석 (음성 텍스트 + 영상)
            if transcribed_text and video_frames and self.multimodal_analyzer:
                print("\n🔍 멀티모달 분석 수행 중...")
                
                # 대표 프레임 선택 (중간 프레임)
                representative_frame = video_frames[len(video_frames) // 2] if video_frames else None
                
                if representative_frame is not None:
                    # 음성 특성 정보를 추가 컨텍스트로 전달
                    additional_context = None
                    if voice_features:
                        additional_context = self._format_voice_features_context(voice_features)
                    
                    multimodal_result = self.multimodal_analyzer.analyze_with_image(
                        audio_text=transcribed_text,
                        image_source=representative_frame,
                        additional_context=additional_context,
                        audio_file_path=str(audio_path)
                    )
                    
                    result["multimodal_analysis"] = multimodal_result
            
            # 5. 성공 표시
            result["success"] = True
            
            # 6. 결과 로그 저장
            self._save_result_log(result)
            
            # 7. 임시 오디오 파일 삭제 (선택적)
            # os.remove(audio_path)
            
            return result
        
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 분석 오류: {e}")
            return result
        
        finally:
            # 카메라 닫지 않음 (연속 모니터링을 위해)
            pass
    
    def _analyze_voice_characteristics(self, audio_path: str) -> Dict[str, Any]:
        """음성 특성 분석"""
        print("   🎤 음성 특성 분석 중...")
        
        try:
            features = self.voice_characteristics_analyzer.extract_features(audio_path)
            
            # 긴급도 점수 계산
            emergency_indicators = self._calculate_voice_emergency_indicators(features)
            
            print("   ✅ 음성 특성 분석 완료")
            
            return {
                "features": features,
                "emergency_indicators": emergency_indicators
            }
        
        except Exception as e:
            print(f"   ⚠️  음성 특성 분석 실패: {e}")
            return None
    
    def _calculate_voice_emergency_indicators(self, features: Dict) -> Dict[str, Any]:
        """음성 특성에서 긴급 지표 계산"""
        indicators = {
            "high_pitch": False,
            "high_energy": False,
            "fast_speech": False,
            "voice_trembling": False,
            "overall_score": 0.0
        }
        
        if not features:
            return indicators
        
        score = 0.0
        
        # 피치 분석 (높은 피치 = 긴장/공포)
        pitch = features.get("pitch", {})
        if pitch.get("mean", 0) > 250:  # 평균 피치가 높으면
            indicators["high_pitch"] = True
            score += 0.25
        if pitch.get("std", 0) > 50:  # 피치 변동이 크면 (떨림)
            indicators["voice_trembling"] = True
            score += 0.25
        
        # 에너지 분석 (높은 에너지 = 소리 지름)
        energy = features.get("energy", {})
        if energy.get("max", 0) > 0.3:  # 최대 에너지가 높으면
            indicators["high_energy"] = True
            score += 0.25
        
        # 말 속도 분석 (빠른 말 = 급박함)
        speech_rate = features.get("speech_rate", {})
        if speech_rate.get("estimated_syllables_per_second", 0) > 5:  # 초당 5음절 이상
            indicators["fast_speech"] = True
            score += 0.25
        
        indicators["overall_score"] = min(score, 1.0)
        
        return indicators
    
    def _capture_and_process_video(self) -> Tuple[List[np.ndarray], List[float]]:
        """비디오 캡처 및 다운샘플링"""
        print("   📹 비디오 캡처 중...")
        
        # 비디오 세그먼트 캡처
        frames, timestamps = self.video_manager.capture_video_segment(
            duration=self.downsampling_config.video_capture_duration,
            target_fps=self.downsampling_config.video_fps
        )
        
        # 다운샘플링 적용
        frames, timestamps = self.downsampler.downsample_video_frames(frames, timestamps)
        
        print(f"   ✅ 비디오 처리 완료 ({len(frames)} 프레임, 다운샘플링 적용)")
        
        return frames, timestamps
    
    def _format_voice_features_context(self, voice_features: Dict) -> str:
        """음성 특성을 컨텍스트 문자열로 포맷"""
        if not voice_features:
            return ""
        
        indicators = voice_features.get("emergency_indicators", {})
        
        context_parts = ["**음성 특성 분석 결과:**"]
        
        if indicators.get("high_pitch"):
            context_parts.append("- 높은 피치 감지 (긴장/공포 가능성)")
        
        if indicators.get("high_energy"):
            context_parts.append("- 높은 에너지 감지 (소리 지름/흥분)")
        
        if indicators.get("fast_speech"):
            context_parts.append("- 빠른 말 속도 (급박함)")
        
        if indicators.get("voice_trembling"):
            context_parts.append("- 음성 떨림 감지 (불안/공포)")
        
        score = indicators.get("overall_score", 0)
        context_parts.append(f"- 음성 긴급도 점수: {score:.0%}")
        
        return "\n".join(context_parts)
    
    def _save_result_log(self, result: Dict):
        """결과 로그 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"integrated_analysis_{timestamp}.json"
        
        # numpy array 등 직렬화 불가능한 객체 처리
        serializable_result = self._make_serializable(result)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 로그 저장: {log_file}")
    
    def _make_serializable(self, obj):
        """객체를 JSON 직렬화 가능하게 변환"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape}>"
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    def start_monitoring(
        self, 
        on_result: Callable[[Dict], None] = None,
        max_iterations: int = None
    ):
        """
        연속 모니터링 시작
        
        Args:
            on_result: 결과 콜백 함수
            max_iterations: 최대 반복 횟수 (None이면 무한)
        """
        self.on_result_callback = on_result
        self.is_monitoring = True
        
        # 카메라 미리 열기
        self.video_manager.open()
        
        print("\n" + "=" * 60)
        print("🔄 연속 모니터링 시작")
        print("   - 음성이 감지되면 자동으로 영상 분석")
        print("   - Ctrl+C로 종료")
        print("=" * 60)
        
        iteration = 0
        
        try:
            while self.is_monitoring:
                iteration += 1
                
                if max_iterations and iteration > max_iterations:
                    print(f"\n✅ {max_iterations}회 완료!")
                    break
                
                print(f"\n[{iteration}회차] {datetime.now().strftime('%H:%M:%S')}")
                
                # 분석 수행
                result = self.analyze_once()
                
                # 콜백 호출
                if self.on_result_callback and result.get("success"):
                    self.on_result_callback(result)
                
                # 결과 출력
                self._print_result_summary(result)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  모니터링 중지됨 (Ctrl+C)")
        
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        self.video_manager.close()
        print("✅ 모니터링 종료")
    
    def _print_result_summary(self, result: Dict):
        """결과 요약 출력"""
        print("\n" + "-" * 40)
        
        if not result.get("success"):
            print(f"❌ 분석 실패: {result.get('error', '알 수 없는 오류')}")
            return
        
        # 음성 텍스트
        text = result.get("transcribed_text", "")
        if text:
            print(f"📝 음성: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        # 음성 특성
        voice = result.get("voice_characteristics")
        if voice:
            indicators = voice.get("emergency_indicators", {})
            score = indicators.get("overall_score", 0)
            print(f"🎤 음성 긴급도: {score:.0%}")
        
        # 멀티모달 분석
        analysis = result.get("multimodal_analysis")
        if analysis:
            print(f"🔍 상황: {analysis.get('situation_type', 'N/A')}")
            print(f"⚡ 위급도: {analysis.get('urgency', 'N/A')}")
            print(f"🚨 우선순위: {analysis.get('priority', 'N/A')}")
            
            if analysis.get("is_emergency"):
                print(f"⚠️  긴급 상황: {analysis.get('emergency_reason', '')}")
        
        print("-" * 40)


# 테스트 및 실행
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 통합 멀티모달 시스템 테스트")
    print("=" * 70)
    
    # 다운샘플링 설정 (성능 최적화)
    config = DownsamplingConfig(
        max_image_size=640,
        jpeg_quality=75,
        video_fps=2.0,
        max_video_frames=10,
        video_resolution_scale=0.5,
        video_capture_duration=5.0
    )
    
    # 시스템 초기화
    system = IntegratedMultimodalSystem(
        camera_id=0,
        model="gpt-4o-mini",
        downsampling_config=config
    )
    
    print("\n테스트 모드 선택:")
    print("1. 단발성 분석 (한 번만)")
    print("2. 연속 모니터링 (무한)")
    print("3. 연속 모니터링 (3회)")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == "1":
        print("\n단발성 분석 시작...")
        result = system.analyze_once()
        print(f"\n최종 결과: {json.dumps(system._make_serializable(result), ensure_ascii=False, indent=2)}")
    
    elif choice == "2":
        system.start_monitoring()
    
    elif choice == "3":
        system.start_monitoring(max_iterations=3)
    
    else:
        print("잘못된 선택입니다.")
