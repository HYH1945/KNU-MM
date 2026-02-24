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

import json
import logging
import shutil
import subprocess
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# config 로드
try:
    from core.config_manager import get_config
except ImportError:
    try:
        from config_manager import get_config
    except ImportError:
        def get_config(key, default=None):
            return default

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False

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

try:
    from core.multimodal_analyzer import MultimodalAnalyzer
    MULTIMODAL_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from multimodal_analyzer import MultimodalAnalyzer
        MULTIMODAL_ANALYZER_AVAILABLE = True
    except ImportError:
        MULTIMODAL_ANALYZER_AVAILABLE = False

try:
    from core.sound_event_detector import SoundEventDetector
    SOUND_EVENT_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from sound_event_detector import SoundEventDetector
        SOUND_EVENT_DETECTOR_AVAILABLE = True
    except ImportError:
        SOUND_EVENT_DETECTOR_AVAILABLE = False


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
                return True
            else:
                return False
    
    def close(self):
        """웹캠 닫기"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.is_opened = False
    
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
                return frames, timestamps
            
            original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
            
            start_time = time.time()
            frame_count = 0
            
            while (time.time() - start_time) < duration:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = time.time() - start_time
                    frames.append(frame.copy())
                    timestamps.append(timestamp)
                
                frame_count += 1
        
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
        # URL 검증 (허용된 프로토콜만)
        allowed_protocols = ('rtsp://', 'http://', 'https://')
        if not any(url.lower().startswith(p) for p in allowed_protocols):
            raise ValueError(f"허용되지 않은 프로토콜입니다. 허용: {allowed_protocols}")
        
        # 로컬호스트/내부 IP 차단 (선택적 - SSRF 방지)
        # from urllib.parse import urlparse
        # parsed = urlparse(url)
        # if parsed.hostname in ('localhost', '127.0.0.1', '0.0.0.0'):
        #     raise ValueError("로컬 주소는 허용되지 않습니다")
        
        self.url = url
        self.cap = None
        self.source_type = VideoSourceType.NETWORK
    
    def open(self) -> bool:
        """네트워크 카메라 연결"""
        with self.lock:
            if self.is_opened:
                return True
            
            self.cap = cv2.VideoCapture(self.url)
            
            # 버퍼 크기 줄이기 (지연 감소)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_opened = True
                return True
            else:
                return False
    
    def close(self):
        """네트워크 카메라 연결 종료"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.is_opened = False
    
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
    """파일 기반 비디오 소스 (이미지/비디오/오디오 파일)"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: 이미지/비디오/오디오 파일 경로
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.cap = None
        self.is_video = False
        self.is_image = False
        self.is_audio = False
        self.image = None
        self.source_type = VideoSourceType.FILE
        
        # 파일 타입 확인
        self._detect_file_type()
    
    def _detect_file_type(self):
        """파일 타입 감지"""
        suffix = self.file_path.suffix.lower()
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        
        if suffix in video_extensions:
            self.is_video = True
        elif suffix in image_extensions:
            self.is_image = True
        elif suffix in audio_extensions:
            self.is_audio = True
    
    def open(self) -> bool:
        """파일 열기"""
        with self.lock:
            if self.is_opened:
                return True
            
            if not self.file_path.exists():
                return False
            
            if self.is_video:
                self.cap = cv2.VideoCapture(str(self.file_path))
                if self.cap.isOpened():
                    self.is_opened = True
                    return True
                else:
                    return False
            
            elif self.is_image:
                self.image = cv2.imread(str(self.file_path))
                if self.image is not None:
                    self.is_opened = True
                    return True
                else:
                    return False
            elif self.is_audio:
                self.is_opened = True
                return True
            
            else:
                return False
    
    def close(self):
        """파일 닫기"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.image = None
            self.is_opened = False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """프레임/이미지 가져오기"""
        with self.lock:
            if not self.is_opened:
                return None
            
            if self.is_image:
                return self.image.copy() if self.image is not None else None
            elif self.is_audio:
                return None
            
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
                return frames, timestamps
            
            if self.is_image:
                # 이미지인 경우 같은 이미지를 여러 번 반환
                num_frames = int(duration * target_fps)
                
                for i in range(num_frames):
                    frames.append(self.image.copy())
                    timestamps.append(i / target_fps)
                
                return frames, timestamps
            elif self.is_audio:
                return [], []
            
            elif self.is_video and self.cap:
                original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_frames / original_fps
                
                # 요청된 시간이 비디오 길이보다 길면 비디오 길이로 제한
                actual_duration = min(duration, video_duration)
                
                frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
                frame_count = 0
                
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
        
        return frames, timestamps
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["file_path"] = str(self.file_path)
        info["is_video"] = self.is_video
        info["is_image"] = self.is_image
        info["is_audio"] = self.is_audio
        
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
    """테스트셋 폴더 비디오 소스 (폴더 내 이미지/비디오/오디오 순차 선택)"""
    
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
        
        # 초기화 시 파일 스캔 (open 전에도 파일 목록 확인 가능)
        self._scan_files()
    
    def _scan_files(self):
        """폴더 내 미디어 파일 스캔"""
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            print(f"⚠️  폴더가 존재하지 않거나 유효하지 않음: {self.folder_path}")
            self.files = []
            return
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        all_extensions = video_extensions | image_extensions | audio_extensions
        
        self.files = sorted([
            f for f in self.folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in all_extensions
        ])
    
    def open(self) -> bool:
        """테스트셋 폴더 열기"""
        with self.lock:
            if self.is_opened:
                return True
            
            # 파일이 아직 스캔되지 않았으면 다시 스캔
            if not self.files:
                self._scan_files()
            
            if not self.folder_path.exists():
                return False
            
            if not self.folder_path.is_dir():
                return False
            
            self._scan_files()
            
            if not self.files:
                return False
            
            # 첫 번째 파일 열기
            self.current_index = 0
            self.current_source = FileVideoSource(str(self.files[0]))
            
            if self.current_source.open():
                self.is_opened = True
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
    
    def _next_file(self) -> bool:
        """다음 파일로 이동"""
        if self.current_source:
            self.current_source.close()
        
        self.current_index += 1
        
        if self.current_index >= len(self.files):
            if self.loop:
                self.current_index = 0
            else:
                return False
        
        self.current_source = FileVideoSource(str(self.files[self.current_index]))
        return self.current_source.open()
    
    def select_file(self, index: int) -> bool:
        """특정 인덱스의 파일 선택"""
        with self.lock:
            if index < 0 or index >= len(self.files):
                return False
            
            if self.current_source:
                self.current_source.close()
            
            self.current_index = index
            self.current_source = FileVideoSource(str(self.files[index]))
            success = self.current_source.open()
            
            return success
    
    def select_file_by_name(self, filename: str) -> bool:
        """파일 이름으로 선택"""
        for i, f in enumerate(self.files):
            if f.name == filename or f.stem == filename:
                return self.select_file(i)
        
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
                return True
            else:
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
                return frames, timestamps
            
            # 원본 FPS 가져오기
            original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1
            
            start_time = time.time()
            frame_count = 0
            
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
        
        return frames, timestamps


class SpeechDetector:
    """음성 감지 및 인식"""
    
    def __init__(self, energy_threshold: int = 400, pause_threshold: float = 3.0, dynamic_threshold: bool = False):
        """
        Args:
            energy_threshold: 음성 감지 에너지 임계값 (낮을수록 민감함) - 기본값 400
            pause_threshold: 문장 끝 판단 대기 시간 (초) - 기본값 3.0 (자연스러운 대화 흐름)
                           3초 침묵 후 문장 끝으로 판단 → 자연스러운 대화 포함
            dynamic_threshold: 동적 에너지 임계값 조정 여부 - False=고정(스피커 소리용), True=자동(실시간 조정)
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("speech_recognition이 필요합니다: pip install SpeechRecognition")
        
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = pause_threshold
        # dynamic_energy_threshold 설정
        # False: 고정 임계값 (스피커/유튜브 소리 인식에 더 좋음)
        # True: 동적 조정 (실시간 마이크 입력에 좋음)
        self.recognizer.dynamic_energy_threshold = dynamic_threshold
        
        # 마이크 초기화
        self.microphone = sr.Microphone()
        
        # 주변 소음 조정 후 에너지 임계값 설정
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            # 조정된 값을 그대로 사용 (추가 배수 없음)
            self.recognizer.energy_threshold = max(energy_threshold, self.recognizer.energy_threshold)
        
        # 백그라운드 음성 인식용 저장소
        self._bg_audio_queue = None
        self._is_listening = False
    
    def listen_and_recognize(self, timeout: float = None, phrase_time_limit: float = None, language: str = "ko-KR") -> Tuple[Optional[str], Optional[Any]]:
        """
        음성을 듣고 바로 인식 (감지 + 인식 통합)
        pause_threshold 내에서 수집한 모든 음성을 인식
        
        Args:
            timeout: 최대 대기 시간 (초) - None이면 무한 대기
            phrase_time_limit: 최대 발화 시간 (초) - None이면 pause_threshold 사용
            language: 인식 언어
        
        Returns:
            (인식된 텍스트 또는 None, AudioData 또는 None)
        """
        try:
            with self.microphone as source:
                # pause_threshold 시간 동안 계속 수집하도록 설정
                # phrase_time_limit=None이면, pause_threshold 시간만큼 대기 후 인식
                # 예: pause_threshold=10초 → 10초 동안 계속 음성 수집 후 인식
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                    # phrase_time_limit=None이 핵심: pause_threshold까지 기다린 후 인식
                )
                
                # 오디오 길이 확인 (너무 짧으면 노이즈)
                audio_data = audio.get_raw_data()
                duration = len(audio_data) / (audio.sample_rate * audio.sample_width)
                
                if duration < 0.3:  # 0.5초 → 0.3초로 완화
                    return None, None
                
                # 바로 텍스트 인식
                try:
                    text = self.recognizer.recognize_google(audio, language=language)
                    return text, audio
                except sr.UnknownValueError:
                    # 음성/소리는 감지됐으나 텍스트 인식 불가 -> 비음성 이벤트 검출을 위해 오디오 반환
                    return None, audio
                except sr.RequestError as e:
                    logger.warning("Speech recognition request failed: %s", e)
                    return None, None
        
        except sr.WaitTimeoutError:
            return None, None
        except Exception as e:
            logger.debug("listen_and_recognize failed: %s", e)
            return None, None
    
    def start_background_listening(self, language: str = "ko-KR"):
        """
        백그라운드에서 계속 음성을 감지하고 인식
        루프가 멈추지 않고 음성이 감지되면 큐에 추가
        
        Args:
            language: 인식 언어
        """
        import queue
        import threading
        
        if self._is_listening:
            return  # 이미 실행 중
        
        self._is_listening = True
        self._bg_audio_queue = queue.Queue()
        
        def background_worker():
            """백그라운드 음성 인식 워커"""
            
            while self._is_listening:
                try:
                    with self.microphone as source:
                        # 음성 감지
                        audio = self.recognizer.listen(source, timeout=None)
                        
                        # 오디오 길이 확인
                        audio_data = audio.get_raw_data()
                        duration = len(audio_data) / (audio.sample_rate * audio.sample_width)
                        
                        if duration < 0.5:  # 0.5초 이상만 인식 시도
                            continue
                        
                        # 텍스트 인식
                        try:
                            text = self.recognizer.recognize_google(audio, language=language)
                            print(f'\n음성 인식됨: "{text}"')
                            # 큐에 추가 (메인 루프에서 꺼낼 수 있음)
                            self._bg_audio_queue.put((text, audio))
                        except sr.UnknownValueError:
                            # 비음성/짧은 발화도 사운드 이벤트 감지 경로로 전달
                            self._bg_audio_queue.put((None, audio))
                        except sr.RequestError as e:
                            logger.warning("Background speech recognition request failed: %s", e)
                
                except Exception as e:
                    logger.debug("Background speech worker error: %s", e)
        
        # 백그라운드 스레드 시작
        bg_thread = threading.Thread(target=background_worker, daemon=True)
        bg_thread.start()
    
    def get_recognized_speech(self):
        """
        백그라운드에서 인식된 음성 가져오기 (논블로킹)
        
        Returns:
            (텍스트, 오디오) 또는 (None, None) - 큐가 비어있으면 None 반환
        """
        if not self._bg_audio_queue:
            return None, None
        
        try:
            return self._bg_audio_queue.get_nowait()
        except queue.Empty:
            return None, None
    
    def stop_background_listening(self):
        """백그라운드 리스닝 중지"""
        self._is_listening = False
    
    def listen_continuous(self, duration: float = 5.0, language: str = "ko-KR") -> Tuple[Optional[str], Optional[Any]]:
        """
        지정된 시간(초) 동안 연속으로 음성을 수집하고 인식
        여러 문장을 한번에 모아서 대화 맥락을 파악할 수 있음
        
        Args:
            duration: 음성 수집 시간 (초) - 기본값 5초
            language: 인식 언어
        
        Returns:
            (인식된 전체 텍스트 또는 None, AudioData 또는 None)
        """
        try:
            with self.microphone as source:
                # 음성 감지 및 수집 (최대 duration 초)
                audio = self.recognizer.listen(
                    source,
                    timeout=None,  # 음성이 들어올 때까지 무한 대기
                    phrase_time_limit=duration  # 최대 duration초까지 수집
                )
                
                # 오디오 길이 확인
                audio_data = audio.get_raw_data()
                duration_actual = len(audio_data) / (audio.sample_rate * audio.sample_width)
                
                if duration_actual < 0.5:  # 0.5초 미만이면 무시
                    return None, None
                
                # 텍스트 인식
                try:
                    text = self.recognizer.recognize_google(audio, language=language)
                    return text, audio
                except sr.UnknownValueError:
                    return None, None
                except sr.RequestError as e:
                    logger.warning("Continuous speech recognition request failed: %s", e)
                    return None, None
        
        except sr.WaitTimeoutError:
            return None, None
        except Exception as e:
            logger.debug("listen_continuous failed: %s", e)
            return None, None
    
    def recognize_speech(self, audio: Any, language: str = "ko-KR") -> Optional[str]:
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
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.warning("Speech recognition request failed: %s", e)
            return None
    
    def save_audio_to_wav(self, audio: Any, output_path: str) -> str:
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
        log_dir: str = None,
        energy_threshold: int = 400,
        pause_threshold: float = 3.0,
        dynamic_threshold: bool = False,
        enable_speech: bool = True,
    ):
        """
        통합 멀티모달 시스템 초기화
        
        Args:
            camera_id: 웹캠 카메라 ID
            model: OpenAI 모델명
            downsampling_config: 다운샘플링 설정
            log_dir: 로그 저장 디렉토리
            energy_threshold: 음성 감지 에너지 임계값 (낮을수록 민감함)
            pause_threshold: 문장 끝 침묵 판단 시간 (초)
            dynamic_threshold: 동적 에너지 임계값 여부 (False=고정/스피커소리용, True=자동/마이크용)
            enable_speech: 음성 감지기 초기화 여부 (False면 영상 전용 모드)
        """
        self.camera_id = camera_id
        self.model = model
        self.downsampling_config = downsampling_config or DownsamplingConfig()
        
        # 분석 설정 로드
        self.analysis_config = get_config('analysis', default={}) or {}
        self.voice_analysis_config = get_config('voice_analysis', default={}) or {}
        self.voice_characteristics_config = get_config('voice_characteristics', default={}) or {}
        self.streaming_config = get_config('streaming', default={}) or {}
        self.use_voice_characteristics, self.use_streaming = self._resolve_feature_toggles()
        self.voice_thresholds = self._resolve_voice_threshold_config()
        self.sound_event_config = get_config('sound_event', default={}) or {}
        self.use_sound_event_detection = self.sound_event_config.get('enabled', True)
        
        # 컴포넌트 초기화
        self.video_manager = VideoCaptureManager(camera_id)
        self.downsampler = VideoDownsampler(self.downsampling_config)
        self.enable_speech = enable_speech
        self.speech_detector = self._init_speech_detector(
            energy_threshold=energy_threshold,
            pause_threshold=pause_threshold,
            dynamic_threshold=dynamic_threshold,
            enable_speech=enable_speech,
        )
        
        # 멀티모달 분석기
        if MULTIMODAL_ANALYZER_AVAILABLE:
            try:
                self.multimodal_analyzer = MultimodalAnalyzer(model=model)
            except Exception as e:
                logger.warning("Multimodal analyzer initialization failed: %s", e)
                self.multimodal_analyzer = None
        else:
            self.multimodal_analyzer = None
        
        # 음성 특성 분석기 (config에서 비활성화 가능)
        if VOICE_CHARACTERISTICS_AVAILABLE and self.use_voice_characteristics:
            self.voice_characteristics_analyzer = VoiceCharacteristicsAnalyzer()
        else:
            self.voice_characteristics_analyzer = None

        # 비음성 이벤트 감지기 (YAMNet)
        self.sound_event_detector = None
        if SOUND_EVENT_DETECTOR_AVAILABLE and self.use_sound_event_detection:
            self.sound_event_detector = SoundEventDetector(
                model_url=self.sound_event_config.get('model_url', 'https://tfhub.dev/google/yamnet/1'),
                min_confidence=self.sound_event_config.get('min_confidence', 0.12),
                trigger_threshold=self.sound_event_config.get('trigger_threshold', 0.25),
                top_k=self.sound_event_config.get('top_k', 5),
                emergency_keywords=self.sound_event_config.get('emergency_keywords', []),
            )
            if not getattr(self.sound_event_detector, "enabled", False):
                reason = getattr(self.sound_event_detector, "last_error", "") or "unknown"
                logger.warning(
                    "SoundEventDetector is disabled. "
                    "Install TensorFlow/TensorFlow Hub and verify YAMNet model load. reason=%s",
                    reason,
                )
        elif self.use_sound_event_detection:
            logger.warning("SoundEventDetector import unavailable. YAMNet path is disabled.")
        
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
        
        # 디스플레이 설정
        self.use_opencv_display = False
        self.opencv_display = None
        self.use_web_dashboard = False

    def _resolve_feature_toggles(self) -> Tuple[bool, bool]:
        """레거시/신규 설정 키를 모두 지원해 기능 토글 결정"""
        analysis_voice = self.analysis_config.get('voice_characteristics')
        legacy_voice = self.voice_characteristics_config.get('enabled')

        if analysis_voice is None and legacy_voice is not None:
            logger.warning("Deprecated config key `voice_characteristics.enabled` in use. Prefer `analysis.voice_characteristics`.")
        if analysis_voice is not None and legacy_voice is not None and analysis_voice != legacy_voice:
            logger.warning("Config mismatch: `analysis.voice_characteristics` overrides `voice_characteristics.enabled`.")

        use_voice_characteristics = analysis_voice if analysis_voice is not None else legacy_voice
        if use_voice_characteristics is None:
            use_voice_characteristics = True

        analysis_streaming = self.analysis_config.get('streaming')
        legacy_streaming = self.streaming_config.get('enabled')
        if analysis_streaming is None and legacy_streaming is not None:
            logger.warning("Deprecated config key `streaming.enabled` in use. Prefer `analysis.streaming`.")
        if analysis_streaming is not None and legacy_streaming is not None and analysis_streaming != legacy_streaming:
            logger.warning("Config mismatch: `analysis.streaming` overrides `streaming.enabled`.")

        use_streaming = analysis_streaming if analysis_streaming is not None else legacy_streaming
        if use_streaming is None:
            use_streaming = False

        return bool(use_voice_characteristics), bool(use_streaming)

    def _resolve_voice_threshold_config(self) -> Dict[str, Any]:
        """임계값 설정은 `voice_analysis` 루트 키를 기본으로 사용하고 하위 호환 유지"""
        legacy_voice_cfg = self.analysis_config.get('voice_analysis', {}) or {}
        if legacy_voice_cfg and self.voice_analysis_config:
            logger.warning("Config mismatch: both `voice_analysis` and `analysis.voice_analysis` found. Using `voice_analysis`.")
        return self.voice_analysis_config or legacy_voice_cfg

    def _init_speech_detector(
        self,
        energy_threshold: int,
        pause_threshold: float,
        dynamic_threshold: bool,
        enable_speech: bool,
    ) -> Optional[SpeechDetector]:
        """음성 감지기는 필요할 때만 초기화하여 영상-only 경로를 분리"""
        if not enable_speech:
            return None

        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.warning("SpeechRecognition is not available; speech-triggered modes are disabled.")
            return None

        try:
            return SpeechDetector(
                energy_threshold=energy_threshold,
                pause_threshold=pause_threshold,
                dynamic_threshold=dynamic_threshold,
            )
        except Exception as e:
            logger.warning("SpeechDetector initialization failed: %s", e)
            return None

    def _require_speech_detector(self) -> bool:
        """음성 기반 모드에서 필수인 감지기 존재 여부 확인"""
        if self.speech_detector is not None:
            return True
        logger.warning("Speech detector is unavailable. Install SpeechRecognition/PyAudio or run a video-only mode.")
        return False
    
    # ==================== 디스플레이 설정 메서드 ====================
    
    def enable_opencv_display(self, enable: bool = True):
        """OpenCV 실시간 디스플레이 활성화/비활성화"""
        self.use_opencv_display = enable
        if enable:
            try:
                from core.display_manager import OpenCVDisplay
                self.opencv_display = OpenCVDisplay()
            except ImportError:
                print("⚠️ OpenCV 디스플레이를 로드할 수 없습니다")
                self.use_opencv_display = False
    
    def enable_web_dashboard(self, enable: bool = True):
        """웹 대시보드 연동 활성화/비활성화"""
        self.use_web_dashboard = enable
    
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
    
    def use_file(self, file_path: str):
        """
        파일(이미지/비디오)을 비디오 소스로 사용
        
        Args:
            file_path: 파일 경로
        """
        source = FileVideoSource(file_path)
        self.video_manager.set_source(source)
    
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
    
    def use_testset(self, folder_path: str, loop: bool = True):
        """
        테스트셋 폴더를 비디오 소스로 사용
        
        Args:
            folder_path: 테스트셋 폴더 경로
            loop: 파일 끝에 도달하면 처음부터 다시 시작할지 여부
        """
        source = TestsetVideoSource(folder_path, loop)
        self.video_manager.set_source(source)
    
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
            "sound_event": None,
            "trigger_source": None,
            "multimodal_analysis": None,
            "error": None,
        }
        extracted_video_audio_path: Optional[Path] = None

        try:
            # 1. 비디오 소스 열기
            if not self.video_manager.open():
                result["error"] = "비디오 소스를 열 수 없습니다"
                return result

            # 2. 소스 타입에 따라 프레임 가져오기
            source = self.video_manager.get_source()
            sound_event = None
            is_audio_source = False
            is_video_audio_source = False
            transcribed_from_file = None
            audio_file_path = self._resolve_audio_file_path(source)
            video_file_path = self._resolve_video_file_path(source)

            if audio_file_path:
                is_audio_source = True
                sound_event = self._analyze_sound_event_file(audio_file_path)
                result["sound_event"] = sound_event
                if sound_event and sound_event.get("top_event"):
                    print(
                        "🔊 비음성(YAMNet): %s (%.2f)"
                        % (
                            sound_event.get("top_event"),
                            float(sound_event.get("top_confidence", 0.0) or 0.0),
                        )
                    )
            elif video_file_path:
                extracted_video_audio_path = self._extract_audio_from_video(video_file_path)
                if extracted_video_audio_path:
                    is_video_audio_source = True
                    print(f"🎵 비디오 오디오 추출: {Path(video_file_path).name}")
                    sound_event = self._analyze_sound_event_file(str(extracted_video_audio_path))
                    result["sound_event"] = sound_event
                    transcribed_from_file = self._transcribe_audio_file(str(extracted_video_audio_path))
                    if transcribed_from_file:
                        print(f'음성 인식됨: "{transcribed_from_file}"')
                    if sound_event and sound_event.get("top_event"):
                        print(
                            "🔊 비음성(YAMNet): %s (%.2f)"
                            % (
                                sound_event.get("top_event"),
                                float(sound_event.get("top_confidence", 0.0) or 0.0),
                            )
                        )
                    result["trigger_source"] = "video_audio"
                else:
                    print("⚠️  비디오 오디오 추출 실패 - 영상 프레임만 분석합니다.")

            # 이미지 파일인 경우 단일 프레임만 가져옴
            if is_audio_source:
                frame = self._build_audio_placeholder_frame(Path(audio_file_path).name, sound_event)
                frames = [frame]
                timestamps = [0.0]
                result["trigger_source"] = (
                    "sound_event_file" if (sound_event and sound_event.get("triggered", False)) else "audio_file"
                )
            elif isinstance(source, FileVideoSource) and source.is_image:
                frame = source.capture_frame()
                if frame is not None:
                    frames = [frame]
                    timestamps = [0.0]
                else:
                    result["error"] = "이미지를 읽을 수 없습니다"
                    return result
            elif isinstance(source, TestsetVideoSource):
                # 테스트셋의 현재 파일이 이미지인지 확인
                current_src = source.current_source
                if current_src and current_src.is_image:
                    frame = current_src.capture_frame()
                    if frame is not None:
                        frames = [frame]
                        timestamps = [0.0]
                    else:
                        result["error"] = "이미지를 읽을 수 없습니다"
                        return result
                else:
                    # 비디오인 경우 세그먼트 캡처
                    frames, timestamps = self._capture_and_process_video()
            else:
                # 웹캠/네트워크 등 비디오 소스
                frames, timestamps = self._capture_and_process_video()

            if not frames:
                result["error"] = "프레임을 가져올 수 없습니다"
                return result

            # 이미지 다운샘플링 적용
            frames = [self.downsampler.downsample_image(f) for f in frames]
            result["video_analysis"] = {
                "frame_count": len(frames),
                "timestamps": timestamps,
            }

            # 3. 멀티모달 분석 (영상 + 텍스트 입력)
            if self.multimodal_analyzer and frames:
                representative_frame = frames[len(frames) // 2]

                default_text = "현재 상황을 분석해 주세요. 위험하거나 긴급한 상황인지 판단해 주세요."
                if is_audio_source:
                    sound_hint = ""
                    if sound_event and sound_event.get("top_event"):
                        sound_hint = f" (top_event={sound_event.get('top_event')})"
                    default_text = f"[오디오 파일 분석] {Path(audio_file_path).name}{sound_hint}"
                elif is_video_audio_source:
                    if transcribed_from_file:
                        default_text = transcribed_from_file
                    else:
                        sound_hint = ""
                        if sound_event and sound_event.get("top_event"):
                            sound_hint = f" / top_event={sound_event.get('top_event')}"
                        default_text = f"[비디오 오디오 분석] {Path(video_file_path).name}{sound_hint}"
                analysis_text = text_input or default_text

                additional_context_parts = []
                if is_audio_source:
                    additional_context_parts.append(
                        "[테스트 모드] testset 오디오 파일을 YAMNet으로 분석하고, 시각 입력은 오디오 플레이스홀더 이미지를 사용했습니다."
                    )
                    if sound_event:
                        additional_context_parts.append(self._format_sound_event_context(sound_event))
                elif is_video_audio_source:
                    additional_context_parts.append(
                        "[테스트 모드] 비디오에서 오디오를 추출해 YAMNet/STT 분석 후 시각 프레임과 결합했습니다."
                    )
                    if sound_event:
                        additional_context_parts.append(self._format_sound_event_context(sound_event))
                else:
                    additional_context_parts.append(
                        "[테스트 모드] 실제 음성 입력 없이 영상만 분석. 영상에서 보이는 상황을 객관적으로 분석하세요."
                    )
                additional_context = "\n\n".join(additional_context_parts)

                multimodal_result = self.multimodal_analyzer.analyze_with_image(
                    audio_text=analysis_text,
                    image_source=representative_frame,
                    additional_context=additional_context,
                )

                result["multimodal_analysis"] = multimodal_result

            result["success"] = True
            self._save_result_log(result)
            return result

        except Exception as e:
            logger.exception("analyze_video_only failed")
            result["error"] = str(e)
            return result
        finally:
            if extracted_video_audio_path and extracted_video_audio_path.exists():
                try:
                    extracted_video_audio_path.unlink()
                except Exception as e:
                    logger.warning(
                        "Failed to delete extracted video audio %s: %s",
                        extracted_video_audio_path,
                        e,
                    )

    @staticmethod
    def _build_audio_placeholder_frame(filename: str, sound_event: Optional[Dict[str, Any]]) -> np.ndarray:
        """오디오 파일 분석용 플레이스홀더 프레임 생성"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "AUDIO TESTSET INPUT", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.putText(frame, f"FILE: {filename[:70]}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        top_event = (sound_event or {}).get("top_event", "N/A")
        top_conf = float((sound_event or {}).get("top_confidence", 0.0) or 0.0)
        cv2.putText(frame, f"YAMNET: {top_event} ({top_conf:.2f})", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        return frame

    @staticmethod
    def _resolve_audio_file_path(source: Optional[BaseVideoSource]) -> Optional[str]:
        if isinstance(source, FileVideoSource) and source.is_audio:
            return str(source.file_path)
        if isinstance(source, TestsetVideoSource):
            current_src = source.current_source
            if isinstance(current_src, FileVideoSource) and current_src.is_audio:
                return str(current_src.file_path)
        return None

    @staticmethod
    def _resolve_video_file_path(source: Optional[BaseVideoSource]) -> Optional[str]:
        if isinstance(source, FileVideoSource) and source.is_video:
            return str(source.file_path)
        if isinstance(source, TestsetVideoSource):
            current_src = source.current_source
            if isinstance(current_src, FileVideoSource) and current_src.is_video:
                return str(current_src.file_path)
        return None

    def _extract_audio_from_video(self, video_file_path: str) -> Optional[Path]:
        """
        비디오 파일에서 mono/16k WAV를 추출한다.
        1) ffmpeg CLI 시도
        2) librosa+soundfile 폴백
        """
        if not video_file_path:
            return None

        video_path = Path(video_file_path)
        if not video_path.exists():
            return None

        import tempfile

        temp_audio_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=self.recordings_dir,
            delete=False,
            prefix="video_audio_",
        )
        temp_audio_path = Path(temp_audio_file.name)
        temp_audio_file.close()

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(temp_audio_path),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if temp_audio_path.exists() and temp_audio_path.stat().st_size > 44:
                    return temp_audio_path
            except Exception as e:
                logger.debug("ffmpeg audio extraction failed: %s", e)

        try:
            import librosa
            import soundfile as sf

            waveform, _ = librosa.load(str(video_path), sr=16000, mono=True)
            if waveform is not None and len(waveform) > 0:
                sf.write(str(temp_audio_path), waveform, 16000)
                if temp_audio_path.exists() and temp_audio_path.stat().st_size > 44:
                    return temp_audio_path
        except Exception as e:
            logger.debug("librosa fallback extraction failed: %s", e)

        try:
            if temp_audio_path.exists():
                temp_audio_path.unlink()
        except Exception:
            pass
        return None

    def _transcribe_audio_file(self, audio_file_path: str, language: str = "ko-KR") -> Optional[str]:
        """파일 오디오를 STT로 텍스트 변환(가능한 경우에만)."""
        if not audio_file_path or not SPEECH_RECOGNITION_AVAILABLE:
            return None
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.warning("Audio-file speech recognition request failed: %s", e)
            return None
        except Exception as e:
            logger.debug("Audio-file speech recognition failed: %s", e)
            return None

    def _analyze_sound_event_file(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        if not audio_file_path or not self.sound_event_detector:
            return None
        detect_from_file = getattr(self.sound_event_detector, "detect_from_file", None)
        if not callable(detect_from_file):
            return None
        try:
            return detect_from_file(audio_file_path)
        except Exception as e:
            logger.debug("Sound event detection from file failed: %s", e)
            return None

    def analyze_configured_media_inputs(
        self,
        image_path: str = "",
        video_path: str = "",
        audio_path: str = "",
        text_input: Optional[str] = None,
        phrase_time_limit: float = 6.0,
    ) -> Dict[str, Any]:
        """
        config.media_test 기반 단발 분석.

        규칙:
        - image_path만 있으면: 이미지 + 실시간 음성 입력(STT)
        - image_path + audio_path면: 이미지 + 오디오파일(YAMNet)
        - video_path면: 대표 프레임 캡처 + 비디오 오디오 자동 추출(YAMNet/STT)
          (audio_path가 있으면 해당 파일 오디오를 우선 사용)
        - 경로가 빈 문자열("")이면 미입력으로 간주
        """
        image_path = (image_path or "").strip()
        video_path = (video_path or "").strip()
        audio_path = (audio_path or "").strip()
        fallback_text = (text_input or "").strip() or None

        if image_path and video_path:
            logger.warning("Both image_path and video_path are set; image_path takes precedence.")
            video_path = ""

        frame: Optional[np.ndarray] = None
        audio = None
        sound_event = None
        transcribed_text = None
        trigger_source = "unknown"
        visual_source = "none"
        video_file: Optional[Path] = None
        extracted_video_audio_path: Optional[Path] = None

        try:
            # 1) 시각 입력 준비
            if image_path:
                image_file = Path(image_path)
                if not image_file.exists():
                    return {"success": False, "error": f"이미지 파일을 찾을 수 없습니다: {image_path}"}
                frame = cv2.imread(str(image_file))
                if frame is None:
                    return {"success": False, "error": f"이미지를 읽을 수 없습니다: {image_path}"}
                visual_source = f"image:{image_file.name}"
            elif video_path:
                video_file = Path(video_path)
                if not video_file.exists():
                    return {"success": False, "error": f"비디오 파일을 찾을 수 없습니다: {video_path}"}
                source = FileVideoSource(str(video_file))
                if not source.open():
                    return {"success": False, "error": f"비디오 파일을 열 수 없습니다: {video_path}"}
                frame = source.capture_frame()
                source.close()
                if frame is None:
                    return {"success": False, "error": f"비디오 프레임을 캡처할 수 없습니다: {video_path}"}
                visual_source = f"video:{video_file.name}"
            else:
                if not self.video_manager.open():
                    return {"success": False, "error": "카메라를 열 수 없습니다"}
                frame = self.video_manager.capture_frame()
                if frame is None:
                    return {"success": False, "error": "카메라 프레임을 가져올 수 없습니다"}
                visual_source = "live_camera"

            # 2) 음성/오디오 입력 준비
            if audio_path:
                audio_file = Path(audio_path)
                if not audio_file.exists():
                    return {"success": False, "error": f"오디오 파일을 찾을 수 없습니다: {audio_path}"}
                sound_event = self._analyze_sound_event_file(str(audio_file))
                if sound_event and sound_event.get("top_event"):
                    print(
                        "비음성(YAMNet): %s (%.2f)" % (
                            sound_event.get("top_event"),
                            float(sound_event.get("top_confidence", 0.0) or 0.0),
                        )
                    )
                trigger_source = "audio_file"
                transcribed_text = fallback_text
                if not transcribed_text:
                    top_event = (sound_event or {}).get("top_event", "unknown")
                    transcribed_text = f"[오디오 파일 입력] {audio_file.name} / top_event={top_event}"
            elif video_file is not None:
                extracted_video_audio_path = self._extract_audio_from_video(str(video_file))
                if extracted_video_audio_path is not None:
                    print(f"🎵 비디오 오디오 추출: {video_file.name}")
                    sound_event = self._analyze_sound_event_file(str(extracted_video_audio_path))
                    if sound_event and sound_event.get("top_event"):
                        print(
                            "비음성(YAMNet): %s (%.2f)" % (
                                sound_event.get("top_event"),
                                float(sound_event.get("top_confidence", 0.0) or 0.0),
                            )
                        )
                    transcribed_text = self._transcribe_audio_file(str(extracted_video_audio_path))
                    if transcribed_text:
                        print(f'음성 인식됨: "{transcribed_text}"')
                    trigger_source = "video_audio"
                    if not transcribed_text and fallback_text:
                        transcribed_text = fallback_text
                    if not transcribed_text:
                        top_event = (sound_event or {}).get("top_event", "unknown")
                        transcribed_text = f"[비디오 오디오 입력] {video_file.name} / top_event={top_event}"
                else:
                    print("⚠️  비디오 오디오 추출 실패 - 영상 프레임만 분석합니다.")
                    trigger_source = "video_no_audio"
                    transcribed_text = fallback_text or f"[비디오 입력] {video_file.name} / 오디오 추출 실패"
            else:
                if not self._require_speech_detector():
                    return {"success": False, "error": "speech.enabled=true 및 SpeechRecognition 설치가 필요합니다."}

                print("🎤 음성 입력 대기 중...")
                transcribed_text, audio = self.speech_detector.listen_and_recognize(
                    phrase_time_limit=phrase_time_limit
                )
                if transcribed_text:
                    print(f'음성 인식됨: "{transcribed_text}"')
                sound_event = self._analyze_sound_event(audio)
                if sound_event and sound_event.get("top_event"):
                    print(
                        "비음성(YAMNet): %s (%.2f)" % (
                            sound_event.get("top_event"),
                            float(sound_event.get("top_confidence", 0.0) or 0.0),
                        )
                    )

                has_speech = bool(transcribed_text)
                has_sound_trigger = bool(sound_event and sound_event.get("triggered", False))
                if not has_speech and not has_sound_trigger and not fallback_text:
                    return {"success": False, "error": "음성/비음성 트리거를 감지하지 못했습니다."}

                if not transcribed_text and fallback_text:
                    transcribed_text = fallback_text

                if has_speech and has_sound_trigger:
                    trigger_source = "speech+sound_event"
                elif has_speech:
                    trigger_source = "speech"
                elif has_sound_trigger:
                    trigger_source = "sound_event"
                else:
                    trigger_source = "text_fallback"

            # 3) 분석 실행
            frame = self.downsampler.downsample_image(frame)
            result = self._analyze_with_data(
                transcribed_text=transcribed_text,
                audio=audio,
                frame=frame,
                sound_event=sound_event,
                trigger_source=trigger_source,
            )
            result["media_test"] = {
                "image_path": image_path,
                "video_path": video_path,
                "audio_path": audio_path,
                "visual_source": visual_source,
                "extracted_audio": str(extracted_video_audio_path) if extracted_video_audio_path else "",
            }
            return result

        except Exception as e:
            logger.exception("analyze_configured_media_inputs failed")
            return {"success": False, "error": str(e)}
        finally:
            if extracted_video_audio_path and extracted_video_audio_path.exists():
                try:
                    extracted_video_audio_path.unlink()
                except Exception as e:
                    logger.warning(
                        "Failed to delete extracted video audio %s: %s",
                        extracted_video_audio_path,
                        e,
                    )
    
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
            return []
        
        results = []
        files = source.list_files()
        
        for i, filename in enumerate(files):
            # 파일 선택
            if not source.select_file(i):
                continue
            
            # 분석
            result = self.analyze_video_only(text_input)
            result["file_index"] = i
            result["filename"] = filename
            
            results.append(result)
        
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
            "sound_event": None,
            "trigger_source": None,
            "transcribed_text": None,
            "voice_characteristics": None,
            "video_analysis": None,
            "multimodal_analysis": None,
            "error": None
        }
        
        try:
            if not self._require_speech_detector():
                result["error"] = "음성 감지기가 비활성화되었습니다. SpeechRecognition/PyAudio 설치 후 재시도하세요."
                return result

            # 1. 카메라 미리 열어두기
            if not self.video_manager.open():
                result["error"] = "카메라를 열 수 없습니다"
                return result
            
            # 2. 음성 듣고 인식 (문장이 완성될 때까지 대기)
            #    텍스트가 인식되면 그 순간 반환
            transcribed_text, audio = self.speech_detector.listen_and_recognize(
                phrase_time_limit=phrase_time_limit
            )

            # 비음성 이벤트 분석 (YAMNet)
            sound_event = self._analyze_sound_event(audio)
            result["sound_event"] = sound_event

            has_speech = bool(transcribed_text)
            has_sound_trigger = bool(sound_event and sound_event.get("triggered", False))

            if not has_speech and not has_sound_trigger:
                return result

            result["speech_detected"] = has_speech
            result["transcribed_text"] = transcribed_text
            if has_speech and has_sound_trigger:
                result["trigger_source"] = "speech+sound_event"
            elif has_speech:
                result["trigger_source"] = "speech"
            else:
                result["trigger_source"] = "sound_event"
            
            # 3. 문장이 인식됨! 이 순간 영상 캡처
            if has_speech:
                print(f"🎤 \"{transcribed_text}\"")
            else:
                top_event = (sound_event or {}).get("top_event", "unknown")
                top_conf = (sound_event or {}).get("top_confidence", 0.0)
                print(f"🔊 비음성 이벤트 감지: {top_event} ({top_conf:.2f})")
            print("📸 영상 캡처 중...")
            
            # 현재 프레임 캡처 (이미지 1장)
            frame = self.video_manager.capture_frame()
            if frame is not None:
                frame = self.downsampler.downsample_image(frame)
                video_frames = [frame]
                result["video_analysis"] = {"frame_count": 1}
            else:
                video_frames = []
            
            # 4. 음성 특성 분석 (병렬 처리 가능)
            audio_path = None
            voice_features = None
            
            if has_speech and audio and self.voice_characteristics_analyzer:
                import tempfile
                temp_audio_file = tempfile.NamedTemporaryFile(
                    suffix='.wav', 
                    dir=self.recordings_dir, 
                    delete=False,
                    prefix='temp_audio_'
                )
                audio_path = Path(temp_audio_file.name)
                temp_audio_file.close()
                self.speech_detector.save_audio_to_wav(audio, str(audio_path))
                
                voice_features = self._analyze_voice_characteristics(str(audio_path))
                result["voice_characteristics"] = voice_features
            
            # 5. 멀티모달 분석 (음성 텍스트 + 영상)
            if video_frames and self.multimodal_analyzer:
                print("🔍 멀티모달 분석 중...")
                
                representative_frame = video_frames[0]
                
                # 음성 특성 정보를 추가 컨텍스트로 전달
                additional_context_parts = []
                if voice_features:
                    additional_context_parts.append(self._format_voice_features_context(voice_features))
                if sound_event:
                    additional_context_parts.append(self._format_sound_event_context(sound_event))
                additional_context = "\n\n".join([ctx for ctx in additional_context_parts if ctx]) if additional_context_parts else None

                analysis_text = transcribed_text if transcribed_text else "[음성 텍스트 없음] 비음성 위험 소리가 감지되었습니다."
                
                multimodal_result = self.multimodal_analyzer.analyze_with_image(
                    audio_text=analysis_text,
                    image_source=representative_frame,
                    additional_context=additional_context,
                    audio_file_path=str(audio_path) if audio_path else None
                )
                
                result["multimodal_analysis"] = multimodal_result
            
            # 6. 성공 표시
            result["success"] = True
            
            # 7. 결과 로그 저장
            self._save_result_log(result)
            
            # 8. 임시 오디오 파일 삭제
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.warning("Failed to delete temp audio file %s: %s", audio_path, e)
            
            return result
        
        except Exception as e:
            logger.exception("analyze_once failed")
            result["error"] = str(e)
            return result
    
    def _analyze_voice_characteristics(self, audio_path: str) -> Dict[str, Any]:
        """음성 특성 분석"""
        try:
            features = self.voice_characteristics_analyzer.extract_features(audio_path)
            
            # 긴급도 점수 계산
            emergency_indicators = self._calculate_voice_emergency_indicators(features)
            
            return {
                "features": features,
                "emergency_indicators": emergency_indicators
            }
        except Exception:
            logger.exception("Voice characteristics analysis failed")
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
        
        # voice_analysis 설정에서 임계값 읽기
        # self.config는 초기화 시 get_config로 받은 값
        voice_cfg = self.voice_thresholds or {}
        
        pitch_cfg = voice_cfg.get('pitch', {})
        energy_cfg = voice_cfg.get('energy', {})
        speech_rate_cfg = voice_cfg.get('speech_rate', {})
        
        # config에서 임계값 로드 (없으면 기본값 사용)
        high_pitch_threshold = pitch_cfg.get('high_threshold', 180)
        pitch_variability_threshold = pitch_cfg.get('variability_threshold', 40)
        high_energy_threshold = energy_cfg.get('high_threshold', 0.05)
        fast_speech_threshold = speech_rate_cfg.get('fast_threshold', 7)
        
        score = 0.0
        
        # 디버그: 분석된 특성 출력
        print("\n🔍 음성 특성 분석 상세:")
        print(f"  - Pitch: {features.get('pitch', {})}")
        print(f"  - Energy: {features.get('energy', {})}")
        print(f"  - Speech Rate: {features.get('speech_rate', {})}")
        
        # 피치 분석 (높은 피치 = 긴장/공포)
        pitch = features.get("pitch", {})
        if isinstance(pitch, dict):
            pitch_mean = pitch.get("mean", 0)
            pitch_std = pitch.get("std", 0)
            print(f"  → Pitch Mean: {pitch_mean:.1f} (threshold: {high_pitch_threshold})")
            print(f"  → Pitch Std: {pitch_std:.1f} (threshold: {pitch_variability_threshold})")
            
            if pitch_mean > high_pitch_threshold:
                indicators["high_pitch"] = True
                score += 0.25
            if pitch_std > pitch_variability_threshold:  # 피치 변동이 크면 (떨림)
                indicators["voice_trembling"] = True
                score += 0.25
        
        # 에너지 분석 (높은 에너지 = 소리 지름)
        energy = features.get("energy", {})
        if isinstance(energy, dict):
            energy_max = energy.get("max", 0)
            print(f"  → Energy Max: {energy_max:.3f} (threshold: {high_energy_threshold})")
            if energy_max > high_energy_threshold:
                indicators["high_energy"] = True
                score += 0.25
        
        # 말 속도 분석 (빠른 말 = 급박함)
        speech_rate = features.get("speech_rate", {})
        if isinstance(speech_rate, dict):
            syllables_per_sec = speech_rate.get("estimated_syllables_per_second", 0)
            print(f"  → Speech Rate: {syllables_per_sec:.2f} syllables/sec (threshold: {fast_speech_threshold})")
            if syllables_per_sec > fast_speech_threshold:
                indicators["fast_speech"] = True
                score += 0.25
        
        print(f"  → 결과: {indicators}")
        
        indicators["overall_score"] = min(score, 1.0)
        
        return indicators
    
    def _capture_and_process_video(self) -> Tuple[List[np.ndarray], List[float]]:
        """비디오 캡처 및 다운샘플링"""
        # 비디오 세그먼트 캡처
        frames, timestamps = self.video_manager.capture_video_segment(
            duration=self.downsampling_config.video_capture_duration,
            target_fps=self.downsampling_config.video_fps
        )
        
        # 다운샘플링 적용
        frames, timestamps = self.downsampler.downsample_video_frames(frames, timestamps)
        
        return frames, timestamps

    def _analyze_sound_event(self, audio: Any) -> Optional[Dict[str, Any]]:
        """YAMNet 기반 비음성 이벤트 분석"""
        if not audio or not self.sound_event_detector:
            return None

        try:
            return self.sound_event_detector.detect_from_audio(audio)
        except Exception as e:
            logger.debug("Sound event detection failed: %s", e)
            return None

    def _format_sound_event_context(self, sound_event: Dict[str, Any]) -> str:
        """사운드 이벤트 결과를 LLM 컨텍스트 문자열로 포맷"""
        if not sound_event:
            return ""

        lines = ["**사운드 이벤트 분석 결과(YAMNet):**"]
        top_event = sound_event.get("top_event")
        top_conf = float(sound_event.get("top_confidence", 0.0) or 0.0)
        if top_event:
            lines.append(f"- 최상위 이벤트: {top_event} (신뢰도 {top_conf:.2f})")

        emergency_events = sound_event.get("emergency_events", []) or []
        if emergency_events:
            lines.append("- 위험 이벤트 후보:")
            for event in emergency_events[:3]:
                lines.append(f"  - {event.get('label', 'unknown')} ({float(event.get('confidence', 0.0)):.2f})")
        else:
            lines.append("- 위험 이벤트 후보 없음")

        lines.append(f"- 트리거 여부: {'예' if sound_event.get('triggered', False) else '아니오'}")
        return "\n".join(lines)
    
    def _format_voice_features_context(self, voice_features: Dict) -> str:
        """음성 특성을 LLM 분석용 컨텍스트로 포맷"""
        if not voice_features:
            return ""
        
        indicators = voice_features.get("emergency_indicators", {})
        features = voice_features.get("features", {})
        
        context_parts = ["**음성 특성 분석 결과:**"]
        
        # Raw 특성값 출력 (LLM이 더 정확하게 판단하도록)
        pitch = features.get("pitch", {})
        if isinstance(pitch, dict):
            pitch_mean = pitch.get("mean", 0)
            pitch_std = pitch.get("std", 0)
            context_parts.append(f"**피치(음높이):** 평균 {pitch_mean:.1f}Hz, 변동 {pitch_std:.1f}Hz")
        
        energy = features.get("energy", {})
        if isinstance(energy, dict):
            energy_max = energy.get("max", 0)
            energy_mean = energy.get("mean", 0)
            context_parts.append(f"**에너지(음량):** 최대 {energy_max:.3f}, 평균 {energy_mean:.3f}")
        
        speech_rate = features.get("speech_rate", {})
        if isinstance(speech_rate, dict):
            syllables_per_sec = speech_rate.get("estimated_syllables_per_second", 0)
            context_parts.append(f"**말 속도:** {syllables_per_sec:.2f} 음절/초")
        
        # 구체적인 특성 설명
        context_parts.append("\n**특성 분석:**")
        
        if indicators.get("high_pitch"):
            context_parts.append("- 높은 피치 감지 → 긴장/공포 가능성")
        
        if indicators.get("high_energy"):
            context_parts.append("- 높은 음성 에너지 → 소리 지름/강한 감정 표출")
        
        if indicators.get("fast_speech"):
            context_parts.append("- 빠른 말 속도 → 급박한 상황/불안정 심리")
        
        if indicators.get("voice_trembling"):
            context_parts.append("- 음성 떨림 감지 → 두려움/극심한 스트레스")
        
        # 특성이 없으면 안정적 상태로 기술
        if not any([indicators.get("high_pitch"), indicators.get("high_energy"), 
                    indicators.get("fast_speech"), indicators.get("voice_trembling")]):
            context_parts.append("- 음성이 안정적이고 진정된 상태")
        
        # 전반적 평가 (점수 대신 설명)
        score = indicators.get("overall_score", 0)
        if score > 0.7:
            context_parts.append("\n→ 종합 평가: 매우 절박하고 긴장된 상태")
        elif score > 0.4:
            context_parts.append("\n→ 종합 평가: 부분적인 긴장 또는 스트레스 신호")
        else:
            context_parts.append("\n→ 종합 평가: 음성 특성상 특별한 긴급 신호 없음")
        
        return "\n".join(context_parts)
    
    def _save_result_log(self, result: Dict):
        """결과 로그 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"integrated_analysis_{timestamp}.json"
        
        # numpy array 등 직렬화 불가능한 객체 처리
        serializable_result = self._make_serializable(result)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
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
        max_iterations: int = None,
        verbose: bool = False,
        parallel: bool = False,
    ):
        """
        연속 모니터링 시작
        
        Args:
            on_result: 결과 콜백 함수
            max_iterations: 최대 반복 횟수 (None이면 무한)
            verbose: 상세 출력 여부
            parallel: 호환 옵션 (현재는 순차 처리와 동일)
        """
        if not self._require_speech_detector():
            raise RuntimeError("음성 감지기가 비활성화되어 모니터링을 시작할 수 없습니다.")

        self.on_result_callback = on_result
        self.is_monitoring = True
        self.verbose = verbose
        self.parallel = parallel

        if parallel:
            logger.warning("`parallel` mode is currently mapped to sequential monitoring for compatibility.")
        
        # 카메라 미리 열기
        self.video_manager.open()
        
        # OpenCV 디스플레이 시작 (설정된 경우)
        if self.use_opencv_display and self.opencv_display:
            self.opencv_display.start()
        
        # 웹 대시보드 연동 확인
        try:
            from web.app import dashboard, enable_video_stream
            if dashboard.running:
                self.use_web_dashboard = True
                print("   📡 웹 대시보드 연동 활성화")
                
                # 웹 비디오 스트리밍 비활성화 (localhost 접근 거부 이슈)
                enable_video_stream(False)
                self.web_video_streaming = False
        except Exception as e:
            logger.debug("Web dashboard integration unavailable: %s", e)
            self.web_video_streaming = False
        
        self._start_monitoring_sequential(max_iterations)
    
    
    def _start_monitoring_sequential(self, max_iterations: int = None):
        """순차 모니터링: 백그라운드 음성 감지 방식"""
        print("\n🔄 모니터링 시작 (Ctrl+C로 종료)")
        print("   💡 백그라운드 음성 감지 중... 아무거나 말씀하세요!")
        
        iteration = 0
        
        # 백그라운드 음성 감지 시작
        if not self.speech_detector:
            print("❌ 음성 감지기가 비활성화되어 모니터링을 진행할 수 없습니다.")
            self.stop_monitoring()
            return
        self.speech_detector.start_background_listening()
        
        try:
            while self.is_monitoring:
                if max_iterations and iteration >= max_iterations:
                    print(f"\n✅ {max_iterations}회 분석 완료!")
                    break
                
                # 비블로킹 - 감지된 음성이 있는지 확인
                transcribed_text, audio = self.speech_detector.get_recognized_speech()

                if audio is not None:
                    sound_event = self._analyze_sound_event(audio)
                    has_speech = bool(transcribed_text)
                    has_sound_trigger = bool(sound_event and sound_event.get("triggered", False))

                    if not has_speech and not has_sound_trigger:
                        time.sleep(0.01)
                        continue

                    trigger_source = "speech+sound_event" if (has_speech and has_sound_trigger) else ("speech" if has_speech else "sound_event")
                    if sound_event and sound_event.get("top_event"):
                        top_event = sound_event.get("top_event")
                        top_conf = float(sound_event.get("top_confidence", 0.0) or 0.0)
                        print(f"비음성(YAMNet): {top_event} ({top_conf:.2f})")
                    else:
                        print("비음성(YAMNet): N/A")

                    print("📸 영상 캡처 중...")
                    
                    # 현재 프레임 캡처
                    frame = self.video_manager.capture_frame()
                    if frame is not None:
                        frame = self.downsampler.downsample_image(frame)
                    
                    # 분석 수행
                    result = self._analyze_with_data(transcribed_text, audio, frame, sound_event=sound_event, trigger_source=trigger_source)
                    
                    # 분석 결과 처리
                    if result.get("success"):
                        iteration += 1
                        
                        # 콜백 호출
                        if self.on_result_callback:
                            self.on_result_callback(result)
                        
                        # 결과 출력
                        self._print_result_summary(result, verbose=self.verbose)
                
                # 메인 스레드에서 OpenCV 렌더링 처리 (필수: 메인 스레드만 가능)
                if self.opencv_display and self.opencv_display.is_running():
                    if not self.opencv_display.render():
                        self.is_monitoring = False
                        break
                
                time.sleep(0.01)  # 이벤트 루프 속도 제어
        
        except KeyboardInterrupt:
            print("\n⏹️  모니터링 중지됨")
        
        finally:
            # 백그라운드 리스닝 중지
            if self.speech_detector:
                self.speech_detector.stop_background_listening()
            self.stop_monitoring()
    
    def _analyze_with_data(
        self,
        transcribed_text: Optional[str],
        audio: Any,
        frame: np.ndarray,
        sound_event: Optional[Dict[str, Any]] = None,
        trigger_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """이미 캡처된 데이터로 분석 수행"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "speech_detected": bool(transcribed_text),
            "sound_event": sound_event,
            "trigger_source": trigger_source,
            "transcribed_text": transcribed_text,
            "voice_characteristics": None,
            "video_analysis": None,
            "multimodal_analysis": None,
            "error": None
        }
        
        try:
            video_frames = [frame] if frame is not None else []
            result["video_analysis"] = {"frame_count": len(video_frames)}
            
            # 음성 특성 분석
            audio_path = None
            voice_features = None
            
            if audio and transcribed_text:
                if self.voice_characteristics_analyzer:
                    import tempfile
                    temp_audio_file = tempfile.NamedTemporaryFile(
                        suffix='.wav', 
                        dir=self.recordings_dir, 
                        delete=False,
                        prefix='temp_audio_'
                    )
                    audio_path = Path(temp_audio_file.name)
                    temp_audio_file.close()
                    if self.speech_detector:
                        self.speech_detector.save_audio_to_wav(audio, str(audio_path))
                    else:
                        logger.warning("Speech detector unavailable while trying to save audio.")
                        audio_path = None
                    
                    voice_features = self._analyze_voice_characteristics(str(audio_path))
                    result["voice_characteristics"] = voice_features
                    if voice_features:
                        print("✅ 음성 특성 분석 완료")
                else:
                    print("⚠️  음성 특성 분석기 비활성화")
            else:
                print("⚠️  오디오 데이터 없음")
            
            # 멀티모달 분석
            if video_frames and self.multimodal_analyzer:
                print("🔍 멀티모달 분석 중...")
                
                representative_frame = video_frames[0]
                
                additional_context_parts = []
                if voice_features:
                    additional_context_parts.append(self._format_voice_features_context(voice_features))
                if sound_event:
                    additional_context_parts.append(self._format_sound_event_context(sound_event))
                additional_context = "\n\n".join([ctx for ctx in additional_context_parts if ctx]) if additional_context_parts else None

                analysis_text = transcribed_text if transcribed_text else "[음성 텍스트 없음] 비음성 위험 소리가 감지되었습니다."
                
                multimodal_result = self.multimodal_analyzer.analyze_with_image(
                    audio_text=analysis_text,
                    image_source=representative_frame,
                    additional_context=additional_context,
                    audio_file_path=str(audio_path) if audio_path else None
                )
                
                result["multimodal_analysis"] = multimodal_result
            
            result["success"] = True
            
            # 로그 저장
            self._save_result_log(result)
            
            # 임시 파일 삭제
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.warning("Failed to delete temp audio file %s: %s", audio_path, e)
            
            return result
        
        except Exception as e:
            logger.exception("_analyze_with_data failed")
            result["error"] = str(e)
            return result
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        self.video_manager.close()
        
        # OpenCV 디스플레이 정리
        if self.opencv_display:
            self.opencv_display.stop()
        
        # 웹 비디오 스트리밍 정리
        if getattr(self, 'web_video_streaming', False):
            try:
                from web.app import enable_video_stream
                enable_video_stream(False)
            except Exception as e:
                logger.debug("Failed to disable web video stream: %s", e)
    
    def _push_to_displays(self, result: Dict, frame=None):
        """결과를 디스플레이들(웹, OpenCV)에 전송"""
        # 웹 대시보드로 전송
        if self.use_web_dashboard:
            try:
                from web.app import push_result
                push_result(result)
            except Exception as e:
                logger.warning("Failed to push result to dashboard: %s", e)
        
        # OpenCV 디스플레이 업데이트
        if self.opencv_display and self.opencv_display.is_running():
            self.opencv_display.update_result(result)
            if frame is not None:
                self.opencv_display.update_frame(frame)
    
    def _print_result_summary(self, result: Dict, verbose: bool = False):
        """결과 요약 출력"""
        
        if not result.get("success"):
            return
        
        analysis = result.get("multimodal_analysis")
        if not analysis:
            return
        
        # 디스플레이로 결과 전송
        self._push_to_displays(result, result.get("_frame"))
        
        # 긴급 신호 여부
        is_emergency = analysis.get('is_emergency', False)
        
        # 헤더 색상 구분
        if is_emergency:
            print("\n" + "🚨" * 50)
            print("🚨 ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️ 긴급 상황 감지! 🚨")
            print("🚨" * 50)
        else:
            print("\n" + "=" * 50)
            print("📊 분석 결과")
            print("=" * 50)
        
        # 음성 입력
        text = result.get("transcribed_text", "")
        if text:
            print(f"📝 음성 입력: \"{text}\"")
        else:
            print("📝 음성 입력: (없음)")

        sound_event = result.get("sound_event")
        if sound_event and sound_event.get("top_event"):
            print("\n🔊 사운드 이벤트 분석:")
            print(f"   - 최상위 이벤트: {sound_event.get('top_event')} ({float(sound_event.get('top_confidence', 0.0)):.2f})")
            emergency_events = sound_event.get("emergency_events", []) or []
            if emergency_events:
                labels = ", ".join([f"{e.get('label')}({float(e.get('confidence', 0.0)):.2f})" for e in emergency_events[:3]])
                print(f"   - 위험 후보: {labels}")
            print(f"   - 트리거 소스: {result.get('trigger_source', 'N/A')}")
        
        # 음성 특성 분석
        voice = result.get("voice_characteristics")
        if voice:
            print("\n🎤 음성 특성 분석:")
            indicators = voice.get("emergency_indicators", {})
            if indicators.get("high_pitch"):
                print("   - 높은 피치 감지 (긴장/공포 가능성)")
            if indicators.get("high_energy"):
                print("   - 높은 에너지 감지 (소리 지름/흥분)")
            if indicators.get("fast_speech"):
                print("   - 빠른 말 속도 (급박함)")
            if indicators.get("voice_trembling"):
                print("   - 음성 떨림 감지 (불안/공포)")
        
        # 멀티모달 분석 결과
        print("\n🔍 상황 분석:")
        print(f"   - 상황 유형: {analysis.get('situation_type', 'N/A')}")
        print(f"   - 상황 설명: {analysis.get('situation', 'N/A')}")
        print(f"   - 감정 상태: {analysis.get('emotional_state', 'N/A')}")
        print(f"   - 영상 내용: {analysis.get('visual_content', 'N/A')}")
        
        print("\n⚠️  긴급도 판단:")
        if is_emergency:
            print("   - 긴급 여부: 🚨 YES - 즉시 대응 필요!")
        else:
            print("   - 긴급 여부: ✅ 아니오")
        print(f"   - 우선순위: {analysis.get('priority', 'N/A')}")
        print(f"   - 긴급 판단 근거: {analysis.get('emergency_reason', 'N/A')}")
        
        print("\n🎯 음성-영상 일치도:")
        print(f"   - 일치 여부: {analysis.get('audio_visual_consistency', 'N/A')}")
        
        print("\n💡 권장 조치:")
        if is_emergency:
            print(f"   - 🚨 긴급: {analysis.get('action', 'N/A')}")
        else:
            print(f"   - {analysis.get('action', 'N/A')}")
        
        if is_emergency:
            print("🚨" * 50 + "\n")
        else:
            print("=" * 50 + "\n")


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
