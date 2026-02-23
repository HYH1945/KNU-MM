#!/usr/bin/env python3
"""
멀티모달 컨텍스트 분석 모듈
오디오 + 이미지/비디오를 함께 분석하여 더 정확한 상황 판단
음성 특성 분석으로 응급 신호 신뢰도 검증

사용법:
    analyzer = MultimodalAnalyzer()
    
    # 이미지 + 오디오 분석 (음성 특성 포함)
    result = analyzer.analyze_with_image(
        audio_text="도와주세요!",
        image_path="screenshot.jpg",
        audio_file_path="audio.wav"  # 선택사항
    )
"""

import os
import sys
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  OpenCV가 설치되지 않았습니다. 이미지/비디오 기능이 제한됩니다.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow가 설치되지 않았습니다. 이미지 처리 기능이 제한됩니다.")

# 설정 관리자 임포트
try:
    from core.config_manager import get_config, get_prompt, get_openai_config, get_api_key
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # 기본값 함수들
    def get_config(*keys, default=None):
        return default
    def get_prompt(prompt_type='system'):
        return ""
    def get_openai_config(key, default=None):
        return default
    def get_api_key(service='openai'):
        return os.getenv('OPENAI_API_KEY')

# 음성 특성 분석기 임포트
try:
    from core.voice_characteristics import VoiceCharacteristicsAnalyzer
    VOICE_ANALYSIS_AVAILABLE = True
except ImportError:
    VOICE_ANALYSIS_AVAILABLE = False


class MultimodalAnalyzer:
    """멀티모달 컨텍스트 분석기 (오디오 + 비전)"""
    PRIORITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    _LEVEL_MAP = {
        "critical": "CRITICAL",
        "긴급": "CRITICAL",
        "매우높음": "CRITICAL",
        "high": "HIGH",
        "높음": "HIGH",
        "medium": "MEDIUM",
        "중간": "MEDIUM",
        "보통": "MEDIUM",
        "low": "LOW",
        "낮음": "LOW",
        "일반": "LOW",
    }
    
    # 기본 시스템 프롬프트 (config가 없을 때 사용)
    DEFAULT_SYSTEM_PROMPT = """당신은 음성, 이미지, 음성 특성을 종합적으로 분석하는 상황 분석 AI입니다.

분석 프로세스:
1. 음성 특성 해석: 제공된 음성 분석 결과(피치, 에너지, 말 속도, 떨림)를 맥락 해석에 활용
2. 음성 맥락 분석: 단순 키워드가 아닌 전체 의도와 감정 파악
3. 영상 분석: 이미지에서 보이는 실제 상황, 환경, 신체 상태
4. 일관성 평가: 음성 내용 + 음성 특성 + 영상이 일치하는가?
5. 종합 판단: 모든 신호를 종합하여 긴급도 결정

판단 원칙:
- CRITICAL: 음성 특성이 절박함 + 음성 내용이 위험 상황 + 영상 일치 = 즉시 조치
- HIGH: 부분적 절박함 + 위험 신호 있음 + 영상과 부분 일치 = 빠른 대응
- MEDIUM: 음성 특성과 영상이 부분 일치 또는 불명확 = 모니터링
- LOW: 음성 특성, 내용, 영상 모두 일상적 = 특별 조치 불필요

핵심:
- 음성 특성에서 "떨림", "빠른 속도", "높은 에너지"가 있으면 절박함의 신호
- 영상에서 위험 징후(부상, 폭력, 위험 환경)가 보이면 신뢰도 증가
- 음성-영상-특성이 모두 일치하면 긴급도 상향

다음을 JSON으로만 반환하세요:
{
  "context": "음성 맥락 분석",
  "urgency": "위급도 (LOW/MEDIUM/HIGH/CRITICAL)",
  "situation": "음성 내용 + 영상 + 음성 특성 종합 분석",
  "situation_type": "상황 분류",
  "emotional_state": "음성에서 감지되는 감정 상태",
  "visual_content": "영상에서 보이는 실제 상황",
  "audio_visual_consistency": "음성, 음성 특성, 영상의 일관성 평가",
  "is_emergency": true/false,
  "emergency_reason": "긴급 판단 근거 (음성+특성+영상 기반)",
  "priority": "CRITICAL/HIGH/MEDIUM/LOW",
  "action": "권장 조치"
}"""
    
    def __init__(self, model: str = None):
        """
        멀티모달 분석기 초기화
        
        Args:
            model: 사용할 OpenAI 모델 (None이면 config에서 로드)
        """
        # 모델 설정 (인자 > config > 기본값)
        self.model = model or get_config('model', default='gpt-4o-mini')
        
        # API 키 로드 (환경변수 > .env > config.yaml)
        self.api_key = get_api_key('openai')
        
        if not self.api_key:
            raise ValueError(
                "❌ OpenAI API 키가 설정되지 않았습니다.\n"
                "   다음 방법 중 하나로 설정하세요:\n"
                "   1. config/config.yaml의 api_keys.openai에 입력\n"
                "   2. config/.env 파일에 OPENAI_API_KEY=sk-... 형식으로 입력\n"
                "   3. 환경변수로 설정: export OPENAI_API_KEY=sk-..."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        
        # 음성 특성 분석기 초기화 (config에서 설정 확인)
        analysis_cfg = get_config('analysis', default={}) or {}
        voice_cfg = get_config('voice_characteristics', default={}) or {}
        streaming_cfg = get_config('streaming', default={}) or {}

        analysis_voice = analysis_cfg.get('voice_characteristics')
        legacy_voice = voice_cfg.get('enabled')
        self.use_voice_characteristics = analysis_voice if analysis_voice is not None else legacy_voice
        if self.use_voice_characteristics is None:
            self.use_voice_characteristics = True

        analysis_streaming = analysis_cfg.get('streaming')
        legacy_streaming = streaming_cfg.get('enabled')
        self.use_streaming = analysis_streaming if analysis_streaming is not None else legacy_streaming
        if self.use_streaming is None:
            self.use_streaming = False
        
        if VOICE_ANALYSIS_AVAILABLE and self.use_voice_characteristics:
            self.voice_analyzer = VoiceCharacteristicsAnalyzer()
        else:
            self.voice_analyzer = None
        
        # 시스템 프롬프트 (config에서 로드)
        self.system_prompt = get_prompt('system') or self.DEFAULT_SYSTEM_PROMPT
        
        # OpenAI API 설정
        self.max_tokens = get_openai_config('max_tokens', default=800)
        self.temperature = get_openai_config('temperature', default=0.3)
        self.image_detail = get_openai_config('image_detail', default='low')
        self.timeout = get_openai_config('timeout', default=30)
    
    def encode_image_to_base64(self, image_source: Union[str, np.ndarray], max_size: int = 1024) -> str:
        """
        이미지를 base64로 인코딩 (크기 최적화 포함)
        
        Args:
            image_source: 이미지 파일 경로 또는 numpy array (OpenCV 이미지)
            max_size: 최대 이미지 크기 (픽셀). 이보다 크면 리사이징 (기본값: 1024)
        
        Returns:
            base64 인코딩된 문자열
        """
        if isinstance(image_source, str):
            # 파일 경로인 경우 - PIL로 열어서 리사이징
            if not PIL_AVAILABLE:
                # PIL이 없으면 원본 그대로 인코딩
                with open(image_source, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            # PIL로 이미지 열기
            img = Image.open(image_source)
            
            # 크기 조정 (긴 쪽이 max_size를 넘으면 리사이징)
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # JPEG로 변환하여 base64 인코딩
            from io import BytesIO
            buffer = BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        elif isinstance(image_source, np.ndarray):
            # numpy array (OpenCV 이미지)인 경우
            if not OPENCV_AVAILABLE:
                raise ImportError("OpenCV가 필요합니다: pip install opencv-python")
            
            # 크기 조정
            height, width = image_source.shape[:2]
            if max(height, width) > max_size:
                ratio = max_size / max(height, width)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image_source = cv2.resize(image_source, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # BGR to RGB 변환 (OpenCV는 BGR, PIL은 RGB)
            if len(image_source.shape) == 3 and image_source.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_source
            
            # numpy array를 JPEG로 인코딩 (품질 85%)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', image_rgb, encode_param)
            return base64.b64encode(buffer).decode('utf-8')
        
        else:
            raise TypeError("image_source는 파일 경로(str) 또는 numpy array여야 합니다")

    def _normalize_level(self, value: Any, default: str = "LOW") -> str:
        if value is None:
            return default
        normalized = str(value).strip().upper()
        if normalized in self.PRIORITY_ORDER:
            return normalized
        mapped = self._LEVEL_MAP.get(str(value).strip().lower())
        return mapped if mapped else default

    def _normalize_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 응답 스키마를 통일."""
        normalized = dict(result or {})

        raw_priority = normalized.get("priority")
        raw_urgency = normalized.get("urgency")
        priority = self._normalize_level(raw_priority, default="LOW")
        urgency = self._normalize_level(raw_urgency, default=priority)

        # urgency/priority 중 하나만 와도 양쪽을 채워 통합 시스템에서 재사용 가능하게 유지.
        if raw_priority is None and raw_urgency is not None:
            priority = self._normalize_level(raw_urgency, default=priority)
        if raw_urgency is None and raw_priority is not None:
            urgency = self._normalize_level(raw_priority, default=urgency)

        is_emergency = bool(normalized.get("is_emergency", False))
        if priority in {"CRITICAL", "HIGH"}:
            is_emergency = True
        if is_emergency and priority in {"LOW", "MEDIUM"}:
            priority = "HIGH"
            urgency = "HIGH"

        normalized["priority"] = priority
        normalized["urgency"] = urgency
        normalized["is_emergency"] = is_emergency

        # 공통 필드 기본값 보정
        normalized.setdefault("context", "")
        normalized.setdefault("situation", "")
        normalized.setdefault("situation_type", "UNKNOWN")
        normalized.setdefault("emotional_state", "NEUTRAL")
        normalized.setdefault("visual_content", "")
        normalized.setdefault("audio_visual_consistency", "UNKNOWN")
        normalized.setdefault("action", "모니터링 지속")

        if not normalized["is_emergency"]:
            normalized["emergency_reason"] = None
        else:
            normalized.setdefault("emergency_reason", normalized.get("urgency_reason") or "위험 징후 감지")

        return normalized
    
    def analyze_with_image(
        self, 
        audio_text: str, 
        image_source: Union[str, np.ndarray],
        additional_context: Optional[str] = None,
        audio_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        오디오 텍스트와 이미지를 함께 분석 (음성 특성 분석 포함)
        
        Args:
            audio_text: 음성에서 변환된 텍스트
            image_source: 이미지 파일 경로 또는 numpy array
            additional_context: 추가 컨텍스트 정보 (선택)
            audio_file_path: 오디오 파일 경로 (음성 특성 분석용, 선택)
        
        Returns:
            멀티모달 분석 결과 딕셔너리
        """
        try:
            # 이미지를 base64로 인코딩
            base64_image = self.encode_image_to_base64(image_source)
            
            # 사용자 메시지 구성 (음성 + 특성 + 영상)
            user_message = f"""**1. 음성 입력:**
"{audio_text}"
"""
            
            if additional_context:
                user_message += f"""
**2. 음성 특성 분석 결과:**
{additional_context}
"""
            else:
                print("⚠️  음성 특성 분석 정보 없음")
            
            user_message += f"""
**3. 영상:**
제공된 이미지를 분석하여 위 음성과 음성 특성과 함께 전체 상황을 판단해주세요.
"""
            
            # 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": self.image_detail
                            }
                        }
                    ]
                }
            ]
            
            # OpenAI API 호출 (스트리밍 또는 일반)
            if self.use_streaming:
                content = ""
                print("   ", end="", flush=True)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True,
                    timeout=self.timeout,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        content += chunk_content
                        print("▓", end="", flush=True)  # 진행 표시
                print(" ✓")  # 완료 표시
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content
            
            # 안전 정책 거부 감지
            if content and ("I'm sorry" in content or "I can't assist" in content or "I cannot" in content):
                print(f"⚠️  OpenAI 안전 정책으로 인한 분석 거부")
                print(f"   원본 응답: {content}")
                return self._normalize_analysis_result({
                    'context': '이미지 내용을 분석할 수 없음 (안전 정책)',
                    'urgency': 'LOW',
                    'urgency_reason': 'OpenAI 안전 정책으로 인한 분석 제한',
                    'situation': '이미지가 안전 정책에 위배되거나 분석이 제한되었습니다.',
                    'situation_type': 'UNKNOWN',
                    'emotional_state': 'NEUTRAL',
                    'visual_content': '분석 불가',
                    'audio_visual_consistency': 'UNKNOWN',
                    'action': '다른 이미지로 다시 시도하거나 이미지 내용을 확인하세요',
                    'is_emergency': False,
                    'emergency_reason': None,
                    'priority': 'LOW',
                    'error': '안전 정책 거부',
                    'raw_response': content
                })
            
            # JSON 파싱
            try:
                # 코드 블록 제거 (```json ... ```)
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                result = json.loads(content)
            
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"원본 응답:\n{content}")
                return {
                    'error': 'JSON 파싱 실패',
                    'raw_response': content,
                    'context': '분석 오류',
                    'urgency': 'LOW',
                    'is_emergency': False,
                    'priority': 'LOW'
                }
            
            return self._normalize_analysis_result(result)
        
        except Exception as e:
            logger.exception("Multimodal analysis failed")
            return {
                'error': str(e),
                'context': '분석 실패',
                'urgency': 'LOW',
                'is_emergency': False,
                'priority': 'LOW'
            }
    
    def analyze_with_video_frame(
        self,
        audio_text: str,
        frame: np.ndarray,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        오디오 텍스트와 비디오 프레임을 함께 분석
        
        Args:
            audio_text: 음성에서 변환된 텍스트
            frame: 비디오 프레임 (numpy array, OpenCV format)
            additional_context: 추가 컨텍스트 정보 (선택)
        
        Returns:
            멀티모달 분석 결과 딕셔너리
        """
        # analyze_with_image와 동일하게 처리 (프레임은 이미지로 취급)
        return self.analyze_with_image(audio_text, frame, additional_context)
    
    def capture_screenshot(self, save_path: Optional[str] = None) -> str:
        """
        화면 스크린샷 캡처 (macOS)
        
        Args:
            save_path: 저장 경로 (None이면 자동 생성)
        
        Returns:
            저장된 파일 경로
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            save_path = f"screenshots/screenshot_{timestamp}.png"
        
        # 디렉토리 생성
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import subprocess
            # macOS screencapture 명령 사용
            subprocess.run(['screencapture', '-x', save_path], check=True)
            print(f"✅ 스크린샷 저장: {save_path}")
            return save_path
        except Exception as e:
            print(f"❌ 스크린샷 캡처 실패: {e}")
            return None
    
    def capture_webcam_frame(self, camera_id: int = 0, save_path: Optional[str] = None) -> Union[str, np.ndarray]:
        """
        웹캠에서 프레임 캡처
        
        Args:
            camera_id: 카메라 ID (기본값: 0)
            save_path: 저장 경로 (None이면 저장하지 않고 numpy array 반환)
        
        Returns:
            저장 경로 또는 numpy array
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV가 필요합니다: pip install opencv-python")
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                raise RuntimeError(f"카메라 {camera_id}를 열 수 없습니다")
            
            # 프레임 읽기
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError("프레임 캡처 실패")
            
            if save_path:
                # 디렉토리 생성
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, frame)
                print(f"✅ 웹캠 프레임 저장: {save_path}")
                return save_path
            else:
                return frame
        
        except Exception as e:
            print(f"❌ 웹캠 캡처 실패: {e}")
            return None


# 편의 함수
def analyze_audio_with_screenshot(audio_text: str, screenshot_path: Optional[str] = None) -> Dict[str, Any]:
    """
    음성 텍스트와 스크린샷을 함께 분석하는 편의 함수
    
    Args:
        audio_text: 음성에서 변환된 텍스트
        screenshot_path: 스크린샷 경로 (None이면 자동 캡처)
    
    Returns:
        분석 결과
    """
    analyzer = MultimodalAnalyzer()
    
    # 스크린샷이 없으면 자동 캡처
    if screenshot_path is None:
        screenshot_path = analyzer.capture_screenshot()
        if screenshot_path is None:
            return {'error': '스크린샷 캡처 실패'}
    
    return analyzer.analyze_with_image(audio_text, screenshot_path)


def analyze_audio_with_webcam(audio_text: str, camera_id: int = 0) -> Dict[str, Any]:
    """
    음성 텍스트와 웹캠 프레임을 함께 분석하는 편의 함수
    
    Args:
        audio_text: 음성에서 변환된 텍스트
        camera_id: 카메라 ID
    
    Returns:
        분석 결과
    """
    analyzer = MultimodalAnalyzer()
    
    # 웹캠에서 프레임 캡처
    frame = analyzer.capture_webcam_frame(camera_id)
    if frame is None:
        return {'error': '웹캠 프레임 캡처 실패'}
    
    return analyzer.analyze_with_video_frame(audio_text, frame)


if __name__ == "__main__":
    # 테스트
    print("=" * 70)
    print("🎥 멀티모달 분석기 테스트")
    print("=" * 70)
    
    analyzer = MultimodalAnalyzer()
    
    # 1. 스크린샷과 함께 분석
    print("\n1️⃣ 스크린샷 분석 테스트")
    screenshot = analyzer.capture_screenshot()
    if screenshot:
        result = analyzer.analyze_with_image(
            audio_text="이상한 소리가 들려요",
            image_source=screenshot
        )
        print(f"분석 결과: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 2. 웹캠과 함께 분석
    print("\n2️⃣ 웹캠 분석 테스트")
    frame = analyzer.capture_webcam_frame()
    if frame is not None:
        result = analyzer.analyze_with_video_frame(
            audio_text="지금 상황이 어떤가요?",
            frame=frame
        )
        print(f"분석 결과: {json.dumps(result, ensure_ascii=False, indent=2)}")
