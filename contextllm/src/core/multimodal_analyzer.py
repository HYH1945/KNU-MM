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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

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
    
    # 기본 시스템 프롬프트 (config가 없을 때 사용)
    DEFAULT_SYSTEM_PROMPT = """당신은 음성과 이미지를 함께 분석하는 상황 분석 AI입니다. 객관적으로 상황을 파악하고 분석하세요.

다음을 JSON으로만 반환하세요:
{
  "context": "맥락 설명",
  "urgency": "위급도 (낮음/중간/높음/긴급)",
  "situation": "상황 분석",
  "situation_type": "상황 유형",
  "is_emergency": true/false,
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
        
        # 음성 특성 분석기 초기화
        if VOICE_ANALYSIS_AVAILABLE:
            self.voice_analyzer = VoiceCharacteristicsAnalyzer()
        else:
            self.voice_analyzer = None
        
        # 시스템 프롬프트 (config에서 로드)
        self.system_prompt = get_prompt('system') or self.DEFAULT_SYSTEM_PROMPT
        
        # OpenAI API 설정
        self.max_tokens = get_openai_config('max_tokens', default=800)
        self.temperature = get_openai_config('temperature', default=0.3)
        self.image_detail = get_openai_config('image_detail', default='low')
    
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
            
            # 사용자 메시지 구성
            user_message = f"""**음성 입력:** "{audio_text}"

**분석 요청:** 위 음성과 함께 제공된 이미지를 분석하여 전체 상황을 판단해주세요."""
            
            if additional_context:
                user_message += f"\n\n**추가 정보:** {additional_context}"
            
            # OpenAI API 호출 (Vision 지원)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
                                    "detail": self.image_detail  # config에서 로드
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,  # config에서 로드
                temperature=self.temperature  # config에서 로드
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
            
            # 안전 정책 거부 감지
            if content and ("I'm sorry" in content or "I can't assist" in content or "I cannot" in content):
                print(f"⚠️  OpenAI 안전 정책으로 인한 분석 거부")
                print(f"   원본 응답: {content}")
                return {
                    'context': '이미지 내용을 분석할 수 없음 (안전 정책)',
                    'urgency': '낮음',
                    'urgency_reason': 'OpenAI 안전 정책으로 인한 분석 제한',
                    'situation': '이미지가 안전 정책에 위배되거나 분석이 제한되었습니다.',
                    'situation_type': '분석 제한',
                    'emotional_state': '중립',
                    'visual_content': '분석 불가',
                    'audio_visual_consistency': 'N/A',
                    'action': '다른 이미지로 다시 시도하거나 이미지 내용을 확인하세요',
                    'is_emergency': False,
                    'emergency_reason': None,
                    'priority': 'LOW',
                    'error': '안전 정책 거부',
                    'raw_response': content
                }
            
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
                    'is_emergency': False,
                    'priority': 'LOW'
                }
            
            # 음성 특성 분석으로 신뢰도 검증
            if audio_file_path and self.voice_analyzer:
                try:
                    print("   🎤 음성 특성 분석 중...")
                    audio_features = self.voice_analyzer.extract_features(audio_file_path)
                    
                    confidence_result = self.voice_analyzer.calculate_confidence_score(
                        audio_features=audio_features,
                        llm_priority=result.get('priority', 'LOW'),
                        llm_is_emergency=result.get('is_emergency', False)
                    )
                    
                    # 최종 결과에 음성 분석 정보 추가
                    result['voice_analysis'] = {
                        'voice_emergency_score': confidence_result['voice_emergency_score'],
                        'combined_score': confidence_result['combined_score'],
                        'final_priority': confidence_result['final_priority'],
                        'confidence': confidence_result['confidence'],
                        'indicators': confidence_result['breakdown']['voice_indicators']
                    }
                    
                    # 최종 우선순위 업데이트 (LLM + 음성 특성)
                    original_priority = result.get('priority', 'LOW')
                    final_priority = confidence_result['final_priority']
                    
                    if final_priority != original_priority:
                        result['priority_adjustment'] = {
                            'original': original_priority,
                            'adjusted': final_priority,
                            'reason': f'음성 특성 분석으로 인한 조정 (신뢰도: {confidence_result["confidence"]:.1%})'
                        }
                        result['priority'] = final_priority
                        result['is_emergency'] = final_priority == 'CRITICAL'
                    
                    print(f"      ✅ 음성 분석 완료 (신뢰도: {confidence_result['confidence']:.1%})")
                
                except Exception as e:
                    print(f"      ⚠️  음성 특성 분석 오류: {e}")
            
            return {
                    'error': 'JSON 파싱 실패',
                    'raw_response': content,
                    'context': '분석 오류',
                    'is_emergency': False,
                    'priority': 'LOW'
                }
        
        except Exception as e:
            print(f"❌ 멀티모달 분석 오류: {e}")
            return {
                'error': str(e),
                'context': '분석 실패',
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
