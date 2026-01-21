#!/usr/bin/env python3
"""
온-디맨드 음성 인식 + LLM 컨텍스트 분석 모듈

사용법:
    analyzer = VoiceAnalyzer()
    result = analyzer.transcribe_and_analyze(duration=10)
    print(result['analysis'])
"""

import os
import sys
import json
import subprocess
import re
from datetime import datetime
from pathlib import Path

# 설정 (현재 폴더 기준 상대 경로)
VENV_PYTHON = "./.venv/bin/python3"
WHISPER_SCRIPT = "./whisper_service.py"
RECORDING_DIR = "./recordings"
RESULTS_DIR = "./transcriptions"

class VoiceAnalyzer:
    def __init__(self):
        Path(RECORDING_DIR).mkdir(exist_ok=True)
        Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    def record_audio(self, duration=10, output_file=None):
        """
        Sox를 사용하여 음성 녹음
        
        Args:
            duration: 녹음 시간 (초)
            output_file: 저장할 파일 경로 (None이면 자동 생성)
        
        Returns:
            녹음된 파일 경로 또는 None
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_file = f"{RECORDING_DIR}/audio_{timestamp}.wav"
        
        try:
            print(f"🎤 녹음 중... ({duration}초)")
            subprocess.run([
                'sox', '-d', output_file,
                'rate', '16000',
                'channels', '1',
                'trim', '0', str(duration)
            ], check=True, capture_output=True, timeout=duration + 5)
            
            if os.path.getsize(output_file) > 1000:
                print(f"✅ 녹음 완료: {output_file}")
                return output_file
            else:
                print("❌ 음성이 너무 작음")
                return None
        except subprocess.TimeoutExpired:
            print("❌ 녹음 타임아웃")
            return None
        except Exception as e:
            print(f"❌ 녹음 오류: {e}")
            return None
    
    def transcribe(self, audio_file):
        """
        Whisper로 음성 파일 변환
        
        Args:
            audio_file: 음성 파일 경로
        
        Returns:
            변환된 텍스트 또는 None
        """
        if not os.path.exists(audio_file):
            print(f"❌ 파일을 찾을 수 없음: {audio_file}")
            return None
        
        try:
            print(f"⚙️  Whisper 변환 중...")
            result = subprocess.run(
                [VENV_PYTHON, WHISPER_SCRIPT, audio_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                text = result.stdout.strip()
                print(f"✅ 변환 완료: {text[:50]}...")
                return text
            else:
                print(f"❌ Whisper 오류: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("❌ 변환 타임아웃")
            return None
        except Exception as e:
            print(f"❌ 변환 오류: {e}")
            return None
    
    def analyze_with_llm(self, text, system_prompt=None):
        """
        LLM으로 텍스트 컨텍스트 분석 (Ollama Mistral 사용)
        
        사용자 요구사항:
        1. context (맥락) 확인
        2. urgency (위급도) 판단
        3. situation (상황) 분석
        
        Args:
            text: 분석할 텍스트
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
        
        Returns:
            LLM 분석 결과 (딕셔너리)
        """
        if system_prompt is None:
            system_prompt = """당신은 음성 입력을 분석하는 상황 분석 AI입니다.

다음을 JSON으로만 반환하세요 (다른 텍스트 없이):
{
  "context": "대화의 맥락을 간단히 설명",
  "urgency": "위급도 (낮음/중간/높음/긴급 중 하나)",
  "urgency_reason": "왜 그 위급도인지 간단히",
  "situation": "상황을 2-3줄로 분석",
  "situation_type": "상황 유형 (업무/긴급/일상/정보요청 등)",
  "emotional_state": "감정 상태 (긍정/중립/부정)",
  "action": "권장 즉시 조치",
  "priority": "우선순위 (낮음/중간/높음)"
}"""
        
        try:
            import requests
            
            print(f"🤖 LLM(Mistral) 상황 분석 중...")
            
            # Ollama 요청
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mistral',
                    'prompt': f"{system_prompt}\n\n음성 입력: {text}",
                    'stream': False,
                    'temperature': 0.3  # 분석은 낮은 온도 (일관성)
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                print(f"✅ LLM 분석 완료")
                
                # JSON 파싱 시도
                try:
                    # 응답에서 JSON 부분만 추출
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                    else:
                        analysis = json.loads(response_text)
                    return analysis
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 원본 반환
                    return {'raw': response_text, 'raw_analysis': response_text}
            else:
                print(f"❌ LLM 오류: {response.status_code}")
                print(f"   응답: {response.text[:200]}")
                return {'error': f'LLM 서버 오류 ({response.status_code})'}
        
        except requests.exceptions.ConnectionError:
            print("❌ LLM 서버에 연결할 수 없음 (localhost:11434)")
            print("   이미 실행 중이면: ollama serve가 background에서 실행 중")
            print("   또는 새 터미널에서: ollama serve")
            return {'error': 'LLM 서버 미연결', 'suggestion': 'ollama serve 확인'}
        except ImportError:
            print("❌ requests 라이브러리가 필요합니다")
            print("   설치: pip install requests")
            return {'error': 'requests 라이브러리 필요'}
        except Exception as e:
            print(f"❌ LLM 분석 오류: {e}")
            return {'error': str(e)}
    
    def transcribe_and_analyze(self, duration=10, system_prompt=None):
        """
        음성 녹음 → 변환 → LLM 분석을 한 번에 수행
        
        Args:
            duration: 녹음 시간 (초)
            system_prompt: LLM 시스템 프롬프트
        
        Returns:
            {
                'success': bool,
                'timestamp': str,
                'transcribed_text': str,
                'analysis': dict,
                'audio_file': str
            }
        """
        timestamp = datetime.now().isoformat()
        
        # 1️⃣ 녹음
        audio_file = self.record_audio(duration)
        if not audio_file:
            return {
                'success': False,
                'timestamp': timestamp,
                'error': '녹음 실패'
            }
        
        # 2️⃣ 변환
        transcribed_text = self.transcribe(audio_file)
        if not transcribed_text:
            return {
                'success': False,
                'timestamp': timestamp,
                'audio_file': audio_file,
                'error': '변환 실패'
            }
        
        # 3️⃣ LLM 분석
        analysis = self.analyze_with_llm(transcribed_text, system_prompt)
        
        # 4️⃣ 결과 저장
        self.save_result(timestamp, transcribed_text, analysis)
        
        # 5️⃣ 정리
        try:
            os.remove(audio_file)
        except:
            pass
        
        return {
            'success': True,
            'timestamp': timestamp,
            'transcribed_text': transcribed_text,
            'analysis': analysis,
            'audio_file': audio_file
        }
    
    def save_result(self, timestamp, text, analysis):
        """결과를 파일에 저장"""
        date_folder = f"{RESULTS_DIR}/{datetime.now().strftime('%Y-%m-%d')}"
        Path(date_folder).mkdir(exist_ok=True)
        
        entry = {
            'timestamp': timestamp,
            'text': text,
            'analysis': analysis,
            'model': 'whisper-base',
            'language': 'ko'
        }
        
        # JSON 누적
        json_file = f"{date_folder}/transcriptions.json"
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.append(entry)
        else:
            data = [entry]
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 최신 파일
        latest_file = f"{date_folder}/latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)


# 테스트 코드
if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    
    # 10초 녹음 + 분석
    result = analyzer.transcribe_and_analyze(duration=10)
    
    print("\n" + "="*50)
    print("📊 최종 결과:")
    print("="*50)
    print(json.dumps(result, ensure_ascii=False, indent=2))
