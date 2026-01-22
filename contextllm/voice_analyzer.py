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
import threading
import queue
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
            duration: 녹음 시간 (초). duration=None이면 Enter 키까지 무한 녹음
            output_file: 저장할 파일 경로 (None이면 자동 생성)
        
        Returns:
            녹음된 파일 경로 또는 None
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_file = f"{RECORDING_DIR}/audio_{timestamp}.wav"
        
        try:
            if duration is None:
                # 🎤 무한 녹음 모드 (Enter로 종료)
                import threading
                print("🎤 무한 녹음 시작... (Enter 키를 누르면 종료)")
                
                # 무한 녹음 프로세스 시작 (매우 큰 duration 값 사용)
                sox_process = subprocess.Popen([
                    'sox', '-d', output_file,
                    'rate', '16000',
                    'channels', '1'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Enter 입력 대기
                try:
                    input()  # 사용자가 Enter를 칠 때까지 대기
                    print("⏹️  녹음 중지 중...")
                    sox_process.terminate()
                    sox_process.wait(timeout=3)
                except KeyboardInterrupt:
                    sox_process.terminate()
                    sox_process.wait(timeout=3)
                
                # 파일 크기 확인
                if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                    print(f"✅ 녹음 완료: {output_file}")
                    return output_file
                else:
                    print("❌ 음성이 너무 작음 또는 오류 발생")
                    return None
            else:
                # ⏱️ 고정 시간 녹음 모드
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
    
    def run_continuously(self, interval=10, max_iterations=None, system_prompt=None):
        """
        10초 간격으로 음성 인식 + 분석을 반복 실행 (거의 실시간 처리)
        
        Args:
            interval: 반복 간격 (초). 기본 10초
            max_iterations: 최대 반복 횟수. None이면 무한 반복 (Ctrl+C로 종료)
            system_prompt: LLM 시스템 프롬프트
        
        Example:
            analyzer = VoiceAnalyzer()
            # 10초 간격으로 무한 반복 (거의 실시간)
            analyzer.run_continuously(interval=10)
            
            # 또는 5번만 반복
            analyzer.run_continuously(interval=10, max_iterations=5)
        
        Returns:
            결과 리스트 (완료 시에만 반환, 무한 반복 시 반환 안 됨)
        """
        import time
        
        results = []
        iteration = 0
        
        print(f"\n{'='*60}")
        print(f"🔄 실시간 모니터링 시작 (10초 간격)")
        print(f"{'='*60}\n")
        
        try:
            while True:
                iteration += 1
                
                # 최대 반복 횟수 확인
                if max_iterations and iteration > max_iterations:
                    print(f"\n✅ {max_iterations}회 완료!")
                    break
                
                print(f"\n[{iteration}차] 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 한 사이클 실행 (10초 녹음 + 분석)
                result = self.transcribe_and_analyze(
                    duration=interval, 
                    system_prompt=system_prompt
                )
                
                results.append(result)
                
                # 결과 출력
                if result.get('success'):
                    print(f"📝 음성: {result['transcribed_text'][:50]}...")
                    if result.get('analysis'):
                        analysis = result['analysis']
                        if 'urgency' in analysis:
                            print(f"🚨 위급도: {analysis['urgency']}")
                        if 'emotional_state' in analysis:
                            print(f"😊 감정: {analysis['emotional_state']}")
                else:
                    print(f"❌ 오류: {result.get('error', '알 수 없는 오류')}")
                
                print(f"⏳ {interval}초 후 다시 실행...")
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"⏹️  모니터링 중지됨 (Ctrl+C)")
            print(f"{'='*60}")
            print(f"\n📊 총 {iteration}회 처리 완료")
            print(f"✅ 성공: {sum(1 for r in results if r.get('success'))}")
            print(f"❌ 실패: {sum(1 for r in results if not r.get('success'))}")
        
        return results
    
    def run_parallel_realtime(self, interval=10, max_iterations=None, system_prompt=None):
        """
        병렬 처리로 거의 진정한 실시간 음성 인식 (녹음 + 분석 동시 진행)
        
        장점:
        - 🎤 스레드 1: 계속 녹음 (10초)
        - 🧠 스레드 2: 동시에 이전 녹음 변환/분석
        - ⚡ 결과: 약 10초 간격으로 완료 (15-20초 아님!)
        
        Args:
            interval: 반복 간격 (초). 기본 10초
            max_iterations: 최대 반복 횟수. None이면 무한 반복
            system_prompt: LLM 시스템 프롬프트
        
        Example:
            analyzer = VoiceAnalyzer()
            # 진정한 거의 실시간 처리 (병렬)
            analyzer.run_parallel_realtime(interval=10)
        """
        import time
        
        results = []
        audio_queue = queue.Queue()  # 녹음된 파일을 저장할 큐
        
        print(f"\n{'='*60}")
        print(f"⚡ 병렬 처리 실시간 모니터링 시작 (진정한 실시간!)")
        print(f"{'='*60}\n")
        
        # 스레드 1: 백그라운드 녹음
        def recording_thread():
            iteration = 0
            try:
                while True:
                    iteration += 1
                    if max_iterations and iteration > max_iterations:
                        audio_queue.put(None)  # 종료 신호
                        break
                    
                    print(f"\n[녹음 {iteration}차] 시간: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"  🎤 {interval}초 녹음 중...")
                    
                    # 녹음 실행
                    audio_file = self.record_audio(duration=interval)
                    
                    if audio_file:
                        print(f"  ✅ 녹음 저장: {audio_file}")
                        audio_queue.put(audio_file)  # 큐에 추가
                    else:
                        print(f"  ⚠️  녹음 실패")
            
            except KeyboardInterrupt:
                audio_queue.put(None)
        
        # 스레드 2: 변환 + 분석
        def analysis_thread():
            analysis_count = 0
            try:
                while True:
                    # 큐에서 녹음 파일 대기
                    audio_file = audio_queue.get(timeout=30)
                    
                    if audio_file is None:  # 종료 신호
                        break
                    
                    analysis_count += 1
                    print(f"\n[분석 {analysis_count}차] 시간: {datetime.now().strftime('%H:%M:%S')}")
                    
                    # 변환
                    print(f"  📝 Whisper 변환 중...")
                    transcribed_text = self.transcribe(audio_file)
                    
                    if transcribed_text:
                        print(f"  ✅ 음성: {transcribed_text[:60]}...")
                        
                        # 분석
                        print(f"  🧠 LLM 분석 중...")
                        analysis = self.analyze_with_llm(transcribed_text, system_prompt)
                        
                        # 저장
                        timestamp = datetime.now().isoformat()
                        self.save_result(timestamp, transcribed_text, analysis)
                        
                        # 결과 출력
                        if analysis.get('urgency'):
                            print(f"  🚨 위급도: {analysis['urgency']}")
                        if analysis.get('emotional_state'):
                            print(f"  😊 감정: {analysis['emotional_state']}")
                        
                        results.append({
                            'success': True,
                            'timestamp': timestamp,
                            'transcribed_text': transcribed_text,
                            'analysis': analysis
                        })
                    else:
                        print(f"  ❌ 변환 실패")
                        results.append({
                            'success': False,
                            'error': '변환 실패'
                        })
                    
                    # 파일 정리
                    try:
                        os.remove(audio_file)
                    except:
                        pass
            
            except queue.Empty:
                print("  ⚠️  녹음 큐 타임아웃")
            except KeyboardInterrupt:
                pass
        
        try:
            # 두 스레드 동시 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                rec_thread = executor.submit(recording_thread)
                ana_thread = executor.submit(analysis_thread)
                
                # 두 스레드 완료 대기
                rec_thread.result()
                ana_thread.result()
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print(f"⏹️  병렬 처리 중지됨 (Ctrl+C)")
            print(f"{'='*60}")
        
        print(f"\n📊 처리 완료")
        print(f"✅ 성공: {sum(1 for r in results if r.get('success'))}")
        print(f"❌ 실패: {sum(1 for r in results if not r.get('success'))}")
        
        return results


# 테스트 코드
# 테스트 코드
if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    
    print("음성 입력 모드를 선택하세요:")
    print("1. 고정 시간 녹음 (10초) - 일회성")
    print("2. 무한 녹음 (Enter로 종료) - 일회성")
    print("3. 실시간 모니터링 (순차 처리) - 약 15-20초 간격")
    print("4. 병렬 처리 모니터링 (진정한 실시간!) ⭐ 약 10초 간격")
    choice = input("선택 (1, 2, 3, 또는 4): ").strip()
    
    if choice == "2":
        # 무한 녹음 모드
        result = analyzer.transcribe_and_analyze(duration=None)
        print("\n" + "="*50)
        print("📊 최종 결과:")
        print("="*50)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif choice == "3":
        # 순차 처리 실시간 모니터링
        print("\n몇 회를 반복할까요?")
        print("1. 무한 반복 (Ctrl+C로 중지)")
        print("2. 5회만 실행")
        print("3. 10회 실행")
        repeat_choice = input("선택 (1, 2, 또는 3): ").strip()
        
        if repeat_choice == "2":
            analyzer.run_continuously(interval=10, max_iterations=5)
        elif repeat_choice == "3":
            analyzer.run_continuously(interval=10, max_iterations=10)
        else:
            analyzer.run_continuously(interval=10)
    
    elif choice == "4":
        # 병렬 처리 (진정한 실시간)
        print("\n몇 회를 반복할까요?")
        print("1. 무한 반복 (Ctrl+C로 중지) ⭐ 권장")
        print("2. 5회만 실행")
        print("3. 10회 실행")
        repeat_choice = input("선택 (1, 2, 또는 3): ").strip()
        
        if repeat_choice == "2":
            analyzer.run_parallel_realtime(interval=10, max_iterations=5)
        elif repeat_choice == "3":
            analyzer.run_parallel_realtime(interval=10, max_iterations=10)
        else:
            analyzer.run_parallel_realtime(interval=10)
    
    else:
        # 기본: 1번을 선택했거나 잘못된 입력을 했을 경우 10초 녹음 (일회성)
        print("\n[기본 모드] 10초 동안 녹음을 시작합니다...")
        result = analyzer.transcribe_and_analyze(duration=10)
        print("\n" + "="*50)
        print("📊 최종 결과:")
        print("="*50)
        print(json.dumps(result, ensure_ascii=False, indent=2))