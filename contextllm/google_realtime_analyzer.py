#!/usr/bin/env python3
"""
Google Cloud Speech-to-Text 실시간 스트리밍 모듈
문장마다 LLM 분석 (진정한 실시간!)

설치:
    pip install google-cloud-speech pyaudio
    
인증:
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    from google.cloud import speech_v1
    from google.api_core.gapic_v1 import client_info as grpc_client_info
    import pyaudio
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Cloud Speech 라이브러리가 설치되지 않았습니다.")
    print("   설치: pip install google-cloud-speech pyaudio")

class GoogleRealtimeAnalyzer:
    """Google Cloud Speech-to-Text + 실시간 LLM 분석"""
    
    def __init__(self, ollama_url="http://localhost:11434"):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-cloud-speech와 pyaudio가 필요합니다")
        
        self.client = speech_v1.SpeechClient()
        self.ollama_url = ollama_url
        self.results_dir = "./transcriptions"
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # 부분 결과 저장소
        self.interim_results = deque(maxlen=10)
        self.final_results = []
    
    def stream_audio(self, sample_rate=16000, chunk_duration=0.1):
        """마이크에서 실시간 음성 스트리밍"""
        audio = pyaudio.PyAudio()
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=int(sample_rate * chunk_duration)
        )
        
        try:
            print("🎤 마이크 입력 중... (Ctrl+C로 종료)")
            while True:
                chunk = stream.read(int(sample_rate * chunk_duration))
                if chunk:
                    yield chunk
        except KeyboardInterrupt:
            print("\n⏹️  녹음 중지")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
    
    def analyze_sentence(self, text, system_prompt=None):
        """문장을 LLM으로 분석"""
        try:
            analysis_prompt = system_prompt or """
다음 문장을 분석하여 JSON으로 반환하세요:
{
  "emotion": "긍정/중립/부정",
  "urgency": "낮음/중간/높음/긴급",
  "intent": "의도 요약",
  "keywords": ["키워드1", "키워드2"]
}
"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "mistral",
                    "prompt": f"{analysis_prompt}\n\n문장: {text}",
                    "stream": False,
                    "format": "json"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                try:
                    analysis = json.loads(result.get('response', '{}'))
                    return analysis
                except:
                    return {"raw": result.get('response', '')}
            return {"error": "LLM 분석 실패"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def listen_and_analyze_realtime(self, system_prompt=None, max_duration=None):
        """
        실시간으로 음성을 듣고 문장마다 분석 (진정한 실시간!)
        
        Args:
            system_prompt: LLM 분석 프롬프트
            max_duration: 최대 실행 시간 (초)
        
        Returns:
            분석 결과 리스트
        """
        import time
        
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            enable_automatic_punctuation=True,
            max_alternatives=1,
        )
        
        streaming_config = speech_v1.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )
        
        results = []
        start_time = time.time()
        current_sentence = ""
        last_analysis_time = 0
        
        print(f"\n{'='*60}")
        print("⚡ Google Speech-to-Text 실시간 모니터링")
        print("문장이 완성되면 자동으로 LLM 분석")
        print(f"{'='*60}\n")
        
        try:
            # 음성 스트림 생성
            requests_gen = (
                speech_v1.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in self.stream_audio()
            )
            
            # 실시간 인식
            responses = self.client.streaming_recognize(
                streaming_config,
                requests_gen
            )
            
            for response in responses:
                # 최대 시간 체크
                if max_duration and (time.time() - start_time) > max_duration:
                    print(f"\n⏰ {max_duration}초 경과 - 종료")
                    break
                
                if not response.results:
                    continue
                
                result = response.results[0]
                
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    
                    if result.is_final:
                        # 🟢 최종 결과 (문장 완성)
                        print(f"\n✅ [최종] {transcript}")
                        
                        # 문장이 끝났으므로 분석 실행
                        current_sentence += transcript
                        
                        print(f"🧠 LLM 분석 중...")
                        analysis = self.analyze_sentence(current_sentence, system_prompt)
                        
                        # 결과 출력
                        print(f"📊 분석 결과:")
                        if 'error' not in analysis:
                            if 'emotion' in analysis:
                                print(f"  😊 감정: {analysis['emotion']}")
                            if 'urgency' in analysis:
                                print(f"  🚨 위급도: {analysis['urgency']}")
                            if 'intent' in analysis:
                                print(f"  💭 의도: {analysis['intent']}")
                            if 'keywords' in analysis:
                                print(f"  🏷️  키워드: {', '.join(analysis['keywords'])}")
                        else:
                            print(f"  ❌ {analysis['error']}")
                        
                        # 결과 저장
                        entry = {
                            'timestamp': datetime.now().isoformat(),
                            'text': current_sentence,
                            'analysis': analysis
                        }
                        results.append(entry)
                        self.final_results.append(entry)
                        
                        # 다음 문장 준비
                        current_sentence = ""
                    else:
                        # 🟡 부분 결과 (입력 중...)
                        print(f"\r⏳ [입력중] {transcript[:60]}...", end="", flush=True)
                        current_sentence = transcript
        
        except Exception as e:
            print(f"\n❌ 오류: {e}")
        
        finally:
            # 결과 저장
            self._save_results(results)
            
            print(f"\n\n{'='*60}")
            print(f"📊 처리 완료")
            print(f"✅ 분석된 문장: {len(results)}개")
            print(f"{'='*60}")
        
        return results
    
    def _save_results(self, results):
        """결과를 파일에 저장"""
        if not results:
            return
        
        date_folder = f"{self.results_dir}/{datetime.now().strftime('%Y-%m-%d')}"
        Path(date_folder).mkdir(exist_ok=True)
        
        # JSON 저장
        json_file = f"{date_folder}/google_realtime_{datetime.now().strftime('%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장: {json_file}")


# 테스트
if __name__ == "__main__":
    if not GOOGLE_AVAILABLE:
        print("❌ Google Cloud Speech-to-Text이 설치되지 않았습니다.")
        print("\n설치 방법:")
        print("  1. pip install google-cloud-speech pyaudio")
        print("  2. Google Cloud 인증: export GOOGLE_APPLICATION_CREDENTIALS=...")
        exit(1)
    
    analyzer = GoogleRealtimeAnalyzer()
    
    print("Google Cloud Speech-to-Text 실시간 분석 시작")
    print("=" * 60)
    print("마이크로 말하세요. 문장 단위로 실시간 분석됩니다.")
    print("Ctrl+C로 종료")
    print("=" * 60)
    
    # 실시간 분석 실행 (최대 60초)
    results = analyzer.listen_and_analyze_realtime(max_duration=60)
    
    print("\n최종 결과:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['text']}")
        print(f"   분석: {result['analysis']}")
