#!/usr/bin/env python3
"""
Google Realtime STT 분석기
"""

from google.cloud import speech_v1
import pyaudio
from queue import Queue

class GoogleRealtimeAnalyzer:
    """Google Cloud Speech-to-Text 실시간 분석"""
    
    def __init__(self):
        self.client = speech_v1.SpeechClient()
        self.config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            streaming_config=speech_v1.StreamingRecognitionConfig(
                config=speech_v1.RecognitionConfig(
                    encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="ko-KR",
                ),
                interim_results=True,
            ),
        )
    
    def listen_and_transcribe(self, duration=10):
        """마이크에서 음성을 인식"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            
            text = recognizer.recognize_google(audio, language='ko-KR')
            return text
        except Exception as e:
            print(f"오류: {e}")
            return None
