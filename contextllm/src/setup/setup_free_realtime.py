#!/usr/bin/env python3
"""
무료 버전 설정 검증
"""

import subprocess
import sys
import os
from pathlib import Path

def check_module(module_name, package_name=None):
    """모듈 설치 확인"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - 설치 필요: pip install {package_name}")
        return False

def main():
    print("=" * 60)
    print("🧪 무료 버전 설정 검증")
    print("=" * 60)
    
    # 1. 필수 모듈 확인
    print("\n1️⃣ 필수 모듈 확인...")
    modules = [
        ('speech_recognition', 'SpeechRecognition'),
        ('pyaudio', 'pyaudio'),
        ('openai', 'openai'),
        ('dotenv', 'python-dotenv'),
    ]
    
    all_ok = True
    for module, package in modules:
        if not check_module(module, package):
            all_ok = False
    
    # 2. OpenAI API 키 확인
    print("\n2️⃣ OpenAI API 키 확인...")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY 설정됨 (길이: {len(api_key)}자)")
    else:
        print("⚠️  OPENAI_API_KEY가 설정되지 않음")
        print("   설정: export OPENAI_API_KEY='sk-your-key'")
        all_ok = False
    
    # 3. 마이크 확인
    print("\n3️⃣ 마이크 확인...")
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("✅ 마이크 감지됨")
    except Exception as e:
        print(f"⚠️  마이크 오류: {e}")
        all_ok = False
    
    # 최종 결과
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 모든 검증 완료! 테스트 시작 가능")
        print("   python tests/test_free_realtime_simple.py")
    else:
        print("⚠️  일부 항목이 설정되지 않았습니다")
        print("   위의 지시사항을 따라 설정을 완료하세요")
    print("=" * 60)

if __name__ == '__main__':
    main()
