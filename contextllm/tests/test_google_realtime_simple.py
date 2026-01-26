#!/usr/bin/env python3
"""
Google Realtime STT + ChatGPT 분석 통합 테스트
마이크 입력 → Google STT → ChatGPT 분석 → 결과 출력

사용법:
  python tests/test_google_realtime_simple.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

# 로깅 디렉토리 생성
LOG_DIR = Path(__file__).parent.parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 실행 시작 시간으로 로그 파일명 생성
SESSION_START = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f'conversation_history_{SESSION_START}.json'

def save_conversation_log(turn, text, analysis):
    """대화 내용과 분석 결과를 하나의 파일에 기록"""
    # 기존 로그 로드 (있으면)
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    # 새 기록 추가
    log_entry = {
        "turn": turn,
        "timestamp": datetime.now().isoformat(),
        "transcribed_text": text,
        "analysis": analysis,
        "version": "google-realtime"
    }
    history.append(log_entry)
    
    # 파일에 저장
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"   💾 기록 저장 (총 {len(history)}건): {LOG_FILE.name}")

print("=" * 70)
print("🎤 Google Realtime STT + ChatGPT 분석 (Google Cloud 유료 버전)")
print("=" * 70)

# 1. 모듈 로드
print("\n1️⃣ 모듈 로드 중...")
try:
    from stt.google_realtime_analyzer import GoogleRealtimeAnalyzer
    from core.voice_analyzer import VoiceAnalyzer
    print("   ✅ 모든 모듈 로드 완료")
except Exception as e:
    print(f"   ❌ 모듈 로드 실패: {e}")
    exit(1)

# 2. 객체 초기화
print("\n2️⃣ 객체 초기화 중...")
try:
    realtime_analyzer = GoogleRealtimeAnalyzer()
    voice_analyzer = VoiceAnalyzer()
    print("   ✅ 객체 초기화 완료")
except Exception as e:
    print(f"   ❌ 초기화 실패: {e}")
    print("   💡 Google Cloud 자격증명이 필요합니다")
    exit(1)

# 3. 마이크 입력
print("\n3️⃣ 마이크에서 입력 받는 중...")
print("   💬 지금부터 말씀해주세요 (최대 10초)")
try:
    text = realtime_analyzer.listen_and_transcribe()
    
    if not text:
        print("   ⚠️  음성을 감지하지 못했습니다")
        exit(1)
    
    print(f"\n   📝 인식된 텍스트:")
    print(f"      '{text}'")
    
except Exception as e:
    print(f"   ❌ 음성 인식 실패: {e}")
    exit(1)

# 4. ChatGPT 분석
print("\n4️⃣ ChatGPT 분석 중...")
try:
    analysis = voice_analyzer.analyze_with_llm(text)
    print("   ✅ 분석 완료\n")
    
    # 결과 출력
    print("   📊 분석 결과:")
    print(f"      감정: {analysis.get('emotion', '미분류')}")
    print(f"      긴급도: {analysis.get('urgency', '미분류')}")
    print(f"      우선순위: {analysis.get('priority', '미분류')}")
    print(f"      의도: {analysis.get('intent', '미분류')}")
    print(f"      핵심 키워드: {analysis.get('keywords', [])}")
    
    # 💾 기록 저장
    save_conversation_log(1, text, analysis)
    
except Exception as e:
    print(f"   ❌ 분석 실패: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ 테스트 완료!")
print("=" * 70)
