#!/usr/bin/env python3
"""
무료 SpeechRecognition + ChatGPT 분석 통합 테스트
마이크 입력 → SpeechRecognition (무료) → ChatGPT 분석 → 결과 출력

사용법:
  python tests/test_free_realtime_simple.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv

# .env 파일 로드 (config 폴더에서)
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

import speech_recognition as sr
from core.alert_manager import get_alert_manager

# 글로벌 alert manager 초기화
alert_manager = get_alert_manager()

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
        "analysis": analysis
    }
    history.append(log_entry)
    
    # 파일에 저장
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"   💾 기록 저장 (총 {len(history)}건): {LOG_FILE.name}")

# 우선순위별 상황 설명
SITUATION_GUIDE = {
    'CRITICAL': {
        'description': '🚨 긴급 상황 - 즉시 대응 필요',
        'examples': ['서버 다운', '보안 침해', '시스템 오류'],
    },
    'HIGH': {
        'description': '⚠️ 높은 우선순위 - 빠른 대응 필요',
        'examples': ['성능 저하', '사용자 불만', '오류 발생'],
    },
    'MEDIUM': {
        'description': '📌 중간 우선순위 - 일반적 대응',
        'examples': ['일반 질문', '정보 제공', '기타'],
    },
    'LOW': {
        'description': 'ℹ️ 낮은 우선순위 - 배경 정보',
        'examples': ['인사', '일반 대화', '참고 사항'],
    }
}

print("=" * 70)
print("🎤 무료 SpeechRecognition + ChatGPT 분석 (무한 모드)")
print("=" * 70)
print("💡 팁: 'quit' 또는 'exit'을 말하면 종료됩니다\n")

# 1. 모듈 로드
print("1️⃣ 모듈 로드 중...")
try:
    from core.voice_analyzer import VoiceAnalyzer
    from core.priority_manager import PriorityQueue, PriorityLevel
    print("   ✅ 모든 모듈 로드 완료")
except Exception as e:
    print(f"   ❌ 모듈 로드 실패: {e}")
    exit(1)

# 2. 객체 초기화
print("\n2️⃣ 객체 초기화 중...")
try:
    voice_analyzer = VoiceAnalyzer()
    recognizer = sr.Recognizer()
    print("   ✅ 객체 초기화 완료")
except Exception as e:
    print(f"   ❌ 초기화 실패: {e}")
    exit(1)

# 3. 무한 루프 모니터링
print("\n" + "=" * 70)
print("✅ 시작 준비 완료! 마이크로 말씀해주세요")
print("=" * 70 + "\n")

turn = 1
while True:
    try:
        print(f"\n📍 [회차 {turn}] 마이크에서 입력 받는 중...")
        print("   💬 지금부터 말씀해주세요 ('quit' 또는 'exit'으로 종료)")
        
        with sr.Microphone() as source:
            # 배경 소음 적응
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # 음성 입력 (최대 30초)
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=30)
        
        # Google Speech Recognition (무료)
        print("   🔄 음성을 텍스트로 변환 중...")
        text = recognizer.recognize_google(audio, language='ko-KR')
        
        # 종료 조건
        if text.lower() in ['quit', 'exit', '종료', '끝']:
            print("\n👋 프로그램을 종료합니다")
            break
        
        print(f"\n   📝 인식된 텍스트:")
        print(f"      '{text}'")
        
        # ChatGPT 분석
        print("\n   🤖 ChatGPT 분석 중...")
        analysis = voice_analyzer.analyze_with_llm(text)
        print("      ✅ 분석 완료\n")
        
        priority = analysis.get('priority', 'LOW')
        is_emergency = analysis.get('is_emergency', False)
        
        # 🚨 긴급 상황 감지 및 알림 (alert_manager 사용)
        if is_emergency or priority == 'CRITICAL':
            alert_manager.trigger_alert({
                'is_emergency': is_emergency,
                'emergency_reason': analysis.get('emergency_reason', '알 수 없는 긴급 상황'),
                'priority': priority,
                'situation_type': analysis.get('situation_type', '미분류')
            })
        
        # 상황 설명
        situation_text = analysis.get('situation', '미분류')
        print(f"   🎯 상황 분석:")
        print(f"      {situation_text}")
        
        situation_type = analysis.get('situation_type', '미분류')
        print(f"\n   📌 상황 유형: {situation_type}")
        
        action = analysis.get('action', '미분류')
        print(f"   🔧 권장 조치: {action}")
        
        # 상황 가이드 (참고용)
        if priority in SITUATION_GUIDE:
            guide_info = SITUATION_GUIDE[priority]
            print(f"\n   {guide_info['description']}")
            print(f"   📋 참고 예시: {', '.join(guide_info['examples'])}")
        
        # 💾 기록 저장
        save_conversation_log(turn, text, analysis)
        
        print("\n   " + "=" * 65)
        turn += 1
        
    except sr.UnknownValueError:
        print("   ⚠️  음성을 인식하지 못했습니다. 다시 시도해주세요.")
        turn += 1
    except sr.RequestError as e:
        print(f"   ❌ 음성 인식 서비스 오류: {e}")
        break
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다")
        break
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        turn += 1

print("\n" + "=" * 70)
print(f"✅ 총 {turn - 1}회 분석 완료")
print("=" * 70)
