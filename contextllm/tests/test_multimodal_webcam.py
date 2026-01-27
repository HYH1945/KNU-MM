#!/usr/bin/env python3
"""
멀티모달 컨텍스트 분석 - 웹캠 버전
음성 + 웹캠 실시간 영상을 함께 분석하여 더 정확한 상황 판단

사용법:
  python tests/test_multimodal_webcam.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv

# .env 파일 로드 (config 폴더에서)
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

import speech_recognition as sr
from core.multimodal_analyzer import MultimodalAnalyzer
from core.video_capture import VideoMonitor
from core.alert_manager import get_alert_manager

# 글로벌 alert manager 초기화
alert_manager = get_alert_manager()

# 로깅 디렉토리 생성
LOG_DIR = Path(__file__).parent.parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 웹캠 프레임 디렉토리 생성
WEBCAM_DIR = Path(__file__).parent.parent / 'webcam_frames'
WEBCAM_DIR.mkdir(parents=True, exist_ok=True)

# 실행 시작 시간으로 로그 파일명 생성
SESSION_START = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f'multimodal_webcam_history_{SESSION_START}.json'

# 최신 웹캠 프레임 저장용
latest_frame = None
latest_frame_time = None

def save_conversation_log(turn, text, frame_path, analysis):
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
        "webcam_frame_path": frame_path,
        "analysis": analysis
    }
    history.append(log_entry)
    
    # 파일에 저장
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"   💾 기록 저장 (총 {len(history)}건): {LOG_FILE.name}")

def on_webcam_frame(frame, timestamp):
    """웹캠 프레임 콜백"""
    global latest_frame, latest_frame_time
    latest_frame = frame.copy()
    latest_frame_time = timestamp

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
print("🎥 멀티모달 분석 (음성 + 웹캠)")
print("=" * 70)
print("💡 팁: 말할 때 웹캠에서 실시간 프레임을 캡처하여 함께 분석합니다")
print("💡 팁: 'quit' 또는 'exit'을 말하면 종료됩니다\n")

# 1. 모듈 로드
print("1️⃣ 모듈 로드 중...")
try:
    multimodal_analyzer = MultimodalAnalyzer()
    recognizer = sr.Recognizer()
    video_monitor = VideoMonitor(camera_id=0)
    print("   ✅ 모든 모듈 로드 완료")
except Exception as e:
    print(f"   ❌ 모듈 로드 실패: {e}")
    exit(1)

# 2. 웹캠 모니터링 시작
print("\n2️⃣ 웹캠 모니터링 시작...")
try:
    video_monitor.start_monitoring(
        on_frame_callback=on_webcam_frame,
        frame_interval=0.5,  # 0.5초마다 프레임 업데이트
        show_preview=False  # 프리뷰 창 비활성화 (필요시 True로 변경)
    )
    
    # 웹캠이 준비될 때까지 대기
    print("   ⏳ 웹캠 준비 중...")
    for _ in range(10):
        if latest_frame is not None:
            break
        time.sleep(0.5)
    
    if latest_frame is None:
        print("   ⚠️  웹캠 프레임을 받을 수 없습니다. 계속 진행하지만 이미지 없이 음성만 분석됩니다.")
    else:
        print("   ✅ 웹캠 준비 완료")
except Exception as e:
    print(f"   ⚠️  웹캠 시작 실패: {e}")
    print("   계속 진행하지만 이미지 없이 음성만 분석됩니다.")

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
        
        # 웹캠 프레임 저장
        frame_path = None
        if latest_frame is not None:
            print("\n   📹 웹캠 프레임 캡처 중...")
            frame_path = str(WEBCAM_DIR / f"webcam_{SESSION_START}_{turn:03d}.jpg")
            
            import cv2
            cv2.imwrite(frame_path, latest_frame)
            print(f"      ✅ 웹캠 프레임 저장: {frame_path}")
        else:
            print("\n   ⚠️  웹캠 프레임 없음 (음성만으로 분석)")
        
        # 멀티모달 분석 (음성 + 웹캠)
        print("\n   🤖 멀티모달 분석 중 (음성 + 비디오)...")
        if latest_frame is not None:
            analysis = multimodal_analyzer.analyze_with_video_frame(text, latest_frame)
        else:
            # 웹캠 프레임 없을 시 음성만 분석 (폴백)
            from core.voice_analyzer import VoiceAnalyzer
            voice_analyzer = VoiceAnalyzer()
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
        
        # 상황 분석
        situation_text = analysis.get('situation', '미분류')
        print(f"   🎯 상황 분석:")
        print(f"      {situation_text}")
        
        # 시각적 내용
        visual_content = analysis.get('visual_content', 'N/A')
        if visual_content and visual_content != 'N/A':
            print(f"\n   👁️  시각 정보:")
            print(f"      {visual_content}")
        
        # 음성-시각 일치도
        consistency = analysis.get('audio_visual_consistency', 'N/A')
        if consistency and consistency != 'N/A':
            consistency_emoji = {
                '일치': '✅',
                '불일치': '⚠️',
                '부분일치': '🔶'
            }.get(consistency, '❓')
            print(f"\n   {consistency_emoji} 음성-시각 일치도: {consistency}")
        
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
        save_conversation_log(turn, text, frame_path, analysis)
        
        print("\n   " + "=" * 65)
        turn += 1
        
    except sr.UnknownValueError:
        print("   ⚠️  음성을 인식하지 못했습니다. 다시 시도해주세요.")
        turn += 1
    except sr.RequestError as e:
        print(f("   ❌ 음성 인식 서비스 오류: {e}")
        break
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다")
        break
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        turn += 1

# 웹캠 모니터링 중지
print("\n웹캠 모니터링 중지 중...")
try:
    video_monitor.stop_monitoring()
except:
    pass

print("\n" + "=" * 70)
print(f"✅ 총 {turn - 1}회 분석 완료")
print("=" * 70)
