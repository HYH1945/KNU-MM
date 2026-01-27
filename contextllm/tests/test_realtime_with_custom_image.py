#!/usr/bin/env python3
"""
실시간 음성 인식 + 사용자 지정 이미지 멀티모달 분석
마이크로 실시간 입력 + 미리 준비한 이미지를 함께 분석

사용법:
  1. test_images/ 폴더에 분석할 이미지 넣기
  2. python tests/test_realtime_with_custom_image.py
  3. 이미지 선택
  4. 마이크로 말하면 자동으로 선택한 이미지와 함께 분석
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import tempfile

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

import speech_recognition as sr
from core.multimodal_analyzer import MultimodalAnalyzer
from core.alert_manager import get_alert_manager

# 글로벌 alert manager 초기화
alert_manager = get_alert_manager()

# 디렉토리 설정
TEST_IMAGES_DIR = Path(__file__).parent.parent / 'test_images'
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(__file__).parent.parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 실행 시작 시간으로 로그 파일명 생성
SESSION_START = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f'realtime_custom_image_{SESSION_START}.json'

def save_conversation_log(turn, text, image_path, analysis):
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
        "image_path": image_path,
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
print("🎤 실시간 음성 + 사용자 지정 이미지 멀티모달 분석")
print("=" * 70)

# 이미지 폴더 확인
image_files = list(TEST_IMAGES_DIR.glob('*'))
image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]

if not image_files:
    print(f"\n⚠️  {TEST_IMAGES_DIR} 폴더에 이미지가 없습니다!")
    print(f"\n📌 다음 단계를 수행하세요:")
    print(f"   1. 이미지를 {TEST_IMAGES_DIR} 폴더에 복사")
    print(f"   2. 다시 실행")
    exit(1)

print(f"\n✅ {len(image_files)}개의 이미지 파일 발견:")
for i, img_file in enumerate(image_files, 1):
    file_size = img_file.stat().st_size / 1024  # KB
    print(f"   {i}. {img_file.name} ({file_size:.1f} KB)")

# 이미지 선택
print("\n" + "=" * 70)
print("📸 분석에 사용할 이미지를 선택하세요")
print("=" * 70)

selected_image = None

try:
    choice = input(f"\n이미지 번호 선택 (1-{len(image_files)}): ")
    choice_num = int(choice)
    
    if 1 <= choice_num <= len(image_files):
        selected_image = str(image_files[choice_num - 1])
        print(f"✅ 선택된 이미지: {image_files[choice_num - 1].name}")
    else:
        print("❌ 잘못된 번호입니다")
        exit(1)
except ValueError:
    print("❌ 숫자를 입력해주세요")
    exit(1)

print(f"\n💡 이제 마이크로 말하면 '{Path(selected_image).name}' 이미지와 함께 분석됩니다")
print("💡 'quit' 또는 'exit'을 말하면 종료됩니다\n")

# 1. 모듈 로드
print("1️⃣ 모듈 로드 중...")
try:
    multimodal_analyzer = MultimodalAnalyzer()
    recognizer = sr.Recognizer()
    print("   ✅ 모든 모듈 로드 완료")
except Exception as e:
    print(f"   ❌ 모듈 로드 실패: {e}")
    exit(1)

# 2. 무한 루프 모니터링
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
        
        # 임시 오디오 파일 저장 (음성 특성 분석용)
        temp_audio_file = None
        try:
            print("   💾 오디오 데이터 저장 중...")
            
            # WAV 데이터 추출
            wav_data = audio.get_wav_data()
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_audio_file = f.name
                f.write(wav_data)
            
            print(f"      ✅ 임시 파일 저장: {temp_audio_file}")
            
            # 멀티모달 분석 (음성 + 선택한 이미지 + 음성 특성)
            print(f"\n   🤖 멀티모달 분석 중 (음성 + 이미지 + 음성 특성: {Path(selected_image).name})...")
            
            analysis = multimodal_analyzer.analyze_with_image(
                audio_text=text, 
                image_source=selected_image,
                audio_file_path=temp_audio_file  # 음성 특성 분석 포함!
            )
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
            save_conversation_log(turn, text, selected_image, analysis)
            
            print("\n   " + "=" * 65)
            turn += 1
            
        finally:
            # 임시 파일 삭제
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                print(f"   🗑️  임시 파일 삭제됨")
        
        
        
        # 상황 가이드 (참고용)
        if priority in SITUATION_GUIDE:
            guide_info = SITUATION_GUIDE[priority]
            print(f"\n   {guide_info['description']}")
            print(f"   📋 참고 예시: {', '.join(guide_info['examples'])}")
        
        # 💾 기록 저장
        save_conversation_log(turn, text, selected_image, analysis)
        
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
        import traceback
        traceback.print_exc()
        turn += 1

print("\n" + "=" * 70)
print(f"✅ 총 {turn - 1}회 분석 완료")
print(f"📁 사용한 이미지: {Path(selected_image).name}")
print("=" * 70)
