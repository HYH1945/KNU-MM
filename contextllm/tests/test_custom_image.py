#!/usr/bin/env python3
"""
사용자 지정 이미지로 멀티모달 분석 테스트
직접 캡처한 이미지를 사용하여 음성+이미지 분석 검증

사용법:
  1. 이미지를 test_images/ 폴더에 저장
  2. 스크립트 실행: python tests/test_custom_image.py
  3. 이미지 파일명과 테스트할 음성 텍스트 입력
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

from core.multimodal_analyzer import MultimodalAnalyzer
from core.alert_manager import get_alert_manager

# 디렉토리 설정
TEST_IMAGES_DIR = Path(__file__).parent.parent / 'test_images'
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(__file__).parent.parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Alert manager 초기화
alert_manager = get_alert_manager()

print("=" * 70)
print("🎨 사용자 지정 이미지 멀티모달 분석 테스트")
print("=" * 70)
print(f"\n📁 이미지 저장 폴더: {TEST_IMAGES_DIR}")
print("💡 팁: 테스트할 이미지를 위 폴더에 넣어주세요\n")

# 이미지 폴더 확인
image_files = list(TEST_IMAGES_DIR.glob('*'))
image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]

if image_files:
    print(f"✅ {len(image_files)}개의 이미지 파일 발견:")
    for i, img_file in enumerate(image_files, 1):
        file_size = img_file.stat().st_size / 1024  # KB
        print(f"   {i}. {img_file.name} ({file_size:.1f} KB)")
else:
    print("⚠️  이미지 파일이 없습니다!")
    print(f"\n📌 다음 중 하나를 수행하세요:")
    print(f"   1. 이미지를 {TEST_IMAGES_DIR} 폴더에 복사")
    print(f"   2. 또는 아래에서 절대 경로 입력")

print("\n" + "=" * 70)

# 멀티모달 분석기 초기화
print("\n1️⃣ 멀티모달 분석기 로드 중...")
try:
    analyzer = MultimodalAnalyzer()
    print("   ✅ 분석기 로드 완료")
except Exception as e:
    print(f"   ❌ 오류: {e}")
    exit(1)

# 이미지 경로 입력
print("\n2️⃣ 이미지 선택")
print("=" * 70)

image_path = None

if image_files:
    print("\n옵션 1: 번호로 선택")
    for i, img_file in enumerate(image_files, 1):
        print(f"   {i}. {img_file.name}")
    print(f"   0. 다른 경로 직접 입력")
    
    try:
        choice = input("\n이미지 번호 선택 (1-{}, 0=직접입력): ".format(len(image_files)))
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(image_files):
            image_path = str(image_files[choice_num - 1])
            print(f"✅ 선택: {image_files[choice_num - 1].name}")
        elif choice_num == 0:
            image_path = input("이미지 전체 경로 입력: ").strip()
        else:
            print("❌ 잘못된 번호입니다")
            exit(1)
    except ValueError:
        print("❌ 숫자를 입력해주세요")
        exit(1)
else:
    image_path = input("\n이미지 전체 경로 입력: ").strip()

# 경로 검증
if not Path(image_path).exists():
    print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
    exit(1)

print(f"\n📸 선택된 이미지: {image_path}")

# 음성 텍스트 입력
print("\n3️⃣ 음성 텍스트 입력")
print("=" * 70)
print("이미지와 함께 분석할 음성 텍스트를 입력하세요")
print("예시:")
print("  - '도와주세요!'")
print("  - '지금 이 화면이 뭔가요?'")
print("  - '이상한 소리가 들려요'")
print("  - '불이야!'")
print()

audio_text = input("음성 텍스트: ").strip()

if not audio_text:
    print("❌ 텍스트를 입력해주세요")
    exit(1)

print(f"\n💬 입력된 텍스트: '{audio_text}'")

# 추가 컨텍스트 (선택사항)
print("\n4️⃣ 추가 컨텍스트 (선택사항)")
print("=" * 70)
print("분석에 도움이 될 추가 정보가 있으면 입력하세요 (없으면 Enter)")
print("예시: '재택근무 중', '밤 10시', '혼자 있음' 등")
print()

additional_context = input("추가 정보 (선택): ").strip()

# 멀티모달 분석 실행
print("\n5️⃣ 분석 실행")
print("=" * 70)
print("🤖 GPT-4o로 멀티모달 분석 중...")
print("   (음성 텍스트 + 이미지 내용 종합 분석)")
print()

try:
    result = analyzer.analyze_with_image(
        audio_text=audio_text,
        image_source=image_path,
        additional_context=additional_context if additional_context else None
    )
    
    print("✅ 분석 완료!\n")
    
    # 결과 출력
    print("=" * 70)
    print("📊 분석 결과")
    print("=" * 70)
    
    # 우선순위 및 긴급도
    priority = result.get('priority', 'LOW')
    is_emergency = result.get('is_emergency', False)
    
    priority_emoji = {
        'CRITICAL': '🚨',
        'HIGH': '⚠️',
        'MEDIUM': '📌',
        'LOW': 'ℹ️'
    }.get(priority, '❓')
    
    print(f"\n{priority_emoji} 우선순위: {priority}")
    print(f"{'🚨' if is_emergency else '✅'} 긴급 상황: {'예' if is_emergency else '아니오'}")
    
    if is_emergency:
        emergency_reason = result.get('emergency_reason', 'N/A')
        print(f"   🔴 긴급 사유: {emergency_reason}")
    
    # 상황 분석
    context = result.get('context', 'N/A')
    print(f"\n📝 맥락:")
    print(f"   {context}")
    
    situation = result.get('situation', 'N/A')
    print(f"\n🎯 상황 분석:")
    print(f"   {situation}")
    
    # 시각 정보
    visual_content = result.get('visual_content', 'N/A')
    print(f"\n👁️  시각 정보:")
    print(f"   {visual_content}")
    
    # 음성-시각 일치도
    consistency = result.get('audio_visual_consistency', 'N/A')
    consistency_emoji = {
        '일치': '✅',
        '불일치': '⚠️',
        '부분일치': '🔶'
    }.get(consistency, '❓')
    print(f"\n{consistency_emoji} 음성-시각 일치도: {consistency}")
    
    # 감정 상태
    emotional_state = result.get('emotional_state', 'N/A')
    print(f"\n😊 감정 상태: {emotional_state}")
    
    # 상황 유형
    situation_type = result.get('situation_type', 'N/A')
    print(f"\n📌 상황 유형: {situation_type}")
    
    # 권장 조치
    action = result.get('action', 'N/A')
    print(f"\n🔧 권장 조치:")
    print(f"   {action}")
    
    # 긴급 알림 트리거
    if is_emergency or priority == 'CRITICAL':
        print("\n" + "=" * 70)
        alert_manager.trigger_alert({
            'is_emergency': is_emergency,
            'emergency_reason': result.get('emergency_reason', '알 수 없는 긴급 상황'),
            'priority': priority,
            'situation_type': situation_type
        })
    
    # 결과 저장
    print("\n" + "=" * 70)
    print("💾 결과 저장")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f'custom_image_test_{timestamp}.json'
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'image_path': image_path,
        'audio_text': audio_text,
        'additional_context': additional_context,
        'analysis': result
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 로그 저장: {log_file.name}")
    
    # 전체 JSON 출력 (선택)
    print("\n" + "=" * 70)
    show_json = input("전체 JSON 결과를 보시겠습니까? (y/N): ").strip().lower()
    
    if show_json == 'y':
        print("\n📄 전체 JSON 결과:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
except Exception as e:
    print(f"❌ 분석 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ 테스트 완료!")
print("=" * 70)
print(f"\n💡 다른 이미지로 테스트하려면 다시 실행하세요:")
print(f"   python {Path(__file__).name}")
