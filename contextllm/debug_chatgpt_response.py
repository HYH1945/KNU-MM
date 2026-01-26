#!/usr/bin/env python3
"""
ChatGPT 분석 결과 디버그 테스트
긴급 키워드로 테스트해서 실제 응답값 확인
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
from core.voice_analyzer import VoiceAnalyzer

# .env 로드
env_path = Path(__file__).parent / 'config' / '.env'
load_dotenv(env_path)

# VoiceAnalyzer 초기화
analyzer = VoiceAnalyzer()

# 테스트 케이스
test_cases = [
    "도와줘! 침입자가 들어왔어!",
    "살려줘!",
    "집에 불이 났어!",
    "안녕하세요 오늘 날씨 어때요?",
    "경찰을 불러줘! 긴급이야!"
]

print("=" * 70)
print("🧪 ChatGPT 긴급 감지 테스트")
print("=" * 70)

for i, test_text in enumerate(test_cases, 1):
    print(f"\n[테스트 {i}] 입력: '{test_text}'")
    print("-" * 70)
    
    analysis = analyzer.analyze_with_llm(test_text)
    
    print(f"전체 응답: {analysis}\n")
    
    priority = analysis.get('priority', 'NO_PRIORITY')
    is_emergency = analysis.get('is_emergency', 'NO_EMERGENCY')
    emergency_reason = analysis.get('emergency_reason', 'N/A')
    
    print(f"✓ priority: {priority}")
    print(f"✓ is_emergency: {is_emergency}")
    print(f"✓ emergency_reason: {emergency_reason}")
    
    if is_emergency or priority == 'CRITICAL':
        print(f"🚨 => 긴급 조건 만족!")
    else:
        print(f"ℹ️  => 일반 상황")

print("\n" + "=" * 70)
