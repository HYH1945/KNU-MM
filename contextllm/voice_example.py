#!/usr/bin/env python3
"""
음성 인식 + LLM 분석을 활용하는 예제 프로그램

이 파일을 실행하면 필요한 시점에 음성 녹음 및 분석이 가능합니다.
"""

import sys
import json
from voice_analyzer import VoiceAnalyzer

def example_1_simple_transcription():
    """예제 1: 단순 음성 변환"""
    print("\n" + "="*60)
    print("예제 1: 단순 음성 변환 (10초)")
    print("="*60)
    
    analyzer = VoiceAnalyzer()
    result = analyzer.transcribe_and_analyze(duration=10)
    
    if result['success']:
        print(f"\n📝 텍스트: {result['transcribed_text']}")
        print(f"🤖 분석: {json.dumps(result['analysis'], ensure_ascii=False, indent=2)}")
    else:
        print(f"❌ 오류: {result.get('error')}")


def example_2_custom_prompt():
    """예제 2: 커스텀 프롬프트로 분석"""
    print("\n" + "="*60)
    print("예제 2: 커스텀 프롬프트로 분석")
    print("="*60)
    
    custom_prompt = """당신은 회의 기록을 분석하는 전문가입니다.
다음을 JSON으로 반환하세요:
- meeting_topic: 회의 주제
- key_decisions: 주요 결정사항 (배열)
- action_items: 액션 아이템 (배열)
- attendees: 참석자 (배열)"""
    
    analyzer = VoiceAnalyzer()
    result = analyzer.transcribe_and_analyze(
        duration=10,
        system_prompt=custom_prompt
    )
    
    if result['success']:
        print(f"\n📝 텍스트: {result['transcribed_text']}")
        print(f"📋 분석: {json.dumps(result['analysis'], ensure_ascii=False, indent=2)}")
    else:
        print(f"❌ 오류: {result.get('error')}")


def example_3_different_durations():
    """예제 3: 다양한 녹음 시간"""
    print("\n" + "="*60)
    print("예제 3: 다양한 녹음 시간 테스트")
    print("="*60)
    
    analyzer = VoiceAnalyzer()
    durations = [5, 10, 15]
    
    for duration in durations:
        print(f"\n⏱️  {duration}초 녹음...")
        result = analyzer.transcribe_and_analyze(duration=duration)
        
        if result['success']:
            print(f"   📝 {result['transcribed_text'][:40]}...")
        else:
            print(f"   ❌ 오류: {result.get('error')}")


def example_4_real_time_monitoring():
    """예제 4: 실시간 모니터링 (특정 키워드 감지)"""
    print("\n" + "="*60)
    print("예제 4: 실시간 모니터링 (3회 반복)")
    print("="*60)
    
    analyzer = VoiceAnalyzer()
    keyword_prompt = """사용자가 말한 내용에서 다음을 분석하세요:
- has_urgent: 긴급 상황인가? (true/false)
- has_question: 질문인가? (true/false)
- sentiment: 감정 (긍정/중립/부정)
- key_phrase: 가장 중요한 단어"""
    
    for i in range(3):
        print(f"\n[{i+1}번째] 음성 입력 대기... (10초)")
        result = analyzer.transcribe_and_analyze(
            duration=10,
            system_prompt=keyword_prompt
        )
        
        if result['success']:
            analysis = result['analysis']
            print(f"   입력: {result['transcribed_text']}")
            
            # 중요도에 따라 반응
            if analysis.get('has_urgent') == True:
                print("   🚨 긴급 상황 감지!")
            if analysis.get('has_question') == True:
                print("   ❓ 질문 감지!")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "="*60)
    print("🎤 대화형 음성 인식 + LLM 분석")
    print("="*60)
    print("명령어:")
    print("  1. 분석 시작 (10초)")
    print("  2. 분석 시작 (5초)")
    print("  3. 분석 시작 (15초)")
    print("  q. 종료\n")
    
    analyzer = VoiceAnalyzer()
    
    while True:
        cmd = input("명령 입력: ").strip()
        
        if cmd == '1':
            result = analyzer.transcribe_and_analyze(duration=10)
        elif cmd == '2':
            result = analyzer.transcribe_and_analyze(duration=5)
        elif cmd == '3':
            result = analyzer.transcribe_and_analyze(duration=15)
        elif cmd.lower() == 'q':
            print("종료합니다.")
            break
        else:
            print("❌ 잘못된 명령")
            continue
        
        if result['success']:
            print(f"\n✅ 텍스트: {result['transcribed_text']}")
            print(f"   분석: {json.dumps(result['analysis'], ensure_ascii=False)}\n")
        else:
            print(f"❌ 오류: {result.get('error')}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == '1':
            example_1_simple_transcription()
        elif cmd == '2':
            example_2_custom_prompt()
        elif cmd == '3':
            example_3_different_durations()
        elif cmd == '4':
            example_4_real_time_monitoring()
        elif cmd == 'interactive':
            interactive_mode()
        else:
            print(f"❌ 알 수 없는 명령: {cmd}")
    else:
        print("\n사용법:")
        print("  python3 voice_example.py 1         # 단순 변환")
        print("  python3 voice_example.py 2         # 커스텀 프롬프트")
        print("  python3 voice_example.py 3         # 다양한 시간")
        print("  python3 voice_example.py 4         # 모니터링")
        print("  python3 voice_example.py interactive # 대화형 모드")
        print()
        interactive_mode()
