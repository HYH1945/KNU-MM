#!/usr/bin/env python3
"""
voice_analyzer.py 실행 테스트
"""
import sys
import os

# 현재 디렉토리 추가
sys.path.insert(0, '/Users/jangjun-yong/Desktop/github/KNU-MM/contextllm')

try:
    from voice_analyzer import VoiceAnalyzer
    print("✅ voice_analyzer.py 임포트 성공!")
    
    # 인스턴스 생성
    analyzer = VoiceAnalyzer()
    print("✅ VoiceAnalyzer 인스턴스 생성 성공!")
    
    # 메서드 확인
    print("\n📋 사용 가능한 메서드:")
    methods = [m for m in dir(analyzer) if not m.startswith('_')]
    for method in methods:
        print(f"  • {method}")
    
    print("\n✨ 모든 테스트 통과!")
    print("\n🎤 사용 예:")
    print("  1. analyzer.transcribe_and_analyze(duration=10)")
    print("  2. analyzer.transcribe_and_analyze(duration=None)")
    print("  3. analyzer.run_continuously(interval=10)")
    
except Exception as e:
    print(f"❌ 오류: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
