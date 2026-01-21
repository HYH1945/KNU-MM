#!/usr/bin/env python3
"""
음성 입력 → 즉시 분석 스크립트 (테스트용)

역할:
1. 문맥(context) 파악
2. 위급도(urgency) 판단
3. 상황(situation) 분석
"""

import requests
import json
import sys

def analyze_voice_input(text):
    """
    음성 입력을 실시간으로 분석
    
    반환:
    {
        "context": "대화의 맥락",
        "urgency": "위급도 (낮음/중간/높음/긴급)",
        "situation": "상황 분석",
        "action": "권장 조치"
    }
    """
    
    # 사용자의 요구사항에 맞춘 최적화된 프롬프트
    prompt = """당신은 음성 입력을 실시간으로 분석하는 상황 분석 AI입니다.

다음을 JSON으로만 반환하세요:
{
  "context": "대화의 맥락을 한 문장으로 설명",
  "urgency": "위급도 (낮음/중간/높음/긴급 중 하나)",
  "urgency_reason": "왜 그 위급도인지 설명",
  "situation_type": "상황 유형 (업무/긴급상황/일상/정보요청 등)",
  "situation": "상황을 3-5줄로 자세히 분석",
  "emotional_state": "감정 상태",
  "immediate_action": "즉시 취할 행동",
  "follow_up": "후속 조치"
}

음성 입력 텍스트:"""

    try:
        print(f"\n🤖 분석 중...\n")
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt + f' "{text}"',
                'stream': False,
                'temperature': 0.3  # 일관성
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['response'].strip()
            
            # JSON 추출
            try:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
                else:
                    return {'error': 'JSON 형식을 찾을 수 없음', 'raw': response_text}
            except json.JSONDecodeError as e:
                return {'error': 'JSON 파싱 실패', 'raw': response_text[:200]}
        else:
            return {'error': f'Ollama 오류 ({response.status_code})'}
    
    except requests.exceptions.ConnectionError:
        return {'error': 'Ollama 서버에 연결할 수 없음. 실행: ollama serve'}
    except Exception as e:
        return {'error': str(e)}

def format_analysis(analysis):
    """분석 결과를 보기 좋게 포맷팅"""
    
    if 'error' in analysis:
        print(f"❌ 오류: {analysis['error']}")
        if 'raw' in analysis:
            print(f"   응답: {analysis['raw']}")
        return
    
    # 위급도별 아이콘
    urgency_icons = {
        '낮음': '🟢',
        '중간': '🟡',
        '높음': '🔴',
        '긴급': '🚨'
    }
    
    print("="*70)
    print("📊 상황 분석 결과")
    print("="*70)
    
    print(f"\n📍 맥락: {analysis.get('context', 'N/A')}")
    
    urgency = analysis.get('urgency', 'N/A')
    icon = urgency_icons.get(urgency, '⚪')
    print(f"\n{icon} 위급도: {urgency}")
    print(f"   이유: {analysis.get('urgency_reason', 'N/A')}")
    
    print(f"\n🎯 상황 유형: {analysis.get('situation_type', 'N/A')}")
    
    print(f"\n💭 상황 분석:")
    for line in analysis.get('situation', 'N/A').split('\n'):
        if line.strip():
            print(f"   {line}")
    
    print(f"\n😊 감정 상태: {analysis.get('emotional_state', 'N/A')}")
    
    print(f"\n⚡ 즉시 조치:")
    for line in analysis.get('immediate_action', 'N/A').split('\n'):
        if line.strip():
            print(f"   • {line}")
    
    print(f"\n📋 후속 조치:")
    for line in analysis.get('follow_up', 'N/A').split('\n'):
        if line.strip():
            print(f"   • {line}")
    
    print("\n" + "="*70)

def interactive_test():
    """대화형 테스트 모드"""
    
    print("\n🎤 음성 입력 분석 (대화형 모드)")
    print("="*70)
    print("음성 텍스트를 입력하세요 (q 입력하면 종료)\n")
    
    test_phrases = [
        "서버가 다운됐어요! 지금 접속이 안 됩니다.",
        "오늘 날씨 정말 좋네요",
        "급한데, 회의 시간을 앞당길 수 있을까요?",
        "프로젝트 마감일이 내일이거든요",
        "안녕하세요, 반갑습니다"
    ]
    
    print("📌 테스트 예제:")
    for i, phrase in enumerate(test_phrases, 1):
        print(f"   {i}. {phrase}")
    
    print("\n입력 (1-5 숫자 또는 직접 입력):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'q':
                print("종료합니다.")
                break
            
            # 숫자 입력 처리
            if user_input.isdigit() and 1 <= int(user_input) <= len(test_phrases):
                text = test_phrases[int(user_input) - 1]
                print(f"\n📝 입력: {text}")
            else:
                text = user_input
            
            if text:
                analysis = analyze_voice_input(text)
                format_analysis(analysis)
        
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")

def test_various_scenarios():
    """다양한 상황 시나리오 테스트"""
    
    scenarios = [
        {
            "name": "🚨 긴급 상황 1: 시스템 다운",
            "text": "우리 서버가 완전히 다운 됐어요! 고객들이 접속 못 하고 있습니다. 매분 매초가 중요합니다!"
        },
        {
            "name": "🚨 긴급 상황 2: 마감 임박",
            "text": "프로젝트 마감이 2시간 뒤입니다. 아직도 버그가 남아있어요."
        },
        {
            "name": "🟡 중간 상황: 일상적 업무",
            "text": "내일 회의 시간을 10시로 변경해주세요."
        },
        {
            "name": "🟢 낮은 상황: 일상 대화",
            "text": "오늘 날씨 정말 좋네요. 주말에 산책 갈 계획입니다."
        },
        {
            "name": "🟡 정보 요청",
            "text": "파이썬에서 리스트를 정렬하는 방법이 뭐예요?"
        },
        {
            "name": "🔴 높은 위급도: 고객 불만",
            "text": "3일째 문제가 안 풀려요. 진짜 실망했습니다. 환불 처리해주세요!"
        }
    ]
    
    print("\n🎯 다양한 상황 시나리오 테스트")
    print("="*70)
    
    for scenario in scenarios:
        print(f"\n\n{scenario['name']}")
        print("-"*70)
        print(f"📝 입력: {scenario['text']}")
        
        analysis = analyze_voice_input(scenario['text'])
        format_analysis(analysis)
        
        # 잠깐 대기
        import time
        time.sleep(1)

if __name__ == "__main__":
    print("\n" + "🎤 음성 입력 상황 분석 시스템" + "\n")
    
    # 연결 확인
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code != 200:
            print("❌ Ollama에 연결할 수 없습니다")
            sys.exit(1)
        print("✅ Ollama 연결됨\n")
    except:
        print("❌ Ollama가 실행 중이 아닙니다. 실행: ollama serve")
        sys.exit(1)
    
    print("선택:")
    print("  1. 대화형 테스트 (직접 입력)")
    print("  2. 시나리오 테스트 (자동 실행)")
    print("  3. 한 가지 예제 분석\n")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == '1':
        interactive_test()
    elif choice == '2':
        test_various_scenarios()
    elif choice == '3':
        text = input("\n분석할 텍스트 입력: ").strip()
        if text:
            print(f"\n📝 입력: {text}")
            analysis = analyze_voice_input(text)
            format_analysis(analysis)
    else:
        print("❌ 잘못된 선택")
