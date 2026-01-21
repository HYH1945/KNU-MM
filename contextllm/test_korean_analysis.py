#!/usr/bin/env python3
"""
한국어 문맥 분석 테스트 - Mistral 활용
음성 인식 결과를 한국어 문맥으로 분석
"""

import json
import requests
from datetime import datetime

def test_korean_context_analysis():
    """한국어 문맥 분석 테스트"""
    
    test_cases = [
        {
            "name": "회의 결정",
            "text": "우리팀에서 내년도 프로젝트 일정을 앞당기기로 했습니다. 모두 동의했고, 다음주 월요일부터 시작합니다.",
            "prompt": """이 회의 기록을 분석하고 JSON으로만 반환하세요:
{
  "decision": "내린 결정",
  "timeline": "일정",
  "participants": "참석자",
  "urgency": "긴급도 (높음/중간/낮음)"
}"""
        },
        {
            "name": "고객 불만",
            "text": "이 제품 정말 별로네요. 사용해본 지 3일만에 고장났어요. 환불 처리 빨리 해주세요!",
            "prompt": """고객 피드백을 분석하고 JSON으로만 반환하세요:
{
  "sentiment": "긍정/중립/부정",
  "issue": "주요 문제",
  "priority": "우선순위 (높음/중간/낮음)",
  "action": "필요한 조치"
}"""
        },
        {
            "name": "학습 내용",
            "text": "오늘 배운 내용은 파이썬의 데이터 구조입니다. 리스트, 딕셔너리, 튜플의 차이점을 이해했고, 실습 예제도 완료했습니다.",
            "prompt": """학습 기록을 분석하고 JSON으로만 반환하세요:
{
  "topics": ["배운 주제들"],
  "level": "난이도 (초급/중급/고급)",
  "completion": "완성도 (0-100)",
  "next_step": "다음 학습 주제"
}"""
        },
        {
            "name": "긴급 상황",
            "text": "서버가 다운됐어요! 지금 접속이 안 되고 있습니다. 빨리 조치해주세요!",
            "prompt": """상황을 분석하고 JSON으로만 반환하세요:
{
  "severity": "심각도 (심각/중간/경미)",
  "issue_type": "문제 종류",
  "immediate_action": "즉시 조치",
  "impact": "영향 범위"
}"""
        }
    ]
    
    print("🤖 한국어 문맥 분석 테스트 (Mistral)\n")
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test_case['name']}")
        print("-"*70)
        print(f"입력: {test_case['text']}")
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mistral',
                    'prompt': f"{test_case['prompt']}\n\n텍스트: {test_case['text']}",
                    'stream': False,
                    'temperature': 0.3  # 일관성 높임
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
                        print(f"\n✅ 분석 결과:")
                        print(json.dumps(analysis, ensure_ascii=False, indent=2))
                    else:
                        print(f"\n⚠️  JSON 형식 아님:")
                        print(response_text[:200])
                except json.JSONDecodeError:
                    print(f"\n⚠️  파싱 실패:")
                    print(response_text[:200])
            else:
                print(f"❌ 오류: {response.status_code}")
        
        except Exception as e:
            print(f"❌ 예외: {e}")

def test_conversation_context():
    """대화 문맥 추적 테스트"""
    
    print("\n\n" + "="*70)
    print("🗣️  대화 문맥 추적 테스트")
    print("="*70)
    
    conversation = [
        "안녕하세요, 문의가 있습니다.",
        "네, 말씀해주세요. 어떤 문제인가요?",
        "서비스를 사용하다가 오류가 나서요.",
        "어떤 오류인지 자세히 설명해주실 수 있나요?",
        "로그인 후 대시보드가 안 열려요. 계속 로딩 중이라고 표시돼요."
    ]
    
    context_prompt = """사용자의 문제를 분석하고 JSON으로만 반환하세요:
{
  "issue": "주요 문제",
  "symptoms": "증상",
  "severity": "심각도",
  "suggested_solution": "해결책",
  "next_question": "다음 질문"
}"""
    
    print("\n사용자 입력 순서:")
    for i, msg in enumerate(conversation, 1):
        if i % 2 == 1:  # 사용자 입력
            print(f"{i}. 사용자: {msg}")
    
    print("\n\n마지막 사용자 입력 분석:")
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': f"{context_prompt}\n\n사용자 입력 이력:\n" + 
                         "\n".join(f"- {msg}" for msg in conversation[::2]) +
                         f"\n\n최신 입력: {conversation[-1]}",
                'stream': False,
                'temperature': 0.3
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['response'].strip()
            
            try:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    print(json.dumps(analysis, ensure_ascii=False, indent=2))
            except:
                print(response_text[:300])
    
    except Exception as e:
        print(f"❌ 오류: {e}")

def test_sentiment_analysis():
    """감정 분석 테스트"""
    
    print("\n\n" + "="*70)
    print("❤️  감정 분석 테스트")
    print("="*70)
    
    texts = [
        ("매우 행복합니다!", "긍정"),
        ("이건 최악이에요.", "부정"),
        ("그냥 평범해요.", "중립"),
        ("이 제품은 정말 대박이네요!", "강한 긍정"),
        ("엄청 실망했어요.", "강한 부정")
    ]
    
    for text, expected in texts:
        print(f"\n입력: {text}")
        print(f"예상: {expected}")
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mistral',
                    'prompt': f"""감정을 분석하고 JSON으로만 반환:
{{
  "sentiment": "긍정/약한긍정/중립/약한부정/부정",
  "intensity": "0-100 (0:매우낮음, 100:매우높음)",
  "keywords": ["감정 표현 단어들"]
}}

텍스트: {text}""",
                    'stream': False,
                    'temperature': 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['response'].strip()
                
                try:
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                        sentiment = analysis.get('sentiment')
                        intensity = analysis.get('intensity')
                        print(f"🎯 분석: {sentiment} (강도: {intensity})")
                except:
                    print(f"응답: {response_text[:100]}")
        
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    print("🎤 한국어 문맥 분석 데모\n")
    print("⚠️  Ollama가 실행 중인지 확인하세요: ollama serve\n")
    
    try:
        # 연결 테스트
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code != 200:
            print("❌ Ollama에 연결할 수 없습니다")
            exit(1)
    except:
        print("❌ Ollama 서버가 실행 중이 아닙니다")
        print("   실행: ollama serve")
        exit(1)
    
    # 테스트 실행
    test_korean_context_analysis()
    test_conversation_context()
    test_sentiment_analysis()
    
    print("\n\n" + "="*70)
    print("✅ 테스트 완료!")
    print("="*70)
