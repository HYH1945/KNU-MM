#!/usr/bin/env python3
"""
Ollama Mistral 연결 테스트 & 기능 테스트
"""

import requests
import json

def test_ollama_connection():
    """Ollama 서버 연결 테스트"""
    print("🔌 Ollama 연결 테스트 중...")
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            print(f"✅ Ollama 연결 성공!")
            print(f"📦 설치된 모델: {models}")
            return True
        else:
            print(f"❌ Ollama 응답 오류: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama 서버에 연결할 수 없습니다")
        print("   실행: ollama serve")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def test_simple_generation():
    """간단한 생성 테스트"""
    print("\n🤖 간단한 생성 테스트 중...")
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': '안녕하세요. 당신은 누구입니까?',
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 생성 성공!")
            print(f"응답: {result['response'][:100]}...")
            return True
        else:
            print(f"❌ 생성 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def test_context_analysis():
    """문맥 분석 테스트"""
    print("\n📊 문맥 분석 테스트 중...")
    
    test_text = "사용자가 '회의에서 프로젝트 일정을 앞당기기로 했고, 모두가 동의했습니다'라고 말했습니다."
    
    prompt = """다음 텍스트의 의도와 감정을 분석하고 JSON으로 반환하세요:
텍스트: """ + test_text + """

다음 필드를 포함하세요:
- intent: 사용자의 의도
- sentiment: 감정 (긍정/중립/부정)
- key_entities: 중요 개체들
- urgency: 긴급도 (높음/중간/낮음)

JSON만 반환하세요."""
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 분석 성공!")
            print(f"응답:\n{result['response']}")
            
            # JSON 파싱 시도
            try:
                analysis = json.loads(result['response'])
                print(f"\n📋 파싱된 결과:")
                print(json.dumps(analysis, ensure_ascii=False, indent=2))
            except:
                print("\n⚠️  JSON 파싱 실패 (텍스트로 분석함)")
            
            return True
        else:
            print(f"❌ 분석 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def test_korean_support():
    """한글 지원 테스트"""
    print("\n🇰🇷 한글 지원 테스트 중...")
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': '한국어로 다음을 요약하세요: "인공지능은 현대 사회에서 점점 더 중요한 역할을 하고 있습니다. 특히 음성 인식 기술은 사람들의 일상을 더욱 편리하게 만들어주고 있습니다."',
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 한글 처리 성공!")
            print(f"요약: {result['response'][:150]}...")
            return True
        else:
            print(f"❌ 한글 처리 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def test_streaming():
    """스트리밍 응답 테스트"""
    print("\n⚡ 스트리밍 응답 테스트 중...")
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': '5가지 프로그래밍 팁을 제시하세요.',
                'stream': True
            },
            timeout=60,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ 스트리밍 시작!")
            print("응답 (실시간):")
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    print(data['response'], end='', flush=True)
            print("\n\n✅ 스트리밍 완료!")
            return True
        else:
            print(f"❌ 스트리밍 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🎤 Ollama Mistral 종합 테스트")
    print("="*60)
    
    # 1. 연결 테스트
    if not test_ollama_connection():
        exit(1)
    
    # 2. 간단한 생성 테스트
    if not test_simple_generation():
        exit(1)
    
    # 3. 문맥 분석 테스트
    test_context_analysis()
    
    # 4. 한글 지원 테스트
    test_korean_support()
    
    # 5. 스트리밍 테스트
    test_streaming()
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료!")
    print("="*60)
