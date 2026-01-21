# 🤖 Ollama Mistral 테스트 가이드

## ✅ 현재 상황
- **Ollama 버전**: 0.14.2 ✅ 설치됨
- **Mistral 모델**: 다운로드 완료
- **Python 환경**: 준비됨

---

## 🚀 테스트 방법

### 단계 1: Ollama 서버 시작 (터미널 1)
```bash
ollama serve
```

**예상 출력:**
```
time=2026-01-21T11:30:00.000Z level=INFO msg="Listening on 127.0.0.1:11434"
```

### 단계 2: Ollama 테스트 (터미널 2)
```bash
cd /Users/jangjun-yong/Desktop/jongf1

# Ollama 및 Mistral 연결 테스트
python3 test_ollama.py
```

**예상 결과:**
```
🔌 Ollama 연결 테스트 중...
✅ Ollama 연결 성공!
📦 설치된 모델: ['mistral:latest']

🤖 간단한 생성 테스트 중...
✅ 생성 성공!
응답: Hello! I'm Mistral, an AI assistant...

📊 문맥 분석 테스트 중...
✅ 분석 성공!
응답:
{
  "intent": "정보 공유",
  "sentiment": "긍정",
  "key_entities": ["회의", "프로젝트"],
  "urgency": "중간"
}

✅ 모든 테스트 완료!
```

### 단계 3: 음성 인식 + LLM 분석 (터미널 3)
```bash
cd /Users/jangjun-yong/Desktop/jongf1

# 테스트 실행
python3 voice_example.py interactive

# 또는 한 번만 실행
python3 voice_analyzer.py
```

**흐름:**
```
🎤 녹음 중... (10초)
✅ 녹음 완료

⚙️  Whisper 변환 중...
✅ 변환 완료: "안녕하세요 저는..."

🤖 LLM(Mistral) 분석 중...
✅ LLM 분석 완료

결과:
{
  "intent": "인사",
  "sentiment": "긍정",
  "summary": "사용자가 자신을 소개함"
}
```

---

## 💡 Mistral의 문맥 분석 능력

Mistral은 다음을 잘 합니다:

### 1️⃣ **의도 파악 (Intent)**
```
입력: "회의에서 프로젝트 일정을 앞당기기로 했습니다"
분석:
- intent: "결정 보고"
- action: "일정표 업데이트"
```

### 2️⃣ **감정 분석 (Sentiment)**
```
입력: "정말 좋은 뉴스입니다!"
분석:
- sentiment: "긍정" 
- confidence: "높음"
```

### 3️⃣ **개체 추출 (Named Entity Recognition)**
```
입력: "김철수는 서울에서 내일 회의를 합니다"
분석:
- entities: ["김철수", "서울", "내일", "회의"]
```

### 4️⃣ **요약 (Summarization)**
```
입력: "우리는 새로운 기능을 추가했고, 
       버그를 수정했고, 성능을 개선했습니다"
분석:
- summary: "제품 업데이트 완료"
```

### 5️⃣ **우선순위 판단 (Prioritization)**
```
입력: "긴급하게 데이터베이스 서버를 
      점검해야 합니다. 시스템이 다운되었습니다"
분석:
- urgency: "높음"
- severity: "심각"
```

---

## 🎯 실제 사용 예제

### 예제 1: 회의 기록 분석
```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 회의 중 10초 녹음 + 분석
result = analyzer.transcribe_and_analyze(
    duration=10,
    system_prompt="""회의 내용을 분석하세요. JSON으로 반환:
{
  "topic": "주제",
  "decisions": ["결정사항1", "결정사항2"],
  "action_items": [{"task": "할일", "owner": "담당자"}],
  "urgency": "높음/중간/낮음"
}"""
)

# 결과
print(result['transcribed_text'])  # "의장은 4분기 목표를..."
print(result['analysis']['decisions'])  # ["예산 증액", "팀 확대"]
```

### 예제 2: 고객 지원 분석
```python
result = analyzer.transcribe_and_analyze(
    duration=15,
    system_prompt="""고객 피드백을 분석하세요:
{
  "sentiment": "긍정/중립/부정",
  "issue_type": "기술/결제/기타",
  "priority": "높음/중간/낮음",
  "suggested_solution": "해결책"
}"""
)

# 감정 기반 대응
if result['analysis']['sentiment'] == '부정':
    print("⚠️  부정적인 피드백 감지 - 즉시 대응 필요")
```

### 예제 3: 학습 강의 분석
```python
result = analyzer.transcribe_and_analyze(
    duration=30,
    system_prompt="""강의 내용을 분석하세요:
{
  "topics": ["주제1", "주제2"],
  "key_concepts": ["개념1", "개념2"],
  "difficulty": "초급/중급/고급",
  "learning_outcomes": ["목표1", "목표2"]
}"""
)
```

---

## 🔍 문제 해결

### 문제 1: "LLM 서버에 연결할 수 없음"

**해결:**
```bash
# 터미널에서 확인
ollama serve

# 또는 다시 시작
killall ollama
ollama serve
```

### 문제 2: "mistral 모델을 찾을 수 없음"

**해결:**
```bash
# 모델 다운로드
ollama pull mistral

# 또는 다른 모델 사용
ollama pull neural-chat
ollama pull llama2
```

### 문제 3: "응답이 JSON이 아님"

**이유**: Mistral이 JSON 형식을 항상 따르지 않을 수 있음
**해결**: 프롬프트에 "JSON만 반환하세요" 추가

### 문제 4: "느린 응답"

**해결:**
- GPU 사용 확인: `ollama list`
- 더 빠른 모델 사용: `neural-chat` (빠름) 또는 `mistral` (정확함)

---

## 📊 Mistral vs 다른 모델

| 모델 | 크기 | 속도 | 한글 | 용도 |
|------|------|------|------|------|
| **Mistral** | 7B | 보통 | ⭐⭐⭐⭐ | 일반적인 분석 ✅ |
| **Neural Chat** | 7B | 빠름 | ⭐⭐⭐ | 빠른 응답 필요 시 |
| **Llama 2** | 7B | 보통 | ⭐⭐⭐ | 코드 생성 |

---

## ✨ 핵심 포인트

✅ **Mistral은 문맥 분석에 매우 우수합니다**
- 한글 처리 우수
- 의도 파악 정확함
- 감정 분석 신뢰도 높음

✅ **Ollama는 로컬에서 프라이빗하게 실행**
- 데이터 유출 없음
- API 비용 0원
- 인터넷 불필요

✅ **voice_analyzer.py로 통합**
- 녹음 → 변환 → 분석 자동화
- 다양한 프롬프트로 커스터마이징 가능
- 필요한 시점에만 호출 가능

---

## 🚀 다음 단계

1. **test_ollama.py 실행** → Ollama 성능 확인
2. **voice_analyzer.py 실행** → 전체 파이프라인 테스트
3. **프로그램에 임포트** → 필요한 시점에 호출
4. **커스텀 프롬프트** → 분석 정확도 조정

---

## 💬 프롬프트 팁

더 좋은 분석을 위해:

```python
# ❌ 나쁜 프롬프트
"분석해줘"

# ✅ 좋은 프롬프트
"""당신은 음성 기록 분석 전문가입니다.
다음을 JSON으로만 반환하세요 (다른 텍스트 없이):
{
  "intent": "...",
  "sentiment": "긍정/중립/부정 중 하나"
}"""
```

프롬프트가 명확할수록 응답이 더 정확합니다! 🎯
