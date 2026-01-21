# 음성 인식 + LLM 분석 설정 가이드

## 📋 구조

```
음성 녹음 (10초)
    ↓
Whisper 변환 (음성 → 텍스트)
    ↓
LLM 분석 (텍스트 → 컨텍스트)
    ↓
결과 저장 (transcriptions/ 폴더)
```

## 🚀 빠른 시작

### 1단계: 필수 도구 설치

```bash
# Whisper 이미 설치됨 (`.venv/bin/python3`에)

# sox 설치 (녹음용)
brew install sox

# Ollama 설치 (LLM용)
# https://ollama.ai에서 다운로드
```

### 2단계: Ollama 실행

```bash
# 터미널 1: Ollama 서버 시작
ollama serve

# 터미널 2: 모델 다운로드 (선택사항)
ollama pull mistral
ollama pull neural-chat
```

### 3단계: Python 코드 실행

```bash
# 터미널 3
cd /Users/jangjun-yong/Desktop/jongf1

# 예제 1: 단순 변환
python3 voice_analyzer.py

# 또는 예제 프로그램
python3 voice_example.py interactive
```

---

## 🎯 사용 시나리오별 커스텀 프롬프트

### 시나리오 1: 회의 기록

```python
meeting_prompt = """당신은 회의 기록 분석 전문가입니다.
다음을 JSON으로 반환하세요:
{
  "meeting_topic": "회의 주제",
  "key_decisions": ["결정사항 1", "결정사항 2"],
  "action_items": [
    {
      "task": "할일",
      "owner": "담당자",
      "deadline": "마감일"
    }
  ]
}"""

analyzer = VoiceAnalyzer()
result = analyzer.transcribe_and_analyze(duration=10, system_prompt=meeting_prompt)
```

### 시나리오 2: 고객 지원

```python
support_prompt = """당신은 고객 지원 분석 전문가입니다.
{
  "issue_type": "문제 유형 (기술/결제/기타)",
  "priority": "우선순위 (높음/중간/낮음)",
  "sentiment": "고객 감정 (긍정/중립/부정)",
  "suggested_solution": "제안 해결책"
}"""

analyzer = VoiceAnalyzer()
result = analyzer.transcribe_and_analyze(duration=15, system_prompt=support_prompt)
```

### 시나리오 3: 학습 강의

```python
lecture_prompt = """당신은 학습 강의 분석 전문가입니다.
{
  "main_topics": ["주제 1", "주제 2"],
  "key_concepts": ["개념 1", "개념 2"],
  "learning_objectives": ["목표 1", "목표 2"],
  "difficulty_level": "난이도 (초급/중급/고급)"
}"""
```

---

## 📁 파일 구조

```
jongf1/
├── voice_analyzer.py          # 핵심 모듈 (임포트하여 사용)
├── voice_example.py           # 사용 예제
├── voice_monitor.py           # 자동 모니터링용 (참고용)
├── api_server.py              # REST API용 (참고용)
├── whisper_service.py         # Whisper 변환 스크립트
├── recordings/                # 임시 녹음 파일 (자동 정리)
└── transcriptions/            # 결과 저장
    └── 2026-01-21/
        ├── transcriptions.json       # 모든 결과 (배열)
        ├── latest.json              # 최신 결과
        └── 2026-01-21T11-26-30-123.json  # 개별 결과
```

---

## 🔌 다른 프로그램과 연동

### Python에서 사용

```python
from voice_analyzer import VoiceAnalyzer

# 필요한 시점에 호출
analyzer = VoiceAnalyzer()
result = analyzer.transcribe_and_analyze(duration=10)

# 분석 결과 활용
if result['success']:
    text = result['transcribed_text']
    analysis = result['analysis']
    
    # 예: 감정이 부정적이면 경고
    if analysis.get('sentiment') == '부정':
        print("⚠️  부정적인 피드백 감지!")
```

### 파일 기반 연동

```python
import json
from pathlib import Path

# 최신 결과 읽기
latest_file = Path("transcriptions/2026-01-21/latest.json")
with open(latest_file) as f:
    result = json.load(f)
    print(result['analysis'])
```

### Shell/Command Line 연동

```bash
# Python 스크립트 실행 후 결과 읽기
python3 voice_example.py 1 && \
cat transcriptions/*/latest.json | jq '.analysis'
```

---

## 🛠️ 트러블슈팅

### 문제 1: "LLM 서버에 연결할 수 없음"

**해결:**
```bash
# Ollama가 실행 중인지 확인
ollama serve

# 또는 Ollama 설정 변경 필요
OLLAMA_HOST=localhost:11434 ollama serve
```

### 문제 2: "녹음 오류"

**해결:**
```bash
# sox 설치 확인
which sox

# 설치되지 않았으면
brew install sox
```

### 문제 3: "변환 실패 (No module named 'whisper')"

**해결:**
```bash
# 가상환경 활성화 확인
source /Users/jangjun-yong/Desktop/jongf1/.venv/bin/activate
pip list | grep whisper
```

---

## 📊 결과 예시

```json
{
  "success": true,
  "timestamp": "2026-01-21T11:26:30.123Z",
  "transcribed_text": "안녕하세요 저는 음성 인식 시스템을 테스트하고 있습니다",
  "analysis": {
    "intent": "정보 제공",
    "entities": ["음성 인식", "시스템"],
    "sentiment": "긍정",
    "summary": "사용자가 음성 인식 시스템 테스트를 진행 중",
    "action": "분석 완료"
  },
  "audio_file": "/Users/jangjun-yong/Desktop/jongf1/recordings/audio_20260121_112630_123.wav"
}
```

---

## 💡 팁

1. **비용 절감**: 로컬 LLM(Mistral, Neural Chat) 사용으로 API 비용 0원
2. **프라이버시**: 모든 데이터가 로컬에서 처리됨
3. **속도**: GPU 있는 Mac에서 매우 빠름 (Apple Silicon 최적화)
4. **확장성**: `voice_analyzer.py`를 다른 프로젝트에 임포트 가능

---

## 🚀 다음 단계

1. **커스텀 LLM 모델**: Ollama에 다른 모델 추가
2. **결과 데이터베이스**: 분석 결과를 PostgreSQL에 저장
3. **대시보드**: 시각화 및 모니터링 대시보드 구축
4. **API 래핑**: FastAPI로 REST API 구축
