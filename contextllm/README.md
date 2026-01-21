# 🎤 로컬 Whisper + LLM 음성 인식 시스템

**macOS에서 로컬 AI를 사용한 실시간 음성 인식 및 상황 분석 시스템**

- 🎵 **로컬 음성 인식**: OpenAI Whisper (외부 API 불필요)
- 🤖 **상황 분석**: Ollama + Mistral LLM (로컬 실행)
- 🔐 **프라이빗**: 모든 데이터가 컴퓨터 내에서 처리됨
- 💬 **한국어 지원**: 한국어 문맥 분석 완벽 지원

---

## 🚀 빠른 시작

### 1️⃣ 필수 도구 설치

```bash
# Ollama 설치 (macOS)
# https://ollama.ai에서 다운로드 또는
brew install ollama

# sox 설치 (음성 녹음용)
brew install sox

# Python 패키지
pip install requests
```

### 2️⃣ 프로젝트 설정

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/jongf1.git
cd jongf1

# 가상환경 생성 (이미 있으면 스킵)
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3️⃣ Ollama 실행

```bash
# 터미널 1
ollama serve

# 터미널 2 (선택): Mistral 모델 다운로드
ollama pull mistral
```

### 4️⃣ 음성 인식 시작

```bash
# 터미널 3
cd jongf1

# 음성 녹음 + LLM 분석
python3 voice_analyzer.py
```

---

## 📊 기능

### 1. 음성 인식 → 텍스트 변환 (Whisper)

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 10초 음성 녹음 후 자동 변환
result = analyzer.transcribe_and_analyze(duration=10)

print(result['transcribed_text'])  # "안녕하세요..."
```

### 2. 상황 분석 (LLM)

```python
# Mistral이 자동으로 분석
analysis = result['analysis']

print(f"맥락: {analysis['context']}")          # "일상 대화"
print(f"위급도: {analysis['urgency']}")        # "낮음/중간/높음/긴급"
print(f"상황: {analysis['situation']}")        # "사용자가..."
print(f"감정: {analysis['emotional_state']}")  # "긍정/중립/부정"
```

### 3. 자동 저장

```
transcriptions/
├── 2026-01-21/
│   ├── transcriptions.json    # 모든 결과 (배열)
│   ├── latest.json            # 최신 결과
│   └── 2026-01-21T...json     # 개별 결과
```

---

## 🎯 사용 예제

### 회의 기록 분석

```python
analyzer = VoiceAnalyzer()

result = analyzer.transcribe_and_analyze(
    duration=30,
    system_prompt="""회의 내용 분석 (JSON):
{
  "topic": "주제",
  "decisions": ["결정사항"],
  "action_items": [{"task": "할일", "owner": "담당자"}],
  "urgency": "우선순위"
}"""
)

# 결과
print(result['analysis']['decisions'])
```

### 고객 지원 분석

```python
result = analyzer.transcribe_and_analyze(
    duration=15,
    system_prompt="""고객 피드백 분석:
{
  "sentiment": "긍정/중립/부정",
  "issue": "문제",
  "priority": "우선순위"
}"""
)

# 위급도 판단
if result['analysis']['urgency'] == '긴급':
    send_alert()
```

---

## 📁 파일 구조

```
jongf1/
├── voice_analyzer.py           # ⭐ 핵심 모듈 (사용!)
├── voice_example.py            # 사용 예제
├── voice_monitor.py            # 자동 모니터링
├── api_server.py               # REST API 서버
├── test_real_analysis.py       # 실시간 분석 테스트
├── test_korean_analysis.py     # 한국어 분석 테스트
├── whisper_service.py          # Whisper 변환 스크립트
│
├── src/
│   └── extension.ts            # VS Code 확장 (선택)
│
├── transcriptions/             # 결과 저장 폴더
├── recordings/                 # 임시 음성 파일
│
├── REAL_TEST_GUIDE.md          # 테스트 가이드
├── OLLAMA_GUIDE.md             # Ollama 설정 가이드
├── SETUP_GUIDE.md              # 설정 가이드
├── README.md                   # 이 파일
├── requirements.txt            # Python 의존성
├── package.json                # Node.js 의존성 (선택)
└── .gitignore                  # Git 제외 파일
```

---

## 🔧 명령어

### 기본 사용

```bash
# 10초 음성 녹음 + 분석
python3 voice_analyzer.py

# 대화형 모드 (여러 번)
python3 voice_example.py interactive

# 시나리오 테스트
python3 test_real_analysis.py

# 한국어 분석 테스트
python3 test_korean_analysis.py
```

### Ollama 관리

```bash
# Ollama 서버 시작
ollama serve

# 설치된 모델 확인
ollama list

# 모델 다운로드
ollama pull mistral
ollama pull neural-chat

# 모델 삭제
ollama rm mistral
```

---

## 🛠️ 트러블슈팅

### 문제 1: "Ollama 연결 실패"

```bash
# Ollama 실행 중인지 확인
lsof -i :11434

# 없으면 시작
ollama serve
```

### 문제 2: "Whisper 모듈 없음"

```bash
# 가상환경 활성화
source .venv/bin/activate

# Whisper 설치
pip install openai-whisper
```

### 문제 3: "sox 없음"

```bash
# macOS
brew install sox

# Linux
sudo apt-get install sox
```

---

## 📊 성능

| 기능 | 시간 | 리소스 |
|------|------|--------|
| 10초 음성 녹음 | 10초 | 낮음 |
| Whisper 변환 | 3-5초 | 중간 (GPU 사용) |
| Mistral 분석 | 2-3초 | 중간 |
| **전체** | **15-20초** | **낮음** |

---

## 🔐 프라이버시

✅ **완전한 로컬 처리**
- 인터넷 불필요
- OpenAI API 사용 안 함
- Ollama 로컬 모델 사용
- 모든 데이터가 내 컴퓨터에서만 처리

---

## 📝 라이선스

MIT License

---

## 🤝 기여

버그 리포트나 기능 요청은 Issues 탭에서 해주세요.

---

## 💡 팁

1. **더 빠른 응답**: `neural-chat` 모델 사용
   ```bash
   ollama pull neural-chat
   ```

2. **메모리 절약**: 사용하지 않는 모델 삭제
   ```bash
   ollama rm mistral
   ```

3. **GPU 가속**: Apple Silicon Mac에서 자동으로 사용됨

4. **결과 확인**:
   ```bash
   cat transcriptions/*/latest.json
   ```

---

## 🚀 다음 단계

1. [REAL_TEST_GUIDE.md](REAL_TEST_GUIDE.md) - 테스트 방법
2. [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) - Ollama 상세 설정
3. [SETUP_GUIDE.md](SETUP_GUIDE.md) - 전체 설정 가이드

---

**Happy voice transcribing! 🎤**
