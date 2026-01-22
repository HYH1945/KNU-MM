# 🚀 Google Cloud Speech-to-Text 실시간 분석

## 진정한 실시간 음성 처리 (문장 단위 LLM 분석)

사용자 아이디어: **Google Speech-to-Text로 실시간 스트리밍 받아오고, 문장마다 LLM 분석**

**결과: 진정한 실시간 처리 가능! ✨**

---

## 🎯 작동 원리

```
마이크 입력 (실시간)
    ↓
Google Cloud Speech-to-Text (스트리밍)
    ↓
부분 결과 표시 (interim): ⏳ [입력중] "안녕하세..."
    ↓
문장 완성 시 (is_final=True): ✅ [최종] "안녕하세요"
    ↓
즉시 LLM 분석: 🧠 감정/위급도/의도 분석
    ↓
결과 출력: 📊 실시간 분석 결과
```

**지연 시간: ~1-2초 (진정한 실시간!)**

---

## 📋 설치 및 설정

### 1단계: 라이브러리 설치

```bash
# Google Cloud Speech-to-Text + PyAudio
pip install google-cloud-speech pyaudio
```

**macOS 문제 해결:**
```bash
# PyAudio 설치 문제 시
brew install portaudio
pip install pyaudio
```

### 2단계: Google Cloud 인증 설정

```bash
# 1. Google Cloud 프로젝트 생성
# https://console.cloud.google.com

# 2. Speech-to-Text API 활성화
# 3. 서비스 계정 생성 및 JSON 키 다운로드

# 4. 환경 변수 설정
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# 또는 .bashrc/.zshrc에 추가
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"' >> ~/.zshrc
```

### 3단계: Ollama 서버 실행

```bash
ollama serve &
ollama pull mistral  # 또는 선택한 모델
```

---

## 🎤 사용 방법

### 방법 1: 명령줄 (가장 간단)

```bash
cd /Users/jangjun-yong/Desktop/github/KNU-MM/contextllm
source .venv/bin/activate

# 직접 실행
python google_realtime_analyzer.py

# 출력:
# ============================================================
# ⚡ Google Speech-to-Text 실시간 모니터링
# 문장이 완성되면 자동으로 LLM 분석
# ============================================================
# 
# 🎤 마이크 입력 중... (Ctrl+C로 종료)
# 
# ⏳ [입력중] "안녕하세요..."
# ⏳ [입력중] "안녕하세요 날씨가..."
# ✅ [최종] "안녕하세요 날씨가 정말 좋네요"
# 🧠 LLM 분석 중...
# 📊 분석 결과:
#   😊 감정: 긍정
#   🚨 위급도: 낮음
#   💭 의도: 인사 및 날씨 언급
#   🏷️  키워드: ['인사', '날씨', '긍정']
```

### 방법 2: Python 코드

```python
from google_realtime_analyzer import GoogleRealtimeAnalyzer

analyzer = GoogleRealtimeAnalyzer()

# 무한 실시간 모니터링
results = analyzer.listen_and_analyze_realtime()

# 또는 최대 60초 실행
results = analyzer.listen_and_analyze_realtime(max_duration=60)

# 결과 확인
for result in results:
    print(f"음성: {result['text']}")
    print(f"분석: {result['analysis']}")
```

### 방법 3: 커스텀 프롬프트

```python
from google_realtime_analyzer import GoogleRealtimeAnalyzer

analyzer = GoogleRealtimeAnalyzer()

# 비상 상황 감지 프롬프트
emergency_prompt = """
비상 상황을 감지하는 전문가입니다.
다음을 JSON으로 분석하세요:
{
  "is_emergency": true/false,
  "emergency_type": "화재/의료/범죄/기타",
  "action_needed": "해야할 조치"
}
"""

results = analyzer.listen_and_analyze_realtime(
    system_prompt=emergency_prompt,
    max_duration=120
)
```

---

## ⚡ 성능 특성

### 지연 시간

```
음성 입력 → Google STT 인식 → LLM 분석 → 결과 출력
  ~500ms    ~500-1000ms     ~1000-2000ms  ~100ms
  
총: ~2-3초 (진정한 실시간!)
```

### 비교

| 방식 | 지연시간 | 특징 |
|------|---------|------|
| Whisper 순차 | 15-20초 | 느림 |
| Whisper 병렬 | ~10초 | 거의 실시간 |
| **Google STT** | **~2-3초** | **진정한 실시간! ⚡** |

### 리소스 사용

| 항목 | 사용량 |
|------|--------|
| CPU | 20-30% |
| 메모리 | 500MB-1GB |
| 네트워크 | Google Cloud API 호출 (비용 발생) |

---

## 💰 비용 고려사항

### Google Cloud Speech-to-Text 가격

- **월 60분 무료** (항상)
- 초과 시: **$0.024/분** (약 1시간 = $1.44)

### 절약 팁

1. **로컬 Whisper 사용** (무료)
   ```python
   # voice_analyzer.py 사용
   analyzer.run_parallel_realtime()  # 무료, ~10초 지연
   ```

2. **혼합 방식** (선택적)
   - 높은 정확도 필요 시: Google STT
   - 일반 모니터링: Whisper

---

## 🔧 문제 해결

### Q1: "Google Application Credentials not found"

```bash
# 인증 파일 확인
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# 경로 재설정
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/credentials.json"

# 확인
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### Q2: PyAudio 설치 실패

```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu
sudo apt-get install portaudio19-dev
pip install pyaudio

# Windows
# 바이너리 다운로드: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

### Q3: 마이크가 인식 안 됨

```bash
# 마이크 확인
python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"

# macOS 권한 설정 필요할 수 있음
# System Preferences → Security & Privacy → Microphone
```

### Q4: Google API 오류

```bash
# API 활성화 확인
# https://console.cloud.google.com → Speech-to-Text API → ENABLE

# 기한만료 확인
# gcloud auth list
# gcloud auth login
```

---

## 📊 실제 사용 사례

### 사례 1: 비상 상황 실시간 감지

```python
emergency_prompt = """
비상 상황을 감지합니다:
- "도와줘", "긴급", "119", "112" 감지 시 즉시 알림
{
  "emergency": true/false,
  "action": "경찰/소방/119 호출 필요"
}
"""

analyzer = GoogleRealtimeAnalyzer()
results = analyzer.listen_and_analyze_realtime(
    system_prompt=emergency_prompt
)

# 실시간 감지 가능!
```

### 사례 2: 고객 통화 분석

```python
# 고객 감정 실시간 분석
support_prompt = """
고객 감정을 분석합니다:
{
  "sentiment": "긍정/중립/부정",
  "satisfaction": "매우만족/만족/보통/불만족/매우불만족",
  "action_items": ["필요한 조치들"]
}
"""

analyzer.listen_and_analyze_realtime(system_prompt=support_prompt)
```

### 사례 3: 회의 실시간 기록

```python
meeting_prompt = """
회의 내용을 실시간 기록:
{
  "speaker": "발언자",
  "topic": "주제",
  "decision": "결정사항",
  "action_item": "할일"
}
"""

results = analyzer.listen_and_analyze_realtime(
    system_prompt=meeting_prompt,
    max_duration=3600  # 1시간
)

# 회의 내용 자동 기록
```

---

## 🎯 기술 비교 정리

```
┌─────────────────────────────────────────────────┐
│ 방식별 실시간성 비교                              │
├─────────────────────────────────────────────────┤
│ 1️⃣  Whisper 순차 처리                            │
│    녹음(10s) → 변환(2-5s) → 분석(2-3s)          │
│    총: 15-20초 간격  ❌ 비추천                   │
│                                               │
│ 2️⃣  Whisper 병렬 처리                            │
│    스레드1: 녹음 10초                           │
│    스레드2: 동시 분석                           │
│    총: ~10초 간격  ⚠️  괜찮음                   │
│                                               │
│ 3️⃣  Google Speech-to-Text (NEW!)                │
│    실시간 스트리밍 → 문장마다 분석               │
│    총: ~2-3초 간격  ✅ 진정한 실시간!           │
│    비용: 월 60분 무료 + 추가 $0.024/분         │
└─────────────────────────────────────────────────┘
```

---

## 결론

**Google Cloud Speech-to-Text를 사용하면:**

✅ **진정한 실시간 처리** (~2-3초 지연)  
✅ **문장 단위 분석** (완성된 문장만 분석)  
✅ **부분 결과 표시** (입력 중 미리보기)  
✅ **높은 정확도** (Google의 강력한 음성인식)

❌ **비용 발생** (월 60분 무료, 이후 유료)  
❌ **인터넷 필요** (API 호출)  
❌ **설정 복잡** (Google Cloud 인증)

**추천:**
- 높은 정확도 + 실시간 필요 → Google STT ⭐
- 비용 절감 + 로컬 처리 → Whisper 병렬 처리
