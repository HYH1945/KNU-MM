# 🎤 실제 테스트 & 사용 가이드

## ✅ 현재 상태
- **Ollama**: 이미 실행 중 (포트 11434 사용)
- **Mistral 모델**: 준비됨
- **한국어 분석**: ✅ 정상 작동

---

## 🚀 지금 바로 테스트하기

### 가장 간단한 방법 (권장)

```bash
cd /Users/jangjun-yong/Desktop/jongf1

# 바로 실행!
python3 test_real_analysis.py
```

선택하면:
1. **대화형 모드** (1번): 직접 텍스트 입력
2. **시나리오 테스트** (2번): 6가지 상황 자동 분석
3. **한 가지 분석** (3번): 한 문장만 분석

---

## 📊 Ollama가 분석할 항목 (사용자 요구사항)

| 항목 | 설명 | 예시 |
|------|------|------|
| **Context** | 대화의 맥락 | "시스템 긴급 상황" |
| **Urgency** | 위급도 | 낮음 / 중간 / 높음 / **긴급** |
| **Situation** | 상황 분석 | "서버가 다운되어 고객 접속 불가" |
| **Emotional State** | 감정 상태 | 긍정 / 중립 / 부정 |
| **Action** | 즉시 조치 | "관리자에게 통보 → 복구 시작" |

---

## 💡 실제 테스트 시나리오

### 시나리오 1: 🚨 긴급 상황 (위급도: 긴급)
```
입력: "우리 서버가 완전히 다운 됐어요! 
       고객들이 접속 못 하고 있습니다. 
       매분 매초가 중요합니다!"

분석 결과:
{
  "context": "시스템 긴급 장애",
  "urgency": "긴급",
  "urgency_reason": "고객 서비스 중단으로 비즈니스 영향 직접 발생",
  "situation": "데이터베이스 또는 메인 서버가 다운되어 모든 사용자 접속 불가 상태. 
               실시간 비즈니스 손실 발생 중.",
  "situation_type": "긴급상황",
  "emotional_state": "부정 (높은 스트레스)",
  "action": "1. 즉시 시스템 관리자 호출
           2. 장애 원인 파악
           3. 복구 프로세스 시작",
  "priority": "높음"
}
```

### 시나리오 2: 🟡 일반 업무 (위급도: 중간)
```
입력: "내일 회의 시간을 10시로 변경해주세요."

분석 결과:
{
  "context": "일정 변경 요청",
  "urgency": "중간",
  "urgency_reason": "내일 행사이므로 조기 공지 필요",
  "situation": "회의 일정 변경을 요청하고 있음. 
               내일로 예정되어 있어 빠른 공지가 필요.",
  "situation_type": "업무",
  "action": "회의 참석자에게 변경 내역 통보"
}
```

### 시나리오 3: 🟢 일상 대화 (위급도: 낮음)
```
입력: "오늘 날씨 정말 좋네요."

분석 결과:
{
  "context": "일상 대화",
  "urgency": "낮음",
  "urgency_reason": "실시간 대응이 필요 없는 일상 대화",
  "situation": "사용자가 일상적인 관찰을 공유하고 있음.",
  "situation_type": "일상",
  "action": "응답만 필요"
}
```

---

## 📋 테스트 체크리스트

- [ ] Ollama 포트 확인: `lsof -i :11434`
- [ ] test_real_analysis.py 실행
- [ ] 대화형 모드에서 몇 가지 입력
- [ ] 각 위급도별 분석 결과 확인
- [ ] Ollama 응답 속도 확인

---

## 🎯 프로그램에 통합하는 방법

### 방법 1: voice_analyzer.py 사용

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 음성 녹음 + 분석 + LLM 처리 한 번에
result = analyzer.transcribe_and_analyze(duration=10)

# 결과 확인
print(f"위급도: {result['analysis']['urgency']}")
print(f"상황: {result['analysis']['situation']}")
print(f"조치: {result['analysis']['action']}")

# 위급도에 따라 분기
if result['analysis']['urgency'] == '긴급':
    # 알림 발송
    send_alert()
elif result['analysis']['urgency'] == '높음':
    # 로그 기록
    log_event()
```

### 방법 2: 직접 호출

```python
from test_real_analysis import analyze_voice_input, format_analysis

# 텍스트만 분석 (음성 녹음 X)
text = "지금 상황이 긴급입니다"
analysis = analyze_voice_input(text)

# 위급도 확인
urgency = analysis['urgency']  # '긴급', '높음', '중간', '낮음'

# 조치 수행
if urgency in ['긴급', '높음']:
    trigger_emergency_protocol()
```

---

## ⚡ 명령어 정리

### Ollama 상태 확인
```bash
# Ollama 포트 확인
lsof -i :11434

# 또는 Ollama 모델 목록
ollama list
```

### 테스트 실행
```bash
# 실제 분석 테스트
python3 test_real_analysis.py

# 한국어 분석 테스트
python3 test_korean_analysis.py

# 음성 + 분석 통합
python3 voice_analyzer.py

# 대화형 모드
python3 voice_example.py interactive
```

---

## 🔍 결과 해석 가이드

### Urgency (위급도) 판단 기준

| 위급도 | 응답 시간 | 예시 | 액션 |
|--------|---------|------|------|
| **낮음** | 24시간 | "날씨 좋네요" | 일반 응답 |
| **중간** | 1-2시간 | "일정 변경" | 기록 후 처리 |
| **높음** | 15분 | "버그 있어요" | 즉시 확인 |
| **긴급** | 즉시 | "시스템 다운" | 비상 절차 |

### Situation (상황 분석) 활용

```python
situation = result['analysis']['situation']

# 상황에 따라 다른 응답
if '서버' in situation and '다운' in situation:
    restart_server()
elif '버그' in situation:
    start_debugging()
elif '데이터' in situation and '손실' in situation:
    activate_recovery()
```

---

## 🎁 보너스: 결과 저장 및 추적

모든 분석 결과는 자동으로 저장됨:

```bash
# 오늘 분석 결과 확인
cat transcriptions/2026-01-21/transcriptions.json | jq '.[] | {timestamp, urgency: .analysis.urgency, situation: .analysis.situation}'

# 긴급 상황만 필터링
cat transcriptions/2026-01-21/transcriptions.json | jq '.[] | select(.analysis.urgency == "긴급")'
```

---

## 🚀 다음 단계

1. **test_real_analysis.py 실행** → 성능 확인
2. **실제 프로그램에 통합** → voice_analyzer 임포트
3. **위급도 기반 분기** → urgency에 따라 다른 처리
4. **자동화** → cron 또는 스케줄러로 정기 실행

---

## 💬 팁

✅ **빠른 테스트**:
```bash
python3 test_real_analysis.py  # 숫자 2 선택
```

✅ **프로그램 통합**:
```python
from voice_analyzer import VoiceAnalyzer
analyzer = VoiceAnalyzer()
result = analyzer.transcribe_and_analyze(duration=10)
```

✅ **맥락 기반 대응**:
```python
urgency = result['analysis']['urgency']
if urgency == '긴급':
    take_emergency_action()
```

이제 **실제로 필요한 분석**을 할 수 있습니다! 🚀
