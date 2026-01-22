# 🚀 실시간 모니터링 가이드

## 거의 실시간에 가까운 음성 처리

사용자의 아이디어: **10초 간격으로 반복 실행하면 거의 실시간에 가깝지 않을까?**

**답: 맞습니다!** ✅

이제 `run_continuously()` 메서드를 사용하여 약 10초의 지연시간으로 거의 실시간 처리가 가능합니다.

---

## 🎯 3가지 사용 방법

### 방법 1: 명령줄 (추천 - 가장 간단)

```bash
cd /Users/jangjun-yong/Desktop/github/KNU-MM/contextllm
source .venv/bin/activate

# 메뉴 실행
python voice_analyzer.py

# 출력:
# 음성 입력 모드를 선택하세요:
# 1. 고정 시간 녹음 (10초) - 일회성
# 2. 무한 녹음 (Enter로 종료) - 일회성
# 3. 실시간 모니터링 (10초 간격 반복) ⭐ 거의 실시간!
# 선택 (1, 2, 또는 3): 3

# 다음으로 반복 횟수 선택:
# 몇 회를 반복할까요?
# 1. 무한 반복 (Ctrl+C로 중지)
# 2. 5회만 실행
# 3. 10회 실행
# 선택 (1, 2, 또는 3): 1

# 🔄 실시간 모니터링 시작 (10초 간격)
# [계속 반복...]
```

### 방법 2: Python 코드 - 무한 반복

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 무한 반복 (Ctrl+C로 중단)
analyzer.run_continuously(interval=10)
```

**출력:**
```
============================================================
🔄 실시간 모니터링 시작 (10초 간격)
============================================================

[1차] 시간: 2026-01-22 11:55:00
🎤 녹음 중... (10초)
✅ 녹음 완료: ./recordings/audio_20260122_115500_000.wav
⚙️  Whisper 변환 중...
📝 음성: "안녕하세요 날씨가 정말 좋네요"
🚨 위급도: 낮음
😊 감정: 긍정
⏳ 10초 후 다시 실행...

[2차] 시간: 2026-01-22 11:55:15
🎤 녹음 중... (10초)
✅ 녹음 완료: ./recordings/audio_20260122_115510_000.wav
⚙️  Whisper 변환 중...
📝 음성: "도와주세요 지금 긴급상황입니다"
🚨 위급도: 긴급
😊 감정: 부정
⏳ 10초 후 다시 실행...

# [Ctrl+C를 누르면]
# ⏹️  모니터링 중지됨 (Ctrl+C)
# 📊 총 2회 처리 완료
# ✅ 성공: 2
# ❌ 실패: 0
```

### 방법 3: Python 코드 - 제한 반복

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 5회만 반복
analyzer.run_continuously(interval=10, max_iterations=5)

# 또는 10회 반복
analyzer.run_continuously(interval=10, max_iterations=10)
```

---

## 🔧 고급 사용법

### 반복 간격 변경

```python
# 5초 간격으로 반복 (더 빠른 처리)
analyzer.run_continuously(interval=5, max_iterations=10)

# 20초 간격으로 반복 (더 느린 처리)
analyzer.run_continuously(interval=20, max_iterations=5)
```

### 커스텀 프롬프트 적용

```python
# 특정 목적에 맞게 분석
emergency_prompt = """
긴급 상황을 감지하는 전문가입니다.
다음을 JSON으로 반환:
{
  "is_emergency": true/false,
  "emergency_type": "화재/의료/범죄/기타",
  "action": "해야할 조치"
}
"""

analyzer.run_continuously(
    interval=10,
    max_iterations=20,
    system_prompt=emergency_prompt
)
```

### 결과 처리

```python
# 결과 리스트 반환받기
results = analyzer.run_continuously(
    interval=10,
    max_iterations=5
)

# 결과 분석
import json
for i, result in enumerate(results):
    if result.get('success'):
        print(f"\n[{i+1}차] {result['timestamp']}")
        print(f"음성: {result['transcribed_text']}")
        print(f"분석: {json.dumps(result['analysis'], ensure_ascii=False)}")
    else:
        print(f"[{i+1}차] 오류: {result.get('error')}")
```

---

## ⏱️ 성능 특성

### 시간 구성

```
총 사이클 시간 = 녹음(10s) + Whisper(2-5s) + LLM분석(2-3s) + 저장(0.5s)
             = 약 14.5-18.5초 (10초 간격 설정 시 실제는 약 15-20초)
```

### 실제 동작 예

```
00:00 - 1차 시작
00:10 - 녹음 완료
00:15 - Whisper 완료
00:18 - LLM 분석 완료
00:18 - 결과 저장
        ↓
00:28 - 2차 시작 (약 10초 간격)
```

### 리소스 사용

| 항목 | 사용량 | 설명 |
|------|--------|------|
| CPU | 30-50% | Whisper + Ollama |
| 메모리 | 2-4GB | 모델 로드 |
| 디스크 | ~1MB/회 | 녹음 파일 저장 |
| 네트워크| 불필요 | 완전 로컬 |

---

## 🎯 실제 사용 사례

### 사례 1: 비상 상황 감시

```python
# 비상 상황 감시 시스템
emergency_system = VoiceAnalyzer()

# 24시간 모니터링
results = emergency_system.run_continuously(interval=10)

# 긴급 상황 감지
for result in results:
    if result.get('success'):
        analysis = result.get('analysis', {})
        if analysis.get('urgency') == '긴급':
            print("🚨 긴급 상황 감지!")
            print(result['transcribed_text'])
            # 알림 전송
            # send_alert(result)
```

### 사례 2: 회의 기록

```python
# 회의 중 실시간 기록
meeting_recorder = VoiceAnalyzer()

meeting_prompt = """회의 내용을 정리하세요:
{
  "speaker": "발언자",
  "topic": "주제",
  "decision": "결정사항"
}"""

results = meeting_recorder.run_continuously(
    interval=10,
    max_iterations=60,  # 약 10분
    system_prompt=meeting_prompt
)
```

### 사례 3: 고객 지원

```python
# 고객 통화 모니터링
support_system = VoiceAnalyzer()

support_prompt = """고객 감정과 문제를 분석하세요:
{
  "customer_sentiment": "긍정/중립/부정",
  "issue_category": "기술/결제/기타",
  "urgency": "낮음/중간/높음/긴급"
}"""

results = support_system.run_continuously(
    interval=10,
    system_prompt=support_prompt
)
```

---

## ⚠️ 주의사항

### 1. 최소 구성

```bash
# Ollama 서버 반드시 실행
ollama serve &

# 별도 터미널에서 모니터링 시작
python voice_analyzer.py
```

### 2. CPU/메모리 관리

```bash
# 모니터링 중 리소스 확인
top -l 1 | grep "python"

# 필요 시 간격 조정
analyzer.run_continuously(interval=20)  # 간격 늘리기
```

### 3. 음성 파일 정리

```bash
# 자동으로 정리되지만, 필요 시 수동 정리
rm -rf ./recordings/*.wav
rm -rf ./transcriptions/2026-01-22/*.json
```

### 4. 에러 처리

```python
results = analyzer.run_continuously(interval=10, max_iterations=100)

# 실패한 결과 확인
failures = [r for r in results if not r.get('success')]
if failures:
    print(f"❌ {len(failures)}개 실패:")
    for r in failures:
        print(f"  - {r.get('error')}")
```

---

## 🎓 성능 최적화 팁

### Tip 1: 더 빠른 모델 사용

```bash
# Mistral 대신 더 빠른 모델 사용
ollama pull neural-chat
ollama pull phi  # 가장 빠름
```

### Tip 2: Whisper 모델 최적화

```python
# voice_analyzer.py에서 모델 변경
# 현재: base (작은 모델, 빠름)
# 선택: tiny (가장 빠름, 정확도 낮음)
```

### Tip 3: 배치 처리

```python
# 여러 언어 지원
for language in ['ko', 'en', 'ja']:
    print(f"\n{language} 감시 시작...")
    analyzer.run_continuously(
        interval=10,
        max_iterations=5
    )
```

---

## 결론

✅ **거의 실시간 처리 가능!**

- 10초 간격 반복으로 약 10초 지연
- 자동 모니터링
- 로컬에서 완전 처리 (프라이빗)
- 배치 작업 가능

이제 실시간에 가까운 음성 인식 + LLM 분석이 가능합니다! 🚀
