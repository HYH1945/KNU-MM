# voice_analyzer.py - 녹음 기능 업데이트

## 변경 사항

voice_analyzer.py의 `record_audio()` 메서드가 다음과 같이 개선되었습니다:

### 1️⃣ 기존 동작 (고정 시간 녹음)

```python
# 정확히 10초 동안 녹음
result = analyzer.transcribe_and_analyze(duration=10)
```

- 무조건 10초 동안 녹음
- 10초가 지나면 자동으로 종료
- 설정된 시간 전에 종료할 수 없음

### 2️⃣ 새로운 기능 (무한 녹음 + Enter 종료)

```python
# Enter 키를 칠 때까지 무한 녹음
result = analyzer.transcribe_and_analyze(duration=None)
```

- 🎤 무한 녹음 시작... (Enter 키를 누르면 종료)
- 사용자가 Enter를 치면 즉시 녹음 중지
- 10초 이상 녹음 필요할 때 유용

## 사용 예제

### 모드 1: 고정 시간 (추천 - 대부분의 경우)

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 15초 녹음 후 자동 종료
result = analyzer.transcribe_and_analyze(duration=15)
print(result['transcribed_text'])
```

**언제 사용?**
- 정확한 시간 제어 필요
- 자동화된 처리
- 배치 작업

### 모드 2: 무한 녹음 (사용자 제어)

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# Enter 키 누를 때까지 녹음
result = analyzer.transcribe_and_analyze(duration=None)
print(result['transcribed_text'])
```

**언제 사용?**
- 대화형 애플리케이션
- 사용자가 음성 길이 제어
- 실시간 인터랙션

### 모드 3: 대화형 메뉴 (명령줄)

```bash
cd /Users/jangjun-yong/Desktop/github/KNU-MM/contextllm
source .venv/bin/activate
python voice_analyzer.py

# 출력:
# 음성 입력 모드를 선택하세요:
# 1. 고정 시간 녹음 (10초)
# 2. 무한 녹음 (Enter로 종료)
# 선택 (1 또는 2): 2
#
# 🎤 무한 녹음 시작... (Enter 키를 누르면 종료)
# [사용자가 말함...]
# [Enter 키 누름]
# ⏹️  녹음 중지 중...
# ✅ 녹음 완료
```

## 기술 고려사항

### ✅ 가능한 것

1. **Enter로 언제든 종료**
   - 무한 녹음 모드에서 사용자가 Enter를 치면 즉시 중단

2. **고정 시간 녹음**
   - 설정된 시간만큼만 녹음

3. **백그라운드에서 Ctrl+C로 강제 종료**
   - KeyboardInterrupt 처리됨

### ❌ 불가능한 것 (기술 한계)

**실시간 음성 처리**는 불가능합니다.

이유:
- Whisper는 **완성된 음성 파일**이 필요
- 실시간으로 중간 결과를 반환할 수 없음
- 스트리밍 음성인식이 아님

### 실시간 처리가 필요한 경우

다음 서비스 사용 권장:

| 서비스 | 장점 | 단점 |
|--------|------|------|
| Google Cloud Speech-to-Text | 실시간 스트리밍 | 비용 발생 |
| Azure Speech Services | 실시간 + 오프라인 | 복잡한 설정 |
| Deepgram | 빠른 실시간 | API 비용 |
| Assembler AI | 정확한 실시간 | API 비용 |

## 구현 방식

### 코어 알고리즘

```
무한 녹음 모드 (duration=None):
1. sox 프로세스를 PIPE 모드로 시작 (trim 없음)
2. input()으로 사용자 Enter 대기 (블로킹)
3. Enter 입력 시 sox 프로세스 terminate()
4. 녹음 파일 크기 확인 (1000 bytes 이상)
5. Whisper 변환 실행
```

### 코드 구조

```python
if duration is None:
    # 무한 녹음: trim 옵션 없음
    sox_process = subprocess.Popen([...])
    input()  # Enter 대기
    sox_process.terminate()
else:
    # 고정 시간: trim 옵션 포함
    subprocess.run([..., 'trim', '0', str(duration)])
```

## 주의사항

1. **최소 음성 길이**: 1000 bytes 이상 필요
2. **Ctrl+C**: KeyboardInterrupt 처리됨
3. **무한 녹음**: sox 프로세스 메모리 사용 계속
4. **파일 포맷**: 항상 WAV, 16000Hz, 모노

## 테스트 방법

```bash
# 방법 1: 고정 시간 (10초)
source .venv/bin/activate
python voice_analyzer.py
# 선택: 1

# 방법 2: 무한 녹음 (Enter로 종료)
source .venv/bin/activate
python voice_analyzer.py
# 선택: 2
# [말하기...]
# [Enter 누르기]

# 방법 3: 프로그래밍
python << 'EOF'
from voice_analyzer import VoiceAnalyzer
analyzer = VoiceAnalyzer()
result = analyzer.transcribe_and_analyze(duration=None)
print(result['transcribed_text'])
EOF
```

## 결론

이제 voice_analyzer.py는 다음 두 가지 녹음 방식을 지원합니다:

✅ **고정 시간 녹음** (duration=10)
✅ **무한 녹음** (duration=None) - Enter로 종료

실시간 처리는 기술 한계로 불가능하지만, 사용자가 원하는 시점에 녹음을 중단할 수 있도록 개선되었습니다! 🎉
