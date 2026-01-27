# 🎥 멀티모달 컨텍스트 분석 가이드

이 프로젝트는 **음성 + 이미지/비디오**를 함께 분석하여 더 정확한 상황 판단을 제공합니다.

---

## 📋 목차

1. [개요](#개요)
2. [왜 멀티모달인가?](#왜-멀티모달인가)
3. [시스템 구조](#시스템-구조)
4. [사용 가능한 모드](#사용-가능한-모드)
5. [설치 및 설정](#설치-및-설정)
6. [사용 방법](#사용-방법)
7. [고급 기능](#고급-기능)
8. [문제 해결](#문제-해결)

---

## 개요

### 기존 시스템의 한계

기존에는 **오디오만**으로 컨텍스트를 탐지했습니다:

```
🎤 음성 입력 → 🤖 LLM 분석 → 📊 상황 판단
```

**문제점:**
- 음성 내용만으로는 실제 상황을 정확히 파악하기 어려움
- 거짓 긴급 상황 감지 (false positive)
- 시각적 증거 부재

### 멀티모달 시스템의 장점

이제는 **음성 + 이미지/비디오**를 함께 분석합니다:

```
🎤 음성 입력  ╲
               ╲
                ⊕ → 🤖 LLM (멀티모달) → 📊 더 정확한 상황 판단
               ╱
📸 이미지/비디오 ╱
```

**장점:**
- 시각적 증거로 음성 내용 검증
- 더 정확한 긴급 상황 감지
- 음성과 시각 정보의 일치도 확인
- 컨텍스트의 풍부함

---

## 왜 멀티모달인가?

### 실제 사용 사례

#### 사례 1: 긴급 상황 검증

**음성만:**
```
사용자: "불이야!"
→ 시스템: 🚨 긴급! (하지만 실제로는 게임 중일 수도...)
```

**멀티모달:**
```
사용자: "불이야!"
+ 이미지: [컴퓨터 화면에 게임]
→ 시스템: ℹ️ 일상 대화 (게임 플레이 중)
```

#### 사례 2: 컨텍스트 이해

**음성만:**
```
사용자: "지금 뭐 하고 있어?"
→ 시스템: ❓ 애매함
```

**멀티모달:**
```
사용자: "지금 뭐 하고 있어?"
+ 이미지: [코드 에디터 화면]
→ 시스템: 💻 개발 작업 중 (코딩)
```

#### 사례 3: 실제 위험 감지

**음성만:**
```
사용자: "..."  (침묵)
→ 시스템: (아무것도 감지 못함)
```

**멀티모달:**
```
사용자: "..."  (침묵)
+ 이미지: [쓰러진 사람]
→ 시스템: 🚨 긴급! 의료 응급 상황 감지
```

---

## 시스템 구조

### 1. 핵심 모듈

```
src/core/
├── multimodal_analyzer.py    # 멀티모달 분석 엔진
├── video_capture.py           # 비디오/웹캠 캡처
├── voice_analyzer.py          # 음성 분석 (기존)
└── alert_manager.py           # 긴급 알림 관리
```

### 2. 테스트 파일

```
tests/
├── test_multimodal_screenshot.py   # 음성 + 스크린샷
├── test_multimodal_webcam.py       # 음성 + 웹캠
└── test_free_realtime_simple.py    # 음성만 (기존)
```

### 3. 데이터 흐름

```
┌─────────────┐
│  마이크 입력  │ → 음성 인식 → 텍스트
└─────────────┘                ↓
                              합치기
┌─────────────┐                ↓
│ 스크린샷/웹캠 │ → 이미지 캡처 → 이미지
└─────────────┘                ↓
                              GPT-4o (Vision)
                                ↓
                        ┌───────────────┐
                        │ 멀티모달 분석  │
                        │ - 상황 파악    │
                        │ - 일치도 검증  │
                        │ - 긴급도 판단  │
                        └───────────────┘
                                ↓
                        ┌───────────────┐
                        │  결과 출력     │
                        │ + 긴급 알림    │
                        │ + 로그 저장    │
                        └───────────────┘
```

---

## 사용 가능한 모드

### 모드 1: 음성 + 스크린샷

화면 캡처와 함께 음성을 분석합니다.

**사용 시나리오:**
- 원격 근무 모니터링
- 화면 기반 작업 분석
- 프레젠테이션 상황 파악

**실행 방법:**
```bash
python tests/test_multimodal_screenshot.py
```

**장점:**
- 화면 전체 캡처
- 시스템 리소스 적음
- 빠른 처리 속도

**단점:**
- 사용자 얼굴/표정 캡처 불가
- 실시간 동영상 분석 불가

### 모드 2: 음성 + 웹캠

웹캠에서 실시간 영상과 함께 음성을 분석합니다.

**사용 시나리오:**
- 원격 회의 모니터링
- 보안 감시
- 의료 응급 감지
- 사용자 표정 분석

**실행 방법:**
```bash
python tests/test_multimodal_webcam.py
```

**장점:**
- 실시간 사용자 모니터링
- 표정/감정 분석 가능
- 물리적 환경 파악

**단점:**
- 웹캠 필요
- 시스템 리소스 많이 사용
- 프라이버시 이슈

### 모드 3: 음성만 (기존)

음성만으로 분석합니다.

**실행 방법:**
```bash
python tests/test_free_realtime_simple.py
```

---

## 설치 및 설정

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

**추가 패키지:**
```bash
pip install opencv-python Pillow
```

### 2. OpenAI API 키 설정

멀티모달 분석에는 **GPT-4o** (Vision 지원)가 필요합니다.

```bash
# config/.env 파일
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. 웹캠 권한 설정 (macOS)

**시스템 환경설정 → 보안 및 개인 정보 보호 → 카메라**

Terminal.app에 카메라 접근 권한 부여

### 4. 스크린샷 권한 설정 (macOS)

**시스템 환경설정 → 보안 및 개인 정보 보호 → 화면 기록**

Terminal.app에 화면 기록 권한 부여

---

## 사용 방법

### 기본 사용법

#### 1. 음성 + 스크린샷 모드

```bash
# 터미널에서 실행
python tests/test_multimodal_screenshot.py

# 프로그램이 시작되면 마이크로 말하기
"도와주세요!"  # 음성 입력
# → 자동으로 스크린샷 캡처
# → GPT-4o가 음성 + 이미지 분석
# → 결과 출력
```

#### 2. 음성 + 웹캠 모드

```bash
# 터미널에서 실행
python tests/test_multimodal_webcam.py

# 프로그램이 시작되면:
# 1. 웹캠 자동 시작 (백그라운드)
# 2. 마이크로 말하기
"이상한 소리가 나요"  # 음성 입력
# → 자동으로 웹캠 프레임 캡처
# → GPT-4o가 음성 + 비디오 분석
# → 결과 출력
```

### 프로그램 종료

음성으로 `"quit"`, `"exit"`, `"종료"` 중 하나를 말하거나  
`Ctrl+C`를 누르세요.

---

## 고급 기능

### 1. 프로그래밍 방식 사용

#### 스크린샷과 함께 분석

```python
from core.multimodal_analyzer import MultimodalAnalyzer

analyzer = MultimodalAnalyzer()

# 방법 1: 스크린샷 자동 캡처
screenshot = analyzer.capture_screenshot()
result = analyzer.analyze_with_image(
    audio_text="도와주세요!",
    image_source=screenshot
)

# 방법 2: 기존 이미지 파일 사용
result = analyzer.analyze_with_image(
    audio_text="이게 뭔가요?",
    image_source="path/to/image.jpg"
)

print(result)
```

#### 웹캠과 함께 분석

```python
from core.multimodal_analyzer import MultimodalAnalyzer

analyzer = MultimodalAnalyzer()

# 웹캠에서 프레임 캡처
frame = analyzer.capture_webcam_frame(camera_id=0)

# 분석
result = analyzer.analyze_with_video_frame(
    audio_text="지금 상황이 어떤가요?",
    frame=frame
)

print(result)
```

### 2. 비디오 파일에서 프레임 추출

```python
from core.video_capture import VideoFrameExtractor

extractor = VideoFrameExtractor()

# 1초마다 프레임 추출
frames = extractor.extract_frames(
    video_path="video.mp4",
    interval=1.0,
    save_dir="extracted_frames/"
)

# 각 프레임 분석
for timestamp, frame in frames:
    result = analyzer.analyze_with_video_frame(
        audio_text=f"{timestamp}초의 상황",
        frame=frame
    )
    print(f"[{timestamp}초] {result}")
```

### 3. 실시간 비디오 모니터링

```python
from core.video_capture import VideoMonitor
from core.multimodal_analyzer import MultimodalAnalyzer

analyzer = MultimodalAnalyzer()

def on_frame(frame, timestamp):
    """프레임마다 호출되는 콜백"""
    # 음성은 별도로 입력받아야 함
    audio_text = get_audio_input()  # 사용자 구현
    
    result = analyzer.analyze_with_video_frame(audio_text, frame)
    
    if result.get('is_emergency'):
        print(f"🚨 긴급! {result['emergency_reason']}")

monitor = VideoMonitor(camera_id=0)
monitor.start_monitoring(
    on_frame_callback=on_frame,
    frame_interval=2.0  # 2초마다 분석
)

# 모니터링 중지
# monitor.stop_monitoring()
```

### 4. 움직임 감지와 결합

```python
from core.video_capture import MotionDetector, VideoMonitor

detector = MotionDetector(threshold=25.0)

def on_frame(frame, timestamp):
    has_motion, motion_areas = detector.detect_motion(frame)
    
    if has_motion:
        print(f"⚠️  움직임 감지! ({len(motion_areas)}개 영역)")
        
        # 멀티모달 분석 트리거
        result = analyzer.analyze_with_video_frame(
            audio_text="움직임이 감지되었습니다",
            frame=frame
        )
        print(result)

monitor = VideoMonitor()
monitor.start_monitoring(on_frame_callback=on_frame)
```

---

## 분석 결과 형식

멀티모달 분석 결과는 다음과 같은 JSON 형식입니다:

```json
{
  "context": "음성과 이미지를 종합한 맥락 설명",
  "urgency": "긴급",
  "urgency_reason": "화재 징후 감지",
  "situation": "화면에서 연기가 보이고 사용자가 '불이야'라고 외침",
  "situation_type": "긴급",
  "emotional_state": "공포",
  "visual_content": "화면에 연기와 불꽃이 보임",
  "audio_visual_consistency": "일치",
  "action": "즉시 119 신고 및 대피",
  "is_emergency": true,
  "emergency_reason": "화재 발생",
  "priority": "CRITICAL"
}
```

### 주요 필드

| 필드 | 설명 | 예시 값 |
|------|------|---------|
| `visual_content` | 이미지/비디오에서 관찰된 내용 | "컴퓨터 화면에 코드 에디터" |
| `audio_visual_consistency` | 음성-시각 일치도 | "일치", "불일치", "부분일치" |
| `is_emergency` | 긴급 여부 | `true`, `false` |
| `priority` | 우선순위 | "CRITICAL", "HIGH", "MEDIUM", "LOW" |

---

## 문제 해결

### 1. 웹캠이 작동하지 않음

**증상:**
```
❌ 카메라 0를 열 수 없습니다
```

**해결 방법:**

1. **권한 확인**
   ```
   시스템 환경설정 → 보안 및 개인 정보 보호 → 카메라
   → Terminal.app 체크
   ```

2. **다른 카메라 ID 시도**
   ```python
   monitor = VideoMonitor(camera_id=1)  # 0 대신 1 시도
   ```

3. **다른 앱이 웹캠 사용 중인지 확인**
   - Zoom, FaceTime 등 종료

### 2. 스크린샷이 캡처되지 않음

**증상:**
```
❌ 스크린샷 캡처 실패
```

**해결 방법:**

1. **권한 확인**
   ```
   시스템 환경설정 → 보안 및 개인 정보 보호 → 화면 기록
   → Terminal.app 체크
   ```

2. **screencapture 명령 테스트**
   ```bash
   screencapture -x test.png
   ```

### 3. OpenAI API 오류

**증상:**
```
❌ OPENAI_API_KEY가 .env에 설정되지 않았습니다
```

**해결 방법:**

1. **API 키 확인**
   ```bash
   cat config/.env | grep OPENAI_API_KEY
   ```

2. **.env 파일 생성**
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > config/.env
   ```

3. **GPT-4o 지원 확인**
   - GPT-4o는 Vision 기능을 지원합니다
   - 일반 GPT-4는 이미지 분석 불가

### 4. 메모리 부족

**증상:**
```
Out of memory 또는 느린 처리 속도
```

**해결 방법:**

1. **프레임 간격 늘리기**
   ```python
   monitor.start_monitoring(
       on_frame_callback=on_frame,
       frame_interval=5.0  # 1초 → 5초로 증가
   )
   ```

2. **이미지 해상도 낮추기**
   ```python
   import cv2
   frame_small = cv2.resize(frame, (640, 480))
   ```

3. **스크린샷 모드 사용**
   - 웹캠 대신 스크린샷 모드 사용

### 5. 음성 인식 실패

**증상:**
```
⚠️  음성을 인식하지 못했습니다
```

**해결 방법:**

1. **마이크 권한 확인**
2. **배경 소음 줄이기**
3. **마이크 음량 확인**
4. **더 크고 명확하게 말하기**

---

## 성능 최적화

### 리소스 사용량 비교

| 모드 | CPU | 메모리 | 네트워크 |
|------|-----|--------|----------|
| 음성만 | 낮음 | 낮음 | 중간 |
| 음성 + 스크린샷 | 중간 | 중간 | 높음 |
| 음성 + 웹캠 | 높음 | 높음 | 높음 |

### 권장 설정

#### 일반 사용
```python
# 스크린샷 모드 권장
python tests/test_multimodal_screenshot.py
```

#### 보안 감시
```python
# 웹캠 모드 + 긴 간격
monitor.start_monitoring(
    on_frame_callback=on_frame,
    frame_interval=5.0  # 5초마다
)
```

#### 실시간 모니터링
```python
# 웹캠 모드 + 짧은 간격
monitor.start_monitoring(
    on_frame_callback=on_frame,
    frame_interval=0.5  # 0.5초마다
)
```

---

## 다음 단계

1. **커스텀 시스템 프롬프트 작성**
   - `multimodal_analyzer.py`의 `system_prompt` 수정

2. **긴급 상황 커스터마이징**
   - `alert_manager.py`의 알림 방식 변경

3. **다른 프로젝트에 통합**
   - API 서버로 래핑
   - 웹훅 연동

---

## 참고 자료

- [OpenAI Vision API 문서](https://platform.openai.com/docs/guides/vision)
- [OpenCV 공식 문서](https://docs.opencv.org/)
- [SpeechRecognition 문서](https://pypi.org/project/SpeechRecognition/)

---

**문의사항이 있으시면 이슈를 등록해주세요!** 🚀
