# 🎙️ Context LLM - 멀티모달 상황 분석 시스템

**음성 + 영상을 실시간으로 분석하여 긴급 상황을 감지하는 AI 시스템**

---

## 📋 개요

음성이 감지되면 자동으로:
1. **음성 → 텍스트** 변환
2. **음성 특성 분석** (피치, 에너지, 속도, 떨림)
3. **카메라 영상 캡처**
4. **GPT-4o-mini로 멀티모달 분석**

---

## 🚀 빠른 시작

```bash
# 1. 설치
pip install -r requirements.txt

# 2. API 키 설정
cp config/.env.example config/.env
# config/.env 파일에 OPENAI_API_KEY 입력

# 3. 실행
python main.py --help
```

---

## 📁 프로젝트 구조

```
contextllm/
├── main.py                                  # 🚀 메인 진입점 (CLI)
├── src/
│   ├── core/
│   │   ├── integrated_multimodal_system.py  # 🔥 핵심 시스템
│   │   ├── multimodal_analyzer.py           # GPT-4o 멀티모달 분석
│   │   └── voice_characteristics.py         # 음성 특성 분석
│   └── stt/
│       └── google_realtime_analyzer.py      # Google Realtime STT (예정)
│
├── tests/
│   └── test_integrated_multimodal.py        # 대화형 테스트 인터페이스
│
├── testsets/                                # 테스트용 이미지/비디오
├── config/                                  # 설정 파일 (.env)
├── data/logs/                               # 분석 결과 로그
└── requirements.txt
```

---

## 🎮 사용법 (CLI)

### 기본 명령어

```bash
# 도움말
python main.py --help
python main.py -h
```

### 모드별 실행

```bash
# 1. 실시간 모드 (기본) - 음성 감지 → 영상 캡처 → 분석
python main.py
python main.py --mode realtime
python main.py -m realtime

# 5회 반복 후 종료
python main.py -m realtime -n 5

# 2. 테스트셋 모드 - testsets/ 폴더의 파일들 분석
python main.py -m testset                    # 첫 번째 파일 분석
python main.py -m testset --all              # 전체 파일 분석
python main.py -m testset -i 2               # 3번째 파일 분석
python main.py -m testset --testset-path ./my_tests  # 다른 폴더

# 3. 파일 모드 - 특정 이미지/비디오 분석
python main.py -m file -f video.mp4
python main.py -m file -f image.jpg -t "이 상황을 분석해주세요"

# 4. 웹캠 모드 - 음성 없이 웹캠만 분석
python main.py -m webcam
python main.py -m webcam -c 1                # 카메라 ID 1번

# 5. 네트워크 카메라 모드
python main.py -m network -u rtsp://192.168.1.100:554/stream
python main.py -m network -u http://192.168.1.100:8080/video
```

### 고급 옵션

```bash
# 다운샘플링 설정
python main.py -m realtime \
    --image-size 480 \      # 최대 이미지 크기 (기본: 640)
    --quality 60 \          # JPEG 품질 (기본: 75)
    --fps 1.0 \             # 분석 FPS (기본: 2.0)
    --max-frames 5 \        # 최대 프레임 수 (기본: 10)
    --duration 3.0          # 캡처 시간 (기본: 5.0초)

# 모델 선택
python main.py -m webcam --model gpt-4o      # 기본: gpt-4o-mini

# 텍스트 입력 (음성 대신)
python main.py -m webcam -t "도와주세요!"
```

---

## 📊 분석 결과 예시

```json
{
  "transcribed_text": "도와주세요! 누가 저를 공격하고 있어요!",
  "voice_characteristics": {
    "emergency_indicators": {
      "high_pitch": true,
      "high_energy": true,
      "fast_speech": true,
      "overall_score": 0.75
    }
  },
  "multimodal_analysis": {
    "situation": "영상에서 두 사람 사이의 물리적 충돌이 감지됨",
    "situation_type": "보안",
    "urgency": "긴급",
    "priority": "CRITICAL",
    "is_emergency": true,
    "action": "즉시 보안 요원 파견 및 경찰 신고 권장"
  }
}
```

---

## ⚙️ 설정

모든 설정은 `config/config.yaml`에서 관리됩니다. CLI 인자가 config 값보다 우선합니다.

```bash
# 현재 설정 확인
python main.py --show-config
```

### config/config.yaml 주요 섹션

| 섹션 | 설명 |
|------|------|
| `mode` | 기본 실행 모드 |
| `model` | OpenAI 모델 (gpt-4o-mini, gpt-4o 등) |
| `video` | 비디오 소스 설정 (카메라 ID, 테스트셋 경로 등) |
| `downsampling` | 이미지/비디오 다운샘플링 설정 |
| `analysis` | 분석 관련 설정 |
| `prompts` | 시스템 프롬프트, 긴급 키워드 |
| `voice_analysis` | 음성 특성 분석 임계값 |
| `openai` | OpenAI API 설정 (토큰, 온도 등) |
| `logging` | 로그 저장 설정 |

### config/.env

```bash
OPENAI_API_KEY=sk-...
```

### 다운샘플링 기본값

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `max_image_size` | 640 | 최대 이미지 크기 (픽셀) |
| `jpeg_quality` | 75 | JPEG 압축 품질 (1-100) |
| `video_fps` | 2.0 | 비디오 분석 FPS |
| `max_video_frames` | 10 | 최대 프레임 수 |
| `video_capture_duration` | 5.0 | 캡처 시간 (초) |

### 음성 분석 임계값

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `pitch.high_threshold` | 250 | 높은 피치 판정 (Hz) |
| `energy.normalization_factor` | 0.5 | 에너지 정규화 계수 |
| `speech_rate.fast_threshold` | 6 | 빠른 속도 판정 (음절/초) |
| `scoring.llm_weight` | 0.6 | LLM 분석 가중치 |
| `scoring.voice_weight` | 0.4 | 음성 특성 가중치 |

---

## 🧪 테스트셋 사용법

`testsets/` 폴더에 이미지나 비디오를 넣고 테스트:

```bash
# 테스트 파일 추가
cp my_video.mp4 testsets/
cp my_image.jpg testsets/

# 전체 분석
python main.py -m testset --all

# 특정 파일 분석 (텍스트 입력과 함께)
python main.py -m testset -i 0 -t "살려주세요!"
```

### 지원 파일 형식

- **이미지**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`
- **비디오**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

---

## 🔧 프로그래밍 API

```python
from src.core.integrated_multimodal_system import (
    IntegratedMultimodalSystem,
    DownsamplingConfig
)

# 설정
config = DownsamplingConfig(
    max_image_size=640,
    jpeg_quality=75,
    video_fps=2.0,
    max_video_frames=10,
    video_capture_duration=5.0
)

# 시스템 초기화
system = IntegratedMultimodalSystem(
    camera_id=0,
    model="gpt-4o-mini",
    downsampling_config=config
)

# 실시간 모니터링
system.start_monitoring(max_iterations=5)

# 또는 단일 분석
result = system.analyze_once()

# 비디오 소스 변경
system.use_webcam(camera_id=0)
system.use_file("path/to/video.mp4")
system.use_network_camera("rtsp://...")
system.use_testset("testsets/")
```

---

## 📝 로드맵

- [x] 통합 멀티모달 시스템
- [x] 다운샘플링 지원
- [x] 다양한 비디오 소스 (웹캠, 파일, 네트워크, 테스트셋)
- [x] CLI 인터페이스 (main.py)
- [ ] Google Realtime STT 통합
- [ ] 웹 대시보드 UI
- [ ] 알림 시스템 (이메일, SMS)
- [ ] 다중 카메라 동시 모니터링

---

## 📄 라이선스

MIT License
