# KNU-MM 통합 멀티모달 관제 시스템

> **시각(YOLO) + 청각(MicArray + STT) → LLM 통합 분석**을 하나의 파이프라인으로 통합한 스마트 관제 시스템

## 시스템 데이터 흐름

```mermaid
flowchart TD
    subgraph 영상["🎥 영상 입력"]
        CCTV["CCTV / 웹캠"]
        Stream["SharedStreamManager"]
        CCTV --> Stream
    end

    subgraph 음성["🎤 음성 입력"]
        MIC["MicArray\n(ReSpeaker DOA)"]
        STT["STTModule\n(speech_recognition)"]
    end

    subgraph 분석["🧠 분석"]
        YOLO["YOLODetectionModule\n(사람/객체 탐지)"]
        LLM["ContextLLMModule\n(GPT-4o-mini)"]
    end

    subgraph 출력["🖥️ 출력"]
        OPENCV["통합 OpenCV 화면\n(YOLO박스 + DOA컴퍼스\n+ STT텍스트 + LLM분석)"]
        SERVER["ServerReporter\n(대시보드 전송)"]
    end

    Stream -->|frame| YOLO
    YOLO -->|yolo.person_detected| LLM

    MIC -->|mic.speech_detected\nmic.doa_detected| EventBus
    STT -->|stt.text_recognized| LLM

    STT -.->|doa_angle 수신| MIC

    LLM -->|llm.analysis_complete\nllm.emergency| EventBus

    Stream -->|frame| OPENCV
    YOLO -->|boxes| OPENCV
    MIC -->|doa_angle| OPENCV
    STT -->|text| OPENCV
    LLM -->|priority\nsituation| OPENCV
    LLM --> SERVER

    style MIC fill:#FFE082
    style STT fill:#81D4FA
    style LLM fill:#CE93D8
    style OPENCV fill:#A5D6A7
```

## 프로젝트 구조

```
KNU-MM/
├── integrated_system/          ★ 통합 진입점 (이 README)
│   ├── main.py                 # 메인 실행 파일
│   ├── config.yaml             # 통합 설정
│   ├── requirements.txt        # 의존성
│   ├── core/                   # 프레임워크 코어
│   │   ├── event_bus.py        #   이벤트 버스 (Pub/Sub)
│   │   ├── base_module.py      #   모듈 공통 인터페이스
│   │   ├── orchestrator.py     #   파이프라인 오케스트레이터
│   │   └── module_loader.py    #   원본 모듈 경로 관리 + import 유틸
│   └── modules/                # 래퍼 모듈 (원본 폴더에서 직접 import)
│       ├── stream_manager.py   #   공유 영상 스트림
│       ├── yolo_detection.py   #   YOLO 객체 탐지 + Re-ID + 추적
│       ├── ptz_controller.py   #   PTZ 카메라 우선순위 제어
│       ├── mic_array.py        #   ReSpeaker DOA 음원 방향 감지
│       ├── stt_module.py       #   음성→텍스트 변환 (Google STT)
│       ├── context_llm.py      #   GPT 멀티모달 통합 분석
│       └── server_reporter.py  #   대시보드 서버 HTTP 전송
│
├── Detaction_CCTV/             # 원본: YOLO + PTZ + 스트림
│   └── services/               #   VisionProcessor, ReIDManager, ...
├── mic_array_Control/          # 원본: ReSpeaker 마이크 어레이
│   ├── tuning.py               #   Tuning 클래스 (USB 디바이스 제어)
│   └── test.py                 #   DOA + STT + PTZ 원본 로직
├── contextllm/                 # 원본: 멀티모달 LLM 시스템
│   └── src/core/               #   IntegratedMultimodalSystem, MultimodalAnalyzer
├── PTZcamera_Control/          # 원본: ONVIF PTZ 테스트
└── 서버전송예시.py               # 원본: 서버 전송 패턴 참조
```

### 원본 ↔ 통합 모듈 매핑

| 원본 모듈 | 통합 래퍼 | import 방식 |
|-----------|----------|-------------|
| `Detaction_CCTV/services/vision_processor.py` | `modules/yolo_detection.py` | 직접 import |
| `Detaction_CCTV/services/priority_manager.py` | `modules/yolo_detection.py` | 직접 import |
| `Detaction_CCTV/services/reid_manager.py` | `modules/yolo_detection.py` | 직접 import |
| `Detaction_CCTV/services/stream_handler.py` | `modules/stream_manager.py` | 직접 import |
| `Detaction_CCTV/services/ptz_controller.py` | `modules/ptz_controller.py` | 직접 import |
| `mic_array_Control/tuning.py` | `modules/mic_array.py` | 직접 import |
| `mic_array_Control/test.py` | `modules/mic_array.py` | 알고리즘 참조 (부작용으로 import 불가) |
| `contextllm/src/core/*` | `modules/context_llm.py` | importlib 파일경로 로딩 |
| `서버전송예시.py` | `modules/server_reporter.py` | 패턴 참조 |

> **원본 폴더의 파일을 수정하면 통합 시스템에 즉시 반영됩니다** (코드를 복사하지 않고 직접 import)

---

## 빠른 시작

### 1. 환경 설정

```bash
# conda 환경 생성 (권장)
conda create -n knu-mm python=3.10 -y
conda activate knu-mm

# 의존성 설치
cd integrated_system
pip install -r requirements.txt

# macOS: libusb 설치 (ReSpeaker용)
brew install libusb
```

### 2. API 키 설정

프로젝트 루트 또는 `contextllm/config/`에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-...

# 카메라 설정 (선택, config.yaml로도 설정 가능)
RTSP_URL=rtsp://admin:password@ip/Streaming/Channels/101
CAMERA_IP=ip
CAMERA_USER=admin
CAMERA_PASSWORD=password
```

### 3. 실행

```bash
cd integrated_system

# 전체 모듈 실행
python main.py

# 마이크 어레이 없이 (웹캠 + STT + LLM)
python main.py --no-mic

# STT 없이 (영상 분석만)
python main.py --no-stt

# LLM 없이 (YOLO + 마이크만, API 비용 절약)
python main.py --no-llm

# 마이크/STT 모두 없이 (영상만 분석)
python main.py --no-mic --no-stt

# 화면 없이 (서버 전송만)
python main.py --no-display

# 디버그 모드
python main.py --debug

# 커스텀 설정
python main.py --config my_config.yaml
```

### 실행 중 키보드 단축키

| 키 | 기능 |
|----|------|
| `Q` | 시스템 종료 |
| `P` | 파이프라인 전환 (security ↔ full_analysis) |

---

## 핵심 아키텍처

### EventBus (이벤트 기반 통신)

모듈 간 느슨한 결합. 서로 직접 참조하지 않고 이벤트로 통신합니다.

```python
# 주요 이벤트
"mic.speech_detected"     # MicArray → 음성 감지 (DOA 각도 포함)
"mic.doa_detected"        # MicArray → 음원 방향 확정
"stt.text_recognized"     # STT → 음성→텍스트 변환 완료
"stt.non_speech_audio"    # STT → 인식 실패 오디오(비음성 후보)
"yolo.person_detected"    # YOLO → 사람 감지
"yolo.objects_detected"   # YOLO → 객체 목록
"llm.analysis_complete"   # LLM → 분석 완료
"llm.emergency"           # LLM → 긴급 상황 (priority=2)
```

### BaseModule (모듈 인터페이스)

모든 모듈이 동일한 인터페이스를 구현하여 Hot-plug 가능:

```python
class MyModule(BaseModule):
    @property
    def name(self) -> str: return "my_module"
    def initialize(self) -> bool: ...
    def process(self, shared_data) -> dict: ...
    def shutdown(self) -> None: ...
```

### PTZ 우선순위 중재

여러 모듈이 PTZ를 동시에 제어하려 할 때, 우선순위로 결정:

```
EMERGENCY(3) > YOLO_TRACKING(2) > MIC_DOA(1) > PATROL(0)
```

### 분석 트리거 조건

ContextLLM은 다음 트리거 중 하나라도 충족하면 분석을 실행합니다:

| 트리거 | 소스 | 분석 방식 |
|--------|------|-----------|
| 음성 인식 | STT (`stt.text_recognized`) | 음성+영상 통합 분석 (STT 텍스트 + 프레임) |
| 비음성 위험음 | STT + YAMNet (`stt.non_speech_audio`) | 비음성 이벤트+영상 통합 분석 |
| 사람 + 음성/비음성 | YOLO + STT | 영상 맥락 강화 통합 분석 |

멀티모달로 전달되는 YOLO 정보:
- 기존: `person_detected` 여부만 전달
- 현재: `person_detected`, 객체 수, 주요 객체 라벨/신뢰도, 추적 모드/타겟 요약까지 컨텍스트로 전달

---

## OpenCV 통합 화면

```
┌──────────────────────────────────────────────┐
│  FPS: 30.0 │ Mode: TRACKING │ Pipeline: ...  │ 상단 바
│  [YOLO ●] [MIC ●] [STT ●] [LLM ●]          │ 모듈 상태
├──────────────────────────────────────────────┤
│                                              │
│          영상 프레임 (YOLO 바운딩 박스)        │
│                                              │
│  ┌─DOA──┐                                    │
│  │  ➤   │  (미니 컴퍼스 - 음원 방향)          │
│  │ 120° │                                    │
│  └──────┘                                    │
├──────────────────────────────────────────────┤
│  [MIC] "도와주세요! 여기 사람이 쓰러졌어요"   │ 하단 패널
│  [LLM] [EMERGENCY] 긴급상황 - 쓰러진 사람     │ (STT + LLM)
└──────────────────────────────────────────────┘
```

- **긴급 상황 시**: 화면 테두리 빨간색 깜빡임
- **STT 텍스트**: 인식 후 10초간 표시
- **DOA 컴퍼스**: 좌하단에 음원 방향 화살표

---

## 설정 파일 (config.yaml)

주요 설정 항목:

```yaml
camera:
  rtsp_url: 0                  # 0=웹캠, RTSP URL, 파일 경로

yolo:
  enabled: true
  model_path: "yolov8n.pt"
  confidence: 0.3

mic_array:
  enabled: true
  confidence_threshold: 0.6    # DOA 방향 신뢰도

stt:
  enabled: true
  language: "ko-KR"            # 인식 언어
  energy_threshold: 400        # 음성 감지 민감도
  pause_threshold: 3.0         # 문장 끝 판단 (초)
  phrase_time_limit: 15.0      # 최대 발화 시간 (초)

context_llm:
  enabled: true
  model: "gpt-4o-mini"
  analysis_cooldown: 5.0       # API 호출 간격 (초)

pipeline:
  default: "security"
  process_every_n_frames: 3    # 성능 최적화
```

전체 설정은 [config.yaml](config.yaml) 참조.

---

## 새 모듈 추가 방법

```python
# modules/fire_detection.py
from integrated_system.core.base_module import BaseModule

class FireDetectionModule(BaseModule):
    @property
    def name(self) -> str:
        return "fire_detection"

    def initialize(self) -> bool:
        # 모델 로드 등
        return True

    def process(self, shared_data):
        frame = shared_data["frame"]
        # 화재 감지 로직 ...
        if fire_detected:
            self.emit("fire.detected", {"confidence": 0.95}, priority=2)
        return {"fire_detected": True, "priority": "CRITICAL"}

    def shutdown(self):
        pass
```

`main.py`에 등록:

```python
orch.register(FireDetectionModule(event_bus))
```

---

## 기술 스택

| 분야 | 기술 |
|------|------|
| 객체 탐지 | YOLOv8 (ultralytics) |
| PTZ 제어 | ONVIF + Hikvision HTTP |
| 마이크 어레이 | ReSpeaker v2 (pyusb) |
| 음성 인식 | Google Speech API (SpeechRecognition) |
| LLM 분석 | OpenAI GPT-4o-mini |
| 영상 처리 | OpenCV |
| 프레임워크 | EventBus + Orchestrator + BaseModule |

## 팀

경북대학교 멀티모달 관제 시스템 프로젝트 (KNU-MM)
