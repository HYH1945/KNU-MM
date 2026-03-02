# 멀티모달 관제 시스템

> 시각(YOLO + PTZ) + 청각(MicArray + STT + YAMNet) 정보를 융합하여,  
> LLM이 실시간으로 상황을 판단하고 카메라를 제어하는 **멀티모달 관제 시스템**

---

## 프로젝트 개요

기존 CCTV 관제는 상황의 판단은 사람에게 맡겨져 있습니다.

본 프로젝트는 **CCTV 영상(시각)** 에 **소리와 함께 360도 방향을 감지하는 mic array(청각)** 를 결합하고,
수집된 두 데이터를 **LLM** 이 분석하여
긴급 상황을 자동으로 판단해 카메라를 제어하는 멀티모달 관제 시스템을 구축해보는 것이 목표입니다.

### 핵심 기능

- **YOLO의 객체 추적** — 실시간 사람/객체 감지 및 우선순위 기반 추적
- **Mic Array 입력 및 처리** — 360° 음성 방향 감지 및 마이크 입력 
- **STT + YAMNet** — 음성→텍스트 변환 / 비음성 데이터(비명, 폭발음 등) 분류
- **ContextLLM** — 시청각 데이터를 입력으로 받아 상황 판단 (긴급도 판단)
- **PTZ camera 제어** — 현재 상황을 판단후, 우선순위를 고려한 카메라 제어

---

## 시스템 아키텍처

<img width="541" height="536" alt="image" src="https://github.com/user-attachments/assets/9511d242-2339-4a60-85f3-501251459b3a" />


---

## 디렉토리 구조

```
KNU-MM/
├── integrated_system/            # ★ 메인 통합 시스템
│   ├── main.py                   # 진입점 — 모든 모듈 통합 실행
│   ├── config.yaml               # 통합 설정 파일
│   ├── requirements.txt          # 의존성
│   ├── core/                     # 핵심 인프라
│   │   ├── event_bus.py          #   이벤트 Pub/Sub 시스템
│   │   ├── orchestrator.py       #   모듈 등록 + 파이프라인 관리
│   │   ├── base_module.py        #   모듈 공통 인터페이스
│   │   ├── module_loader.py      #   원본 모듈 경로 관리 + import 유틸
│   │   └── spatial_context.py    #   DOA↔YOLO 공간 융합
│   ├── modules/                  # 기능 모듈
│   │   ├── yolo_detection.py     #   YOLO + BoT-SORT 객체 탐지/추적
│   │   ├── mic_array.py          #   ReSpeaker DOA 방향 감지
│   │   ├── stt_module.py         #   음성→텍스트 (Google Speech API)
│   │   ├── context_llm.py        #   LLM 상황 분석 + YAMNet 연동
│   │   ├── ptz_controller.py     #   PTZ 카메라 제어 (ONVIF/HTTP)
│   │   ├── stream_manager.py     #   공유 영상 스트림
│   │   ├── dashboard_server.py   #   이벤트 대시보드 서버 (FastAPI)
│   │   ├── index.html            #   이벤트 대시보드 프론트엔드
│   │   ├── server_reporter.py    #   외부 서버 이벤트 전송
│   │   └── tuning.py             #   ReSpeaker USB Tuning 인터페이스
│   └── tests/                    # 유닛 테스트
│       └── test_spatial_context.py
│
├── contextllm/                   # ContextLLM 독립 모듈
│   ├── src/core/                 #   LLM 분석 엔진 + YAMNet 감지기
│   └── src/web/                  #   ContextLLM 대시보드 (Flask-SocketIO)
│
├── Detaction_CCTV/               # YOLO 원본 서비스 (VisionProcessor 등)
│   └── services/                 #   vision_processor, reid_manager, ...
│
├── _archive/                     # 통합 전 프로토타입 보관
│   ├── mic_array_Control/        #   마이크 어레이 초기 모듈
│   ├── PTZcamera_Control/        #   PTZ 카메라 초기 테스트
│   ├── mic_context_fusion/       #   MicArray + LLM 융합 러너
│   ├── Detection_CCTV/           #   BoT-SORT + 히트맵 모듈
│   └── integrated_system_process/  # integrated_system 포크
│
├── .env                          # 환경변수 (API키, 카메라 크레덴셜)
├── .env.example                  # 환경변수 템플릿
└── yolov8n.pt                    # YOLO 모델 가중치
```

---

## 빠른 시작

### 1. 환경 설정

```bash
conda create -n knu-mm python=3.10 -y && conda activate knu-mm
pip install -r integrated_system/requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에서 아래 항목 설정:
#   OPENAI_API_KEY=sk-...
#   CAMERA_RTSP_URL=rtsp://...
#   CAMERA_IP=192.168.x.x
#   CAMERA_USER=admin
#   CAMERA_PASSWORD=...
```

### 3. 실행

```bash
python integrated_system/main.py
```

#### 실행 옵션

```bash
python integrated_system/main.py --no-mic       # 마이크 어레이 없이
python integrated_system/main.py --no-stt       # 음성 인식 없이
python integrated_system/main.py --no-llm       # LLM 분석 없이
python integrated_system/main.py --no-yolo      # YOLO 없이
python integrated_system/main.py --no-display   # 화면 표시 없이
python integrated_system/main.py --debug        # 디버그 로깅
python integrated_system/main.py --config custom.yaml  # 커스텀 설정
```

#### 실행 중 키보드 조작

| 키 | 동작 |
|----|------|
| `Q` | 종료 |
| `P` | 파이프라인 전환 (security ↔ full_analysis) |

---

## 주요 모듈 상세

### 시각 정보 처리

| 항목 | 내용 |
|------|------|
| **모델** | YOLO26n |
| **추적** | BoT-SORT 알고리즘 |
| **우선순위** | 객체별 가중치 계산으로 추적 |
| **PTZ 연동** | PID 제어로 타겟 중심 추적 / 미탐지 시 순찰 모드 복귀 |

### 청각 정보 처리

| 항목 | 내용 |
|------|------|
| **DOA 감지** | ReSpeaker v2 하드웨어 VAD + 360° 방향 감지 |
| **STT** | Google Speech API |
| **비음성 분류** | YAMNet 모델 사용 |
| **청각→시각 연동** | DOA 방향으로 PTZ 카메라 자동 회전 → 발화자 추정 |

### LLM 상황 분석
<img width="588" height="331" alt="image" src="https://github.com/user-attachments/assets/2ef0e296-c317-4910-815c-a0fecfe27e35" />

| 항목 | 내용 |
|------|------|
| **모델** | GPT-4o-mini |
| **입력** | 영상 프레임 + 음성 텍스트 + YOLO 컨텍스트 + 비음성 분류 결과 |
| **출력** | 긴급도(CRITICAL/HIGH/MEDIUM/LOW) + 상황 유형 + 대응 제안 |
| **쿨다운** | 5초 간격  |
| **대시보드 연동** | 분석 결과를 ContextLLM 대시보드로 자동 푸시 |

### PTZ 카메라 제어

| 우선순위 | 모드 | 트리거 |
|---------|------|--------|
| 0 (최저) | 순찰 (Patrol) | 이벤트 없을 때 |
| 1 | 음원 추적 (MIC_DOA) | 마이크 방향 감지 시 |
| 2 | 객체 추적 (YOLO) | 사람/객체 감지 시 |
| 3 (최고) | LLM 긴급 (Emergency) | 긴급 상황 판단 시 |

---

## 이벤트 흐름

<img width="1632" height="421" alt="image" src="https://github.com/user-attachments/assets/48f9926a-df9b-4f60-a07c-4f2b51fbf805" />


---

## 설정 파일

모든 설정은 `integrated_system/config.yaml`에서 관리함.
민감 정보(API 키, 카메라 비밀번호)는 루트 `.env`에서 로드

| 섹션 | 설명 |
|------|------|
| `camera` | RTSP URL, 테스트 영상 경로, IP, 포트, 인증 정보 |
| `ptz` | 제어 모드(ONVIF/HTTP/both), PID 계수, 데드존, 순찰 속도 |
| `yolo` | 모델 경로, 확신도, 타겟 클래스, 순찰 복귀 딜레이 |
| `mic_array` | AGC 게인, VAD 임계값, DOA 신뢰도, 히스토리 크기 |
| `stt` | 언어, 에너지 임계값, 발화 시간 제한, 동적 임계값 |
| `context_llm` | 모델명, 분석 쿨다운, config 경로 |
| `server` | 대시보드 포트, 자동 시작 여부, 서버 URL, 타임아웃 |
| `fusion` | 카메라 FOV, 공간 매칭 임계 각도, 이벤트 히스토리 설정 |
| `pipeline` | 기본 파이프라인 종류, 프레임 스킵 설정 |
| `display` | OpenCV 윈도우 표시 여부, FPS/모듈 상태 표시 |
| `logging` | 로그 레벨, 로그 파일 경로 |

---

## 하드웨어

| 장비 | 모델 | 용도 |
|------|------|------|
| PTZ 카메라 | Hikvision (ONVIF 지원) | 영상 입력 + 방향 제어 |
| 마이크 어레이 | ReSpeaker Mic Array v2.0 | 360° 음원 방향 감지 |
---

## 기술 스택

| 분야 | 기술 |
|------|------|
| 객체 탐지 | YOLOv8 |
| PTZ 제어 | ONVIF + Hikvision HTTP |
| Mic array | ReSpeaker v2 (pyusb) |
| 음성 인식 | Google Speech API (SpeechRecognition) |
| 비음성 감지 | YAMNet (TensorFlow Hub) |
| LLM 분석 | OpenAI GPT-4o-mini |
| 영상 처리 | OpenCV |
| LLM 대시보드 | Flask + SocketIO |
| 프레임워크 | EventBus + Orchestrator + BaseModule |

---

## 팀
심인영 : PTZ 제어 함수 및 실시간 객체탐지 모듈 개발

장준용 : 음성 처리 및 멀티 모달 시스템 구현 

장호진 : YOLO 기반 실시간 영상 관제 성능 최적화 및 히트맵 기능 구현

황영하 : 팀장, mic array & PTZ 제어 및 프로토타입 제작

