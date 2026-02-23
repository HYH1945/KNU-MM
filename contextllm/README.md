# 🎙️ Context LLM - 멀티모달 상황 분석 시스템

**음성 + 영상을 실시간으로 분석하여 긴급 상황을 감지하는 AI 시스템**

---

## 📋 개요

음성이 감지되면 자동으로:
1. **음성 → 텍스트** 변환
2. **음성 특성 분석** (피치, 에너지, 속도, 떨림)
3. **카메라 영상 캡처**
4. **GPT-4o-mini로 멀티모달 분석**

추가로, **YAMNet 기반 비음성 사운드 이벤트 감지**를 통해
비명, 유리 파손, 경보음 등도 실시간 트리거로 사용할 수 있습니다.

---

## 🚀 빠른 시작

```bash
# 1. 설치
pip install -r requirements.txt

# 2. API 키 설정
cp .env.example config/.env
# config/.env 파일에 OPENAI_API_KEY 입력

# 3. 실행
python main.py --help
```

---

## 📁 프로젝트 구조

```
contextllm/
├── main.py                                  # 🚀 메인 진입점 (얇은 CLI)
├── src/
│   ├── app/
│   │   ├── settings.py                      # 설정 스키마/로더 (config-first)
│   │   ├── runner.py                        # 실행 오케스트레이션
│   │   └── service.py                       # 외부 시스템 결합용 파사드 API
│   ├── core/
│   │   ├── integrated_multimodal_system.py  # 🔥 핵심 시스템
│   │   ├── multimodal_analyzer.py           # GPT-4o 멀티모달 분석
│   │   └── voice_characteristics.py         # 음성 특성 분석
│
├── tests/
│   └── test_integrated_multimodal.py        # 대화형 테스트 인터페이스
│
├── testsets/                                # 테스트용 이미지/비디오
├── config/                                  # 설정 파일 (.env)
├── data/logs/                               # 분석 결과 로그
└── requirements.txt
```

아키텍처 상세는 `ARCHITECTURE.md`를 참고하세요.

---

## 🔌 외부 시스템 결합

CLI 없이 다른 시스템에서 직접 호출하려면 `ContextLLMService`를 사용하세요.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from app.service import ContextLLMService

service = ContextLLMService.from_config(Path("config/config.yaml"))
result = service.analyze_frame(
    text="도와주세요, 사람이 쓰러졌어요",
    frame=frame_bgr_numpy,
    additional_context="Mic DOA=120deg, PTZ settled"
)

if result["success"] and result["is_emergency"]:
    print(result["priority"], result["analysis"]["action"])
```

---

## 🎮 사용법 (Config-First)

실행 옵션은 `config/config.yaml`에서 관리하고, CLI는 최소 옵션만 사용합니다.

```bash
# 도움말
python main.py --help

# 설정값대로 실행 (mode 포함)
python main.py

# 모드만 일시적으로 override
python main.py --mode realtime
python main.py --mode testset
python main.py --mode file
python main.py --mode webcam
python main.py --mode network

# 설정 파일 지정
python main.py --config ./config/config.yaml

# 현재 설정 확인
python main.py --show-config
```

모드별 입력값은 CLI가 아니라 `config/config.yaml`에서 수정합니다.

- `mode: testset` + `analysis.analyze_all_testset: true`
- `mode: file` + `video.file_path: "path/to/file.mp4"`
- `mode: network` + `video.network_url: "rtsp://..."`
- `mode: webcam` + `video.camera_id: 0`

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

모든 설정은 `config/config.yaml`에서 관리됩니다.  
CLI는 `--config`, `--mode`, `--show-config`만 사용합니다.

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
| `sound_event` | YAMNet 비음성 이벤트 감지 설정 |
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

# config/config.yaml 설정 예:
# mode: testset
# analysis.analyze_all_testset: true
python main.py
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

## 보안

### API 키 관리 (필수)

**절대 API 키를 코드나 리포지토리에 커밋하지 마세요!**

```bash
# ❌ 절대 하지 말 것
OPENAI_API_KEY=sk-abc123...  # 코드에 하드코딩
config/config.yaml에 API 키 저장

# ✅ 권장 방법 (아래 중 하나)

# 방법 1: 환경 변수 (권장)
export OPENAI_API_KEY=sk-your-key-here
python main.py --mode realtime

# 방법 2: .env 파일 (로컬 개발)
cp .env.example .env
# .env에 OPENAI_API_KEY=sk-... 입력
# .env는 .gitignore에 등록되어 자동 무시됨

# 방법 3: 설정 파일
cp config/config.yaml.example config/config.yaml
# config/config.yaml의 api_keys.openai 입력
# config.yaml은 .gitignore에 등록되어 자동 무시됨
```

### 보안 체크리스트

- ✅ 환경 변수 또는 `.env` 파일로 API 키 관리
- ✅ 로컬 설정 파일 (`config.yaml`, `.env`) Git에서 제외
- ✅ 웹 대시보드는 `localhost:5000`에서만 실행 (외부 공개 안 함)
- ✅ 녹음/로그 파일은 `data/logs/`, `recordings/`에 저장 (Git 제외)
- ✅ Flask SECRET_KEY는 환경 변수에서 로드

### 배포 시 보안 조치

1. **환경 변수 설정**
   ```bash
   export OPENAI_API_KEY=your-production-key
   export FLASK_SECRET_KEY=your-secret-key
   ```

2. **웹 대시보드 접근 제한**
   ```python
   # 필요시 인증 추가 (예: nginx 기본 인증)
   ```

3. **로그 및 녹음 파일 보안**
   - 개인 정보 포함 가능하므로 접근 제한
   - 정기적인 삭제 정책 수립

---

## 라이선스

MIT License
