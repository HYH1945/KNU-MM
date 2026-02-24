# ContextLLM

음성/비음성 오디오 + 영상을 결합해 상황의 긴급도를 판단하는 멀티모달 분석 시스템입니다.  
`config/config.yaml` 중심으로 실행되며, CLI 옵션은 최소화되어 있습니다.

## 핵심 기능

- 실시간 음성(STT) + 영상 프레임 기반 멀티모달 분석
- YAMNet 기반 비음성 이벤트 감지(예: screaming, glass, alarm)
- `media_test` 기반 이미지/비디오/오디오 조합 테스트
- 비디오 입력 시 오디오 자동 추출 후 YAMNet/STT 결합 분석
- 외부 시스템 결합용 `ContextLLMService` 제공

## 프로젝트 구조

```text
contextllm/
├── main.py
├── config/
│   ├── config.yaml
│   └── config.yaml.example
├── src/
│   ├── app/
│   │   ├── runner.py
│   │   ├── service.py
│   │   └── settings.py
│   └── core/
│       ├── integrated_multimodal_system.py
│       ├── multimodal_analyzer.py
│       ├── sound_event_detector.py
│       └── voice_characteristics.py
├── testsets/
└── requirements.txt
```

## 빠른 시작

```bash
cd contextllm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API 키 설정(권장 순서):

1. 환경변수 `OPENAI_API_KEY`
2. `config/.env` 파일
3. `config/config.yaml`의 `api_keys.openai`

```bash
export OPENAI_API_KEY=sk-...
python main.py --show-config
python main.py --mode testset
```

## 실행 모드

- `realtime`: 실시간 마이크 트리거 기반 모니터링
- `testset`: `testsets/` 파일 기반 테스트
- `file`: `video.file_path` 단일 파일
- `webcam`: 로컬 웹캠
- `network`: RTSP/HTTP 네트워크 카메라

```bash
python main.py
python main.py --mode realtime
python main.py --mode testset
python main.py --mode file
python main.py --mode webcam
python main.py --mode network
python main.py --config ./config/config.yaml
python main.py --show-config
```

## media_test 규칙

`mode: testset` + `media_test.enabled: true`이면 아래 규칙으로 단발 분석합니다.

- 우선순위: `image_path` > `video_path` > live camera
- `audio_path`가 있으면 해당 오디오를 우선 사용
- `audio_path`가 없고 `video_path`가 있으면 비디오 오디오를 자동 추출하여 사용
- `audio_path`와 `video_path`가 모두 없으면 실시간 마이크 입력 사용
- 빈 문자열 `""`은 미입력

예시:

```yaml
mode: testset
media_test:
  enabled: true
  image_path: ""
  video_path: "testsets/sample.mp4"
  audio_path: ""
  text_input: ""
  phrase_time_limit: 6.0
```

## testset 동작

`testsets/`에는 이미지/비디오/오디오를 혼합해서 넣을 수 있습니다.

- 이미지: 프레임 1장 기반 분석
- 오디오 파일: YAMNet 분석 + 오디오 플레이스홀더 프레임 분석
- 비디오 파일: 대표 프레임 + 비디오 오디오 자동 추출(YAMNet/STT) 분석

지원 확장자:

- 이미지: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
- 비디오: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`
- 오디오: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`

## 비디오 오디오 자동 추출

비디오 입력 시 오디오는 다음 순서로 추출됩니다.

1. `ffmpeg` CLI 사용
2. 실패 시 `librosa + soundfile` 폴백
3. 둘 다 실패하면 영상 프레임만 분석

런타임 로그 예시:

- `🎵 비디오 오디오 추출: sample.mp4`
- `음성 인식됨: "..."` (STT 성공 시)
- `비음성(YAMNet): Screaming (0.39)` (YAMNet 결과)

## 외부 시스템 결합

`src/app/service.py`의 `ContextLLMService`를 사용하면 CLI 없이 결합할 수 있습니다.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from app.service import ContextLLMService

service = ContextLLMService.from_config(Path("config/config.yaml"))

result = service.analyze_frame(
    text="도와주세요",
    frame=frame_bgr_numpy,
    additional_context="Mic DOA=120deg, PTZ settled"
)

if result["success"]:
    print(result["analysis"]["priority"], result["analysis"]["is_emergency"])
```

## 주요 설정 포인트

- `analysis.voice_characteristics`: 음성 특성 분석 on/off
- `speech.pause_threshold`: 문장 끝 침묵 기준
- `sound_event.trigger_threshold`: 비음성 트리거 민감도
- `media_test.*`: 멀티 입력 테스트 시나리오
- `prompts.system`: 멀티모달 판단 정책

## 저위험 변경 원칙

운영 중에는 아래 범위를 우선 적용하는 것을 권장합니다.

- 테스트/문서/로그 보강
- 설정 파싱 검증 강화(기능 로직 변경 없이)
- 미사용 코드 정리(동작 경로 불변)

변경 후 최소 검증:

```bash
python -m pytest -q tests/test_settings.py
python main.py --mode testset
```

## 트러블슈팅

`Multimodal analyzer initialization failed: OpenAI API 키가 설정되지 않았습니다`
- `OPENAI_API_KEY` 또는 `config/.env` 또는 `config/config.yaml` 확인

`SoundEventDetector is disabled ... reason=tensorflow/tensorflow_hub import unavailable`
- 현재 실행 중인 파이썬 환경에 `tensorflow`/`tensorflow-hub` 설치 확인
- macOS에서는 파이썬 버전/휠 호환성 확인 후 재설치

비디오 오디오 추출 실패
- `ffmpeg -version` 확인
- ffmpeg가 없으면 `librosa + soundfile` 경로를 사용하므로 관련 패키지 설치 확인

## 보안

- API 키를 Git에 커밋하지 마세요.
- `config/.env`, 로컬 `config/config.yaml`은 반드시 `.gitignore`로 관리하세요.

## 라이선스

MIT License
