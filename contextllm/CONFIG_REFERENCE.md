# Config Reference (Config-First)

## 원칙

- 실행 설정의 단일 소스: `config/config.yaml`
- CLI는 최소 옵션만 사용:
  - `--config`
  - `--mode`
  - `--show-config`
- 외부 시스템 결합은 `src/app/service.py` 사용

## 설정 우선순위

1. `--mode` (일시 override)
2. `config/config.yaml`
3. 코드 기본값

API 키는 별도 우선순위를 가집니다.

1. 환경변수 `OPENAI_API_KEY`
2. `config/.env`
3. `config/config.yaml`의 `api_keys.openai`

## 최소 실행 명령

```bash
python main.py
python main.py --mode testset
python main.py --show-config
python main.py --config ./config/config.yaml
```

## 주요 섹션

### `mode`, `model`

- `mode`: `realtime | testset | file | webcam | network`
- `model`: `gpt-4o-mini` 권장 (비용/속도 균형)

### `video`

- `camera_id`: 웹캠 모드 카메라 인덱스
- `testset_path`: 테스트셋 폴더
- `file_path`: 파일 모드 입력 경로
- `network_url`: 네트워크 카메라 URL

### `analysis`

- `iterations`: `null`이면 무한 루프
- `analyze_all_testset`: 테스트셋 전체 분석 여부
- `testset_index`: 단일 테스트셋 파일 인덱스
- `default_text`: 음성 대신 기본 텍스트
- `parallel`: 호환 옵션 (현재 순차 모니터링 맵핑)

### `speech`

- `enabled`: 음성 트리거 사용 여부
- `energy_threshold`: 낮을수록 민감
- `pause_threshold`: 문장 끝 무음 시간
- `dynamic_threshold`: 주변 소음 동적 보정

참고:
- `speech.enabled: false`면 음성 트리거 대신 `video-only` 분석 루프로 동작합니다.
- 이때 `analysis.iterations: null`이면 무한 루프 대신 1회로 자동 제한됩니다.

### `downsampling`

- `max_image_size`: 이미지 최대 픽셀
- `jpeg_quality`: JPEG 압축 품질
- `video_fps`: 분석 FPS
- `max_video_frames`: 최대 분석 프레임 수
- `video_capture_duration`: 캡처 구간 길이

### `display`

- `web_enabled`: 웹 대시보드 자동 시작
- `web_port`: 대시보드 포트
- `opencv_live`: 라이브 모드 OpenCV 표시

### `openai`

- `max_tokens`: 응답 토큰 상한
- `temperature`: 낮을수록 일관성
- `image_detail`: `low | high | auto`
- `timeout`: API timeout(초)

### `prompts.system`

- 출력 스키마를 엄격히 지정해야 downstream 파이프라인이 안정적
- 권장 필수 필드:
  - `priority`
  - `urgency`
  - `is_emergency`
  - `situation_type`
  - `action`
  - `emergency_reason`

## 모드별 설정 예시

### 실시간

```yaml
mode: realtime
speech:
  enabled: true
analysis:
  iterations: null
display:
  opencv_live: true
```

### 테스트셋 전체

```yaml
mode: testset
video:
  testset_path: testsets
analysis:
  analyze_all_testset: true
```

### 파일 입력

```yaml
mode: file
video:
  file_path: "/path/to/sample.mp4"
```

### 네트워크 카메라

```yaml
mode: network
video:
  network_url: "rtsp://user:pass@host:554/stream"
```

## 외부 결합 권장 패턴

`ContextLLMService`를 직접 호출하면 CLI 의존 없이 연동할 수 있습니다.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from app.service import ContextLLMService

service = ContextLLMService.from_config("config/config.yaml")
result = service.analyze_frame("도와주세요", frame)
```

## 운영 체크리스트

- [ ] `config/config.yaml`의 `mode`가 목적과 일치
- [ ] `video.*` 입력 경로/URL 유효
- [ ] `OPENAI_API_KEY`가 환경변수 또는 `.env`에 설정
- [ ] 프롬프트 출력 스키마(`priority/urgency/is_emergency`) 유지
- [ ] `analysis.iterations`가 운영 정책과 일치
