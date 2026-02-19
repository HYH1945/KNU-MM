# 🔧 Config 설정 참고서

## 📋 Config 구조 및 우선순위

```
CLI 인자 > config.yaml > 코드 기본값
```

각 섹션별로 config.yaml의 설정값이 코드에서 어떻게 사용되는지 정리했습니다.

---

## 1️⃣ 기본 설정 (`mode`, `model`)

| 설정 | 값 | 코드 사용 | 설명 |
|------|-----|---------|------|
| `mode` | realtime / testset / file / webcam / network | main.py | 실행 모드 |
| `model` | gpt-4o-mini / gpt-4o | IntegratedMultimodalSystem.__init__ | OpenAI 모델 |

**CLI 덮어쓰기:**
```bash
python main.py -m realtime --model gpt-4o
```

---

## 2️⃣ 음성 인식 설정 (`speech`)

| 설정 | 기본값 | 코드 위치 | 설명 |
|------|--------|---------|------|
| `energy_threshold` | 400 | SpeechDetector.__init__ | 음성 감지 민감도 (낮을수록 민감) |
| `pause_threshold` | 3.0 | SpeechDetector.__init__ | 문장 끝 판단 침묵 시간 (초) |
| `dynamic_threshold` | false | SpeechDetector.__init__ | 에너지 임계값 동적 조정 |

**CLI 덮어쓰기:**
```bash
# 스피커 소리 감지용 (고정 임계값 + 낮은 민감도)
python main.py -m realtime --static-threshold --energy-threshold 200

# 더 민감하게 (높은 민감도)
python main.py -m realtime --energy-threshold 100
```

**동작:**
- `energy_threshold = 400`: 중간 민감도
- `energy_threshold = 100`: 매우 민감 (배경음도 감지)
- `energy_threshold = 600`: 낮은 민감도 (큰 소리만)
- `pause_threshold = 3.0`: 3초 침묵 후 인식 끝
- `pause_threshold = 10.0`: 10초 침묵 후 인식 끝

---

## 3️⃣ 음성 특성 분석 설정 (`voice_analysis`)

**⚠️ 주의:** 현재 코드에서 음성 특성 분석은 **LLM 기반으로 전환**되었습니다.
- `scoring.llm_weight = 1.0` (LLM이 100% 결정)
- `scoring.voice_weight = 0.0` (음성 특성은 미사용)

### 피치 분석 (`pitch`)

| 설정 | 기본값 | 코드 사용 | 설명 |
|------|--------|---------|------|
| `high_threshold` | 250 | extract_voice_indicators() | 높은 피치 판정 임계값 (Hz) |
| `variability_threshold` | 50 | extract_voice_indicators() | 피치 변동성 (떨림) 임계값 |

**코드:**
```python
# 피치 분석 (높은 피치 = 긴장/공포)
if pitch.get("mean", 0) > high_pitch_threshold:  # 250Hz
    indicators["high_pitch"] = True
if pitch.get("std", 0) > pitch_variability_threshold:  # 50
    indicators["voice_trembling"] = True
```

### 에너지 분석 (`energy`)

| 설정 | 기본값 | 코드 사용 | 설명 |
|------|--------|---------|------|
| `normalization_factor` | 0.5 | extract_voice_indicators() | 에너지 정규화 계수 |

**코드:**
```python
high_energy_threshold = 0.5 * 0.6  # = 0.3
if energy.get("max", 0) > 0.3:  # 최대 에너지 > 0.3
    indicators["high_energy"] = True
```

### 말 속도 분석 (`speech_rate`)

| 설정 | 기본값 | 코드 사용 | 설명 |
|------|--------|---------|------|
| `fast_threshold` | 5 | extract_voice_indicators() | 빠른 말 판정 임계값 (음절/초) |

**코드:**
```python
if speech_rate.get("estimated_syllables_per_second", 0) > 5:
    indicators["fast_speech"] = True
```

---

## 4️⃣ 분석 설정 (`analysis`)

| 설정 | 기본값 | 코드 사용 | 설명 |
|------|--------|---------|------|
| `voice_characteristics` | true | IntegratedMultimodalSystem.__init__ | 음성 특성 분석 활성화 |
| `streaming` | false | MultimodalAnalyzer.__init__ | OpenAI 응답 스트리밍 |
| `parallel` | false | main.py | 호환 옵션 (현재는 순차 모니터링과 동일 동작) |

**CLI 덮어쓰기:**
```bash
# 병렬 호환 옵션 (현재는 순차 모니터링과 동일 동작)
python main.py -m realtime --parallel
```

---

## 5️⃣ 다운샘플링 설정 (`downsampling`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `max_image_size` | 320 | 이미지 최대 크기 (픽셀) - 작을수록 빠름 |
| `jpeg_quality` | 70 | JPEG 압축 품질 (1-100) - 낮을수록 빠름 |
| `video_fps` | 2.0 | 비디오 분석 FPS |
| `max_video_frames` | 10 | 최대 프레임 수 |
| `video_capture_duration` | 5.0 | 캡처 시간 (초) |

**용도:**
- 성능 개선: `max_image_size` 줄이기
- 정확도 개선: `max_image_size` 키우기
- API 비용 절감: `jpeg_quality` 낮추기

---

## 6️⃣ OpenAI API 설정 (`openai`)

| 설정 | 기본값 | 코드 사용 | 설명 |
|------|--------|---------|------|
| `max_tokens` | 800 | MultimodalAnalyzer.__init__ | 최대 응답 토큰 수 |
| `temperature` | 0.3 | MultimodalAnalyzer.__init__ | 응답 다양성 (0-1, 낮을수록 일관성) |
| `image_detail` | low | MultimodalAnalyzer.__init__ | 이미지 분석 상세도 |
| `timeout` | 30 | (설정만, 코드에서 미사용) | API 타임아웃 (초) |

**설정값 의미:**
- `temperature = 0.3`: 매우 일관성 있는 응답 (긴급 감지에 적합)
- `temperature = 0.7`: 중간 정도 창의성
- `temperature = 1.0`: 매우 창의적인 응답
- `image_detail = low`: 빠르고 저비용 (기본)
- `image_detail = high`: 더 정교한 분석, 고비용

---

## 7️⃣ 로깅 설정 (`logging`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `save_results` | true | 분석 결과 JSON 저장 |
| `log_dir` | data/logs | 로그 저장 경로 |
| `verbose` | false | 상세 로그 출력 |

**CLI 덮어쓰기:**
```bash
python main.py -m realtime -v  # verbose 활성화
```

---

## 8️⃣ 디스플레이 설정 (`display`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `web_enabled` | false | 웹 대시보드 자동 시작 |
| `web_port` | 5000 | 웹 대시보드 포트 |
| `opencv_live` | true | 라이브 모드에서 OpenCV 창 자동 활성화 |

**CLI 덮어쓰기:**
```bash
python main.py -m realtime --web  # 웹 대시보드 활성화
```

---

## 🚀 실제 사용 예제

### 예제 1: 기본 실시간 모드
```bash
python main.py -m realtime
```
- config.yaml의 기본값 사용
- 백그라운드 음성 감지 (pause_threshold=3초)
- OpenCV 창 자동 표시

### 예제 2: 병렬 호환 옵션
```bash
python main.py -m realtime --parallel -v
```
- 병렬 모드 호환 옵션 활성화
- 상세 로그 출력
- 현재 엔진은 순차 모니터링으로 동작

### 예제 3: 스피커 소리 감지용
```bash
python main.py -m realtime --static-threshold --energy-threshold 200
```
- 고정 에너지 임계값 (동적 조정 안 함)
- 낮은 민감도 (200)
- 스피커/유튜브 소리도 감지 가능

### 예제 4: 매우 민감한 설정 (마이크 소리만)
```bash
python main.py -m realtime --dynamic-threshold --energy-threshold 100
```
- 동적 에너지 임계값
- 높은 민감도 (100)
- 조용한 음성도 감지

### 예제 5: 빠른 분석 (성능 우선)
```bash
python main.py -m realtime \
  --energy-threshold 400 \
  --image-size 240 \
  --quality 50
```
- 이미지 크기 줄임 (240px)
- JPEG 품질 낮춤 (50)
- API 응답 빨라짐

### 예제 6: 정확한 분석 (품질 우선)
```bash
python main.py -m realtime \
  --image-size 768 \
  --quality 90 \
  --model gpt-4o
```
- 이미지 크기 증가 (768px)
- JPEG 품질 높임 (90)
- 더 강력한 모델 사용

---

## 📊 Config 검증

**현재 설정 상태 확인:**
```bash
# config.yaml의 설정 확인
cat config/config.yaml | grep -A 3 "speech:"
cat config/config.yaml | grep -A 3 "voice_analysis:"

# 코드에서 실제 기본값 확인
grep -r "energy_threshold" src/core/integrated_multimodal_system.py | head -1
grep -r "pause_threshold" src/core/integrated_multimodal_system.py | head -1
```

---

## ✅ 체크리스트

config.yaml 수정 후 확인할 것:

- [ ] 음성 인식: `speech` 섹션 존재 및 값 확인
  - `energy_threshold: 400`
  - `pause_threshold: 3.0`
  - `dynamic_threshold: false`

- [ ] 음성 분석: `voice_analysis` 섹션 존재 및 값 확인
  - `pitch.high_threshold: 250`
  - `speech_rate.fast_threshold: 5`

- [ ] OpenAI 설정: `openai` 섹션 존재 및 값 확인
  - `max_tokens: 800`
  - `temperature: 0.3`

- [ ] 분석 설정: `analysis` 섹션 확인
  - `voice_characteristics: true`
  - `parallel: false` (또는 필요에 따라)

---

**마지막 수정 날짜:** 2026-01-27
**상태:** 모든 설정 검증 완료 ✅
