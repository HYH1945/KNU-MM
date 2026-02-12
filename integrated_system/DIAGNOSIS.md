# integrated_system 통합 진단 리포트

작성일: 2026-02-12  
대상: `integrated_system` 프로토타입 (시각 PTZ + ReSpeaker DOA + STT + LLM)

## 1) 현재 구조 요약

현재 시스템은 다음 방식으로 동작합니다.

- **영상 루프(main loop)**: 프레임 수신 후 `YOLO -> ContextLLM -> ServerReporter` 파이프라인 실행.
- **음성 루프(백그라운드)**:
  - `MicArrayModule`이 별도 스레드에서 `SPEECHDETECTED/DOAANGLE`를 polling.
  - `STTModule`이 별도 스레드에서 오디오를 수집하고 Google STT 요청.
- **모듈 간 통신**: EventBus Pub/Sub.

즉, 영상과 음성은 “병렬 실행”은 맞지만, **멀티모달 융합 지점이 약해 실제 협업이 느슨한 상태**입니다.

---

## 2) 핵심 문제 진단 (우선순위 순)

## [A] Mic DOA PTZ 이동이 사실상 동작하지 않을 가능성 높음 (설정/구현 불일치)

### 증상

- “음성 방향으로 카메라를 돌리고 싶은데 잘 안 잡힘”과 직접 연결될 수 있는 결함.

### 근거

- Mic 모듈은 DOA가 확정되면 `move_type="absolute"`로 PTZ 이동 요청을 보냅니다.
- 그러나 PTZ 기본 설정은 `control_mode: "onvif"`입니다.
- `absolute` 이동은 Hikvision HTTP 경로에서만 처리되며, 해당 인증은 `hikvision_http` 또는 `both`일 때만 초기화됩니다.
- 따라서 기본값(`onvif`)에서는 Mic absolute 이동이 사실상 no-op이 됩니다.

### 영향

- Mic에서 방향을 잘 계산해도 PTZ가 반응하지 않거나, 반응이 매우 제한적으로 보임.

---

## [B] PTZ 우선순위 정책 때문에 YOLO 추적 중에는 Mic DOA 제어가 거의 못 들어감

### 증상

- 객체 탐지(추적)가 시작되면, 음성 방향 제어가 “먹히지 않는” 체감 발생.

### 근거

- PTZ 우선순위는 `YOLO_TRACKING(2) > MIC_DOA(1)`.
- PTZ 중재 로직은 낮은 우선순위 요청을 2초 윈도우에서 거절.
- YOLO는 추적 중 매 프레임 단위로 지속적으로 제어 요청을 발생시키므로, Mic 요청이 연속적으로 밀립니다.

### 영향

- 사용자가 기대한 “객체 탐지와 동시에 음성 방향 반영”이 구조적으로 어렵습니다.

---

## [C] “DOA 방향 + 객체 위치” 융합 로직 부재

### 증상

- 음성 방향을 추정했지만 어떤 객체를 우선 타겟팅할지 일관성이 떨어짐.

### 근거

- 현재는
  - YOLO: 화면상 객체 추적 우선순위로 target 선택
  - Mic: 별도로 PTZ absolute 이동 요청
- 두 정보를 합쳐 “DOA 근처 객체 점수 가산”하는 로직이 없음.

### 영향

- 음성 화자 방향 객체를 선택하는 멀티모달 타겟팅이 구현되지 않음.

---

## [D] STT 입력 소스/임계값/클라우드 인식 의존으로 인해 음성 인식 누락 가능성 높음

### 증상

- “음성탐지를 잘 못 잡는다” 체감.

### 근거

- STT는 `sr.Microphone()` 기본 장치 사용(장치 고정 선택 없음).
- 현장 환경에서 기본 마이크가 ReSpeaker가 아닐 수 있음.
- Google STT API 요청 실패(네트워크/쿼터/지연) 시 인식 누락.
- threshold/pause 설정이 현장 소음과 불일치하면 누락 증가.

### 영향

- 음성 이벤트와 텍스트 이벤트가 안정적으로 누적되지 않음.

---

## [E] LLM 분석 트리거가 “STT 텍스트 있을 때만”이라 비명/짧은 음성 등 비텍스트 상황에 취약

### 증상

- 음성이 있었는데 멀티모달 분석이 수행되지 않거나 늦게 수행됨.

### 근거

- ContextLLM은 내부적으로 pending STT text가 없으면 `reason: no_speech`로 반환.
- `mic.speech_detected` 자체는 로깅만 하고 분석 트리거로 사용하지 않음.

### 영향

- 실제 관제에서 중요한 음향 이벤트(고함, 비명, 충격음 유사 발화) 반영성이 낮음.

---

## 3) 왜 “동시에 처리하는데도 안 된다”처럼 보이는가?

기술적으로는 동시 실행(스레드)이 맞습니다. 다만 아래의 **의사결정 병목** 때문에 결과 체감이 떨어집니다.

1. Mic 방향 제어가 설정 미스매치로 실행 경로를 못 탔을 수 있음.  
2. 실행되더라도 PTZ 우선순위에서 YOLO가 지속 선점.  
3. YOLO와 Mic 신호를 결합해 단일 target 결정하는 융합 로직이 없음.  
4. STT는 장치/네트워크/임계값 영향으로 누락.

즉, 병렬성의 문제가 아니라 **융합 정책 + 제어 중재 + 장치 설정**의 문제입니다.

---

## 4) 개선 로드맵 (실행 우선순위)

## Step 1. PTZ 제어 경로 정합성부터 복구 (즉시)

- 운영 환경에서 Hikvision absolute를 쓸 거면 `ptz.control_mode: both` 또는 `hikvision_http`로 통일.
- onvif만 쓸 거면 Mic의 absolute 요청을 continuous 변환 레이어로 바꿔야 함.

## Step 2. Mic-DOA와 YOLO-Tracking 중재 정책 재설계

- 현행 우선순위 고정값 대신 **시간분할/가중치 기반 arbitration** 권장.
- 예시:
  - 최근 1.5초 내 STT 텍스트 또는 speech burst 발생 시 Mic 우선권 temporary boost.
  - YOLO target 유지 중이라도 DOA와 각도 차가 큰 경우에만 Mic snap-turn 허용.

## Step 3. 진짜 멀티모달 타겟 선택 함수 추가

- 각 객체에 대해 `fusion_score = visual_score + doa_alignment_bonus + speech_activity_bonus` 계산.
- 카메라 FOV 기준으로 `object_screen_x -> 추정 방위각` 맵핑 후 DOA와 차이를 점수화.

## Step 4. STT 안정화 (현장 적용 핵심)

- 마이크 장치 index를 설정 가능하게 하고, 시작 시 장치 리스트 로그 출력.
- `energy_threshold`, `pause_threshold`, `phrase_time_limit` 자동 튜닝 모드 도입.
- 네트워크 불안 대비(백오프/로컬 모델 fallback) 고려.

## Step 5. 관측성(Observability) 강화

- 이벤트 지연/유실 파악용 메트릭 추가:
  - mic speech rate, doa confidence histogram
  - stt success/fail ratio, avg latency
  - ptz owner timeline(yolo/mic/emergency)

---

## 5) 권장 실험 시나리오 (재현 가능 테스트)

1. **PTZ 경로 확인 테스트**
   - YOLO 비활성, Mic만 활성.
   - 여러 방향 발화 후 PTZ absolute 이동 여부 확인.

2. **중재 충돌 테스트**
   - YOLO로 사람 추적 중, 측면 발화 반복.
   - PTZ owner가 yolo에 고정되는지 로그로 검증.

3. **STT 장치 테스트**
   - 기본 장치 vs ReSpeaker 장치 지정 비교.
   - 동일 문장 20회 발화 인식률 측정.

4. **융합 타겟 테스트(개선 후)**
   - 화면 내 2인 이상, 한 명만 발화.
   - 발화자 쪽 객체가 지속 우선 타겟 되는지 측정.

---

## 6) 지금 당장 수정하면 체감 큰 항목 3개

1. `ptz.control_mode`와 Mic move_type 정합 맞추기.
2. PTZ arbitration에서 Mic temporary boost 규칙 추가.
3. STT 입력 장치 고정 설정(기본 장치 의존 제거).

이 3개만 해도 “음성 방향 반영이 안 된다”는 체감은 크게 줄어들 가능성이 높습니다.
