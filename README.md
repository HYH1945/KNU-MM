# KNU-MM

> 시각 + 청각 정보를 고려한 멀티모달 관제 시스템

## 개요

CCTV 영상(YOLO 객체 탐지) + 마이크 어레이(음원 방향 감지 + 음성 인식) → LLM 통합 분석을 통해 긴급 상황을 자동 판단하는 스마트 관제 시스템입니다.

## 모듈 구성

| 모듈 | 설명 | 디렉토리 |
|------|------|----------|
| **통합 시스템** | 모든 모듈을 하나의 파이프라인으로 통합 | [`integrated_system/`](integrated_system/) |
| YOLO + PTZ | 실시간 객체 탐지 + PTZ 추적 + Re-ID | [`Detaction_CCTV/`](Detaction_CCTV/) |
| 마이크 어레이 | ReSpeaker v2 DOA 음원 방향 감지 | [`mic_array_Control/`](mic_array_Control/) |
| Context LLM | GPT-4o-mini 기반 멀티모달 상황 분석 | [`contextllm/`](contextllm/) |
| Mic-Context Fusion | DOA/PTZ + STT + ContextLLM 이벤트 결합 러너 | [`mic_context_fusion/`](mic_context_fusion/) |
| PTZ 제어 | ONVIF PTZ 카메라 제어 | [`PTZcamera_Control/`](PTZcamera_Control/) |

## 최근 반영 사항

- `contextllm` config-first 실행 구조 정리 (`main.py` + `src/app/*`)
- `mic_context_fusion` 추가: `mic.doa_detected -> PTZ -> STT/non-speech -> frame capture -> multimodal`
- STT 인식 실패 오디오를 이벤트로 전달해 YAMNet 비음성 경로 연결
- `contextllm` testset에서 `mp3/wav/...` 오디오 파일 분석 지원 (YAMNet + 멀티모달)
- `contextllm` `media_test` 설정 추가: 이미지/비디오/오디오 조합 입력 테스트

## 빠른 시작

```bash
conda create -n knu-mm python=3.10 -y && conda activate knu-mm
cd integrated_system
pip install -r requirements.txt
python main.py
```

자세한 사용법은 [integrated_system/README.md](integrated_system/README.md) 참조.

<img width="658" height="721" alt="micarray 참고" src="https://github.com/user-attachments/assets/bca58658-6998-41b4-87a7-a6c0502ce6f1" />

mic array 각도 기준
