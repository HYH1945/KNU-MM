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
| PTZ 제어 | ONVIF PTZ 카메라 제어 | [`PTZcamera_Control/`](PTZcamera_Control/) |

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
