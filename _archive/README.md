# _archive — 통합 전 프로토타입 모듈

이 폴더에는 `integrated_system/`으로 통합되기 전에 개별적으로 개발된 프로토타입 모듈들이 보관되어 있습니다.

**현재 실행에는 사용되지 않으며**, 참고/롤백 용도로 보관합니다.

| 폴더 | 설명 | 통합 위치 |
|---|---|---|
| `Detection_CCTV/` | BoT-SORT + heatmap CCTV 모듈 | 참고용 |
| `integrated_system_process/` | integrated_system 포크 | `integrated_system/` |
| `mic_context_fusion/` | 독립 MicArray + LLM 러너 | `integrated_system/main.py` |
| `PTZcamera_Control/` | PTZ 카메라 초기 테스트 | `integrated_system/modules/ptz_controller.py` |
| `mic_array_Control/` | 마이크 어레이 초기 테스트 | `integrated_system/modules/mic_array.py` |
