# 카메라 모듈 개발 가이드

현재 시스템은 **음성 기반 긴급 감지**로 구현되어 있습니다. 

나중에 **OpenCV 카메라**를 추가할 때, 아래 구조를 따르면 음성 및 카메라 기반 긴급 알람을 통합할 수 있습니다.

## 1. Alert Manager 공유 구조

모든 입력 소스(음성, 카메라)는 **`EmergencyAlertManager`를 공유**합니다:

```python
# src/core/alert_manager.py
from core.alert_manager import get_alert_manager

alert_manager = get_alert_manager()

# 어디서나 사용 가능
alert_manager.trigger_alert({
    'is_emergency': True,
    'emergency_reason': '침입자 감지!',
    'priority': 'CRITICAL',
    'situation_type': '보안'
})
```

## 2. 음성 기반 긴급 감지 (현재 ✅)

**파일:** `tests/test_free_realtime_simple.py`

```
마이크 입력 
  ↓
ChatGPT 분석
  ↓
긴급 판정 (is_emergency, priority)
  ↓
alert_manager.trigger_alert() 호출
  ↓
✅ 콘솔 경고 + 시스템 소리
```

## 3. 카메라 기반 긴급 감지 (미래 계획)

예상 파일: `src/camera/camera_monitor.py`

```python
#!/usr/bin/env python3
"""
OpenCV 기반 카메라 실시간 모니터링
음성 및 카메라 긴급 감지 통합
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.alert_manager import get_alert_manager

class CameraMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.alert_manager = get_alert_manager()
    
    def detect_emergency_from_camera(self, frame):
        """
        카메라 프레임에서 긴급 상황 감지
        - 사람 감지
        - 움직임 분석
        - 기타 컴퓨터 비전 기반 감지
        
        Returns:
            {
                'is_emergency': bool,
                'emergency_reason': str,
                'priority': str,
                'situation_type': str
            }
        """
        # TODO: 실제 구현
        pass
    
    def run(self):
        """카메라 실시간 모니터링"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. 카메라에서 긴급 감지
            emergency_info = self.detect_emergency_from_camera(frame)
            
            # 2. 긴급 알람 트리거
            if emergency_info['is_emergency']:
                self.alert_manager.trigger_alert(emergency_info)
            
            # 3. 프레임에 경고 표시 및 표시
            if emergency_info['is_emergency']:
                frame = self.alert_manager.draw_alert_on_frame(frame, emergency_info)
            
            # 4. 화면에 표시
            cv2.imshow('Emergency Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    monitor = CameraMonitor()
    monitor.run()
```

## 4. 통합 모니터링 시스템 (최종 목표)

예상 파일: `tests/test_integrated_monitoring.py`

```python
"""
음성 + 카메라 통합 실시간 모니터링

구조:
┌─────────────────────────────────────────┐
│   EmergencyAlertManager (공유)          │
│   - play_system_alert()                 │
│   - print_console_alert()               │
│   - draw_alert_on_frame()               │
└─────────────────────────────────────────┘
         ↑                    ↑
    [음성 감지]          [카메라 감지]
         ↓                    ↓
    마이크 입력           웹캠 입력
    ChatGPT 분석         컴퓨터 비전 분석
    긴급 판정            긴급 판정
    alert_manager        alert_manager
    
    → 통합 알람 및 표시
"""

import threading
from voice_monitor import VoiceMonitor
from camera_monitor import CameraMonitor

def run_integrated_monitoring():
    voice = VoiceMonitor()
    camera = CameraMonitor()
    
    # 음성과 카메라를 병렬로 실행
    voice_thread = threading.Thread(target=voice.run())
    camera_thread = threading.Thread(target=camera.run())
    
    voice_thread.start()
    camera_thread.start()
    
    voice_thread.join()
    camera_thread.join()
```

## 5. Alert Manager 주요 메서드

### `trigger_alert(emergency_info)`
- **용도:** 긴급 알람 트리거
- **입력:** 긴급 정보 딕셔너리
- **출력:** bool (알람 실행 여부)
- **특징:** 5초 쿨다운으로 반복 알람 방지

### `draw_alert_on_frame(frame, emergency_info)`
- **용도:** OpenCV 프레임에 경고 표시
- **입력:** numpy array (프레임), 긴급 정보
- **출력:** 경고가 그려진 프레임
- **표시 내용:**
  - 빨간 테두리 + 깜박임
  - "EMERGENCY ALERT!" 텍스트
  - 긴급 사유
  - 반투명 오버레이

### `play_system_alert(repeat=3, delay=0.2)`
- **용도:** 시스템 알람음 재생
- **플랫폼:** macOS/Linux

## 6. 개발 순서 (제안)

1. ✅ **음성 기반 긴급 감지** (완료)
   - ChatGPT 분석
   - alert_manager 통합

2. ⏳ **카메라 기반 긴급 감지** (다음 단계)
   - YOLO/OpenCV로 사람/움직임 감지
   - 위험 행동 인식
   - alert_manager 통합

3. ⏳ **통합 모니터링** (최종)
   - 음성 + 카메라 병렬 실행
   - 통일된 알람 시스템

## 7. 필요 라이브러리 (카메라용)

```bash
pip install opencv-python opencv-python-headless
# 선택: YOLO 기반 감지
pip install ultralytics torch torchvision
```

---

**현재 상태:** ✅ Alert Manager 준비 완료  
**다음 단계:** 카메라 모듈 개발 시 이 가이드 참고
