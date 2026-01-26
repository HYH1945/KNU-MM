#!/usr/bin/env python3
"""
긴급 알람 관리 모듈
- 음성 기반 긴급 감지
- 카메라 기반 긴급 표시 (미래용)
- 시스템 소리 및 시각적 경고 통합
"""

import os
import sys
import time


class EmergencyAlertManager:
    """긴급 상황 알람 및 경고 통합 관리"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.is_emergency = False
        self.last_alert_time = 0
        self.alert_cooldown = 5  # 동일 긴급 상황 5초 쿨다운
    
    def play_system_alert(self, repeat=3, delay=0.2):
        """
        시스템 알람음 재생 (macOS 중심, 다중 폴백 지원)
        - 1차: afplay + Alarm/Ping/Funk 사운드
        - 2차: osascript beep
        - 3차: say (음성합성)
        """
        import subprocess

        def try_afplay(sound_name: str) -> bool:
            sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
            volume = os.getenv("ALERT_VOLUME", "1.0")
            try:
                subprocess.run([
                    "/usr/bin/afplay", "-v", str(volume), sound_path
                ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                return False

        def try_beep() -> bool:
            try:
                subprocess.run(["/usr/bin/osascript", "-e", "beep"], check=True)
                return True
            except Exception:
                return False

        def try_say() -> bool:
            try:
                subprocess.run(["/usr/bin/say", "Emergency alert"], check=True)
                return True
            except Exception:
                return False

        # Try multiple common macOS system sounds
        candidate_sounds = [
            "Ping", "Glass", "Pop", "Basso", "Sosumi", "Submarine", "Tink", "Hero"
        ]

        for i in range(repeat):
            played = False
            for name in candidate_sounds:
                if try_afplay(name):
                    played = True
                    break

            if not played:
                played = try_beep() or try_say()

            if not played:
                print("   ⚠️ 알림음 재생 실패: afplay/osascript/say 모두 실패")
                return False

            if i < repeat - 1:
                time.sleep(delay)

        return True
    
    def print_console_alert(self, emergency_reason=""):
        """
        콘솔에 큰 경고 표시
        
        Args:
            emergency_reason: 긴급 사유
        """
        print("\n" + "🚨" * 35)
        print("🚨🚨🚨 ⚠️  **긴급 상황 감지됨!** ⚠️  🚨🚨🚨")
        print("🚨" * 35 + "\n")
        
        if emergency_reason:
            print(f"   🔴 긴급 사유: {emergency_reason}")
            print(f"   📞 즉시 대응 필요!\n")
    
    def trigger_alert(self, emergency_info):
        """
        긴급 알람 트리거 (음성, 카메라 공용)
        
        Args:
            emergency_info (dict): {
                'is_emergency': bool,
                'emergency_reason': str,
                'priority': str,
                'situation_type': str
            }
        
        Returns:
            bool: 알람 실행 여부
        """
        current_time = time.time()
        
        # 쿨다운 체크 (같은 긴급 상황 반복 방지)
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        is_emergency = emergency_info.get('is_emergency', False)
        priority = emergency_info.get('priority', 'LOW')
        
        if is_emergency or priority == 'CRITICAL':
            self.last_alert_time = current_time
            self.is_emergency = True
            
            # 1. 콘솔 경고
            reason = emergency_info.get('emergency_reason', '알 수 없는 긴급 상황')
            self.print_console_alert(reason)
            
            # 2. 시스템 소리
            self.play_system_alert(repeat=3, delay=0.2)
            
            return True
        
        self.is_emergency = False
        return False
    
    def draw_alert_on_frame(self, frame, emergency_info):
        """
        OpenCV 프레임에 경고 표시 (카메라용)
        
        Args:
            frame: OpenCV 이미지 (numpy array)
            emergency_info (dict): 긴급 정보
        
        Returns:
            frame: 경고가 그려진 프레임
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("⚠️  OpenCV가 설치되지 않았습니다")
            return frame
        
        is_emergency = emergency_info.get('is_emergency', False)
        priority = emergency_info.get('priority', 'LOW')
        
        if not (is_emergency or priority == 'CRITICAL'):
            return frame
        
        height, width = frame.shape[:2]
        
        # 1. 화면 전체에 빨간 테두리
        cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
        
        # 2. 반투명 빨간 오버레이 (화면의 1/4 투명도)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # 3. 텍스트: "🚨 긴급 상황!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_main = "EMERGENCY ALERT!"
        text_reason = emergency_info.get('emergency_reason', 'Unknown')
        
        # 메인 텍스트 (위쪽 중앙)
        font_scale = min(width, height) / 400  # 화면 크기에 따라 조정
        thickness = max(3, int(font_scale * 3))
        
        text_size = cv2.getTextSize(text_main, font, font_scale * 2, thickness)[0]
        x = (width - text_size[0]) // 2
        y = int(height * 0.3)
        
        # 텍스트 배경
        cv2.rectangle(frame, 
                     (x - 10, y - text_size[1] - 10),
                     (x + text_size[0] + 10, y + 10),
                     (0, 0, 255), -1)
        
        # 텍스트 (흰색)
        cv2.putText(frame, text_main, (x, y),
                   font, font_scale * 2, (255, 255, 255), thickness)
        
        # 긴급 사유 텍스트 (아래쪽)
        reason_size = cv2.getTextSize(text_reason, font, font_scale, thickness)[0]
        x_reason = (width - reason_size[0]) // 2
        y_reason = int(height * 0.6)
        
        cv2.putText(frame, text_reason, (x_reason, y_reason),
                   font, font_scale, (0, 255, 255), thickness)
        
        # 4. 깜박임 효과 (프레임 경계선 깜박임)
        # 매번 호출될 때마다 선의 굵기를 변경해서 깜박이는 효과
        # (실제로는 frame 호출 시간 기반으로 처리)
        blink_thickness = 5 if int(time.time() * 3) % 2 == 0 else 15
        cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), blink_thickness)
        
        return frame


# 전역 alert manager (음성 및 카메라 모듈에서 공유)
_alert_manager = None


def get_alert_manager():
    """전역 EmergencyAlertManager 인스턴스 반환"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = EmergencyAlertManager()
    return _alert_manager


# 사용 예시
if __name__ == "__main__":
    alert_mgr = get_alert_manager()
    
    # 테스트 1: 긴급 알람
    print("테스트 1: 긴급 알람 트리거")
    emergency_info = {
        'is_emergency': True,
        'emergency_reason': '침입자 감지!',
        'priority': 'CRITICAL',
        'situation_type': '보안'
    }
    alert_mgr.trigger_alert(emergency_info)
    
    # 테스트 2: OpenCV 프레임 경고 표시
    print("\n테스트 2: OpenCV 프레임 경고 표시")
    try:
        import cv2
        import numpy as np
        
        # 샘플 프레임 생성
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # 어두운 배경
        
        # 경고 표시
        alert_frame = alert_mgr.draw_alert_on_frame(frame, emergency_info)
        
        print("✅ 프레임에 경고 표시 완료")
        # 실제로는 여기서 cv2.imshow()로 표시
    except ImportError:
        print("⚠️  OpenCV 미설치 - 프레임 경고 표시 불가")
