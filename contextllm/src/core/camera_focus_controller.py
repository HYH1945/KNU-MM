#!/usr/bin/env python3
"""
카메라 포커스 제어 시스템
"""

class CameraFocusController:
    """카메라 포커스 제어"""
    
    def __init__(self, num_sources=4):
        self.num_sources = num_sources
        self.sources = [f"Camera_{i+1}" for i in range(num_sources)]
    
    def get_focus_command(self, priority):
        """우선순위에 따른 카메라 제어 명령 생성"""
        commands = {}
        
        # 우선순위별 포커스 레벨과 줌
        priority_map = {
            'CRITICAL': {'focus_level': 1, 'zoom': 2.5},
            'HIGH': {'focus_level': 2, 'zoom': 2.0},
            'MEDIUM': {'focus_level': 3, 'zoom': 1.5},
            'LOW': {'focus_level': 4, 'zoom': 1.0},
        }
        
        focus_config = priority_map.get(priority, priority_map['LOW'])
        
        for source in self.sources:
            commands[source] = {
                'focus_level': focus_config['focus_level'],
                'zoom': focus_config['zoom'],
                'priority': priority
            }
        
        return commands


class CameraFocusTranslator:
    """카메라 제어 신호를 플랫폼별 명령으로 변환"""
    
    def translate_to_ptz(self, focus_command):
        """PTZ (Pan-Tilt-Zoom) 형식으로 변환"""
        zoom_speed = focus_command['zoom']
        pan = {'speed': zoom_speed * 20}  # 팬 속도
        tilt = {'speed': zoom_speed * 20}  # 틸트 속도
        zoom = {'speed': zoom_speed}       # 줌 속도
        
        return {'pan': pan, 'tilt': tilt, 'zoom': zoom}
    
    def translate_to_manual(self, focus_command):
        """수동 제어 형식으로 변환"""
        return {
            'manual_focus': focus_command['focus_level'],
            'manual_zoom': focus_command['zoom'],
        }
