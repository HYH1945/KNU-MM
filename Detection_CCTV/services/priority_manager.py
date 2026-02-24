from typing import List, Dict
import math

class VisualPriorityManager:
    """
    탐지된 객체들의 우선순위를 평가하고 정렬하는 클래스
    """
    # --- 우선순위 결정 파라미터 ---
    
    # 각 요소의 가중치
    WEIGHTS = {
        'type': 0.6,
        'size': 0.3,
        'position': 0.1
    }
    
    # 객체 종류별 점수
    TYPE_SCORES = {
        'person': 1.0,  # 사람은 최우선
        'car': 0.5,
        'motorcycle': 0.5,
        'bus': 0.4,
        'truck': 0.4,
        'default': 0.2  # 나머지
    }

    def calculate_priorities(self, objects: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """
        모든 객체에 대해 우선순위 점수를 계산하고, 점수가 높은 순으로 정렬된 리스트를 반환합니다.

        [입력]
         - objects: ReID가 완료된 객체 리스트
         - frame_width: 프레임 너비
         - frame_height: 프레임 높이
        
        [출력]
         - 'priority_score'가 추가되고 점수 순으로 정렬된 객체 리스트
        """
        if not objects:
            return []

        frame_area = frame_width * frame_height
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        max_dist = math.sqrt(frame_center_x**2 + frame_center_y**2)

        for obj in objects:
            # 1. 종류 점수 (Type Score)
            obj_name = obj.get('name', '')
            type_score = self.TYPE_SCORES.get(obj_name, self.TYPE_SCORES['default'])

            # 2. 크기 점수 (Size Score)
            box = obj['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            size_score = area / frame_area if frame_area > 0 else 0

            # 3. 위치 점수 (Position Score)
            obj_center_x, obj_center_y = obj['center']
            dist = math.sqrt((obj_center_x - frame_center_x)**2 + (obj_center_y - frame_center_y)**2)
            position_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0
            
            # 최종 우선순위 점수 계산
            score = (self.WEIGHTS['type'] * type_score) + \
                    (self.WEIGHTS['size'] * size_score) + \
                    (self.WEIGHTS['position'] * position_score)
            
            obj['priority_score'] = score

        # 점수가 높은 순으로 객체 리스트 정렬
        sorted_objects = sorted(objects, key=lambda o: o.get('priority_score', 0), reverse=True)
        
        return sorted_objects