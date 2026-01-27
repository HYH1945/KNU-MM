from typing import List, Dict, Optional

class VisualPriorityManager:
    """
    ReID가 적용된 객체들 중에서 '누구를 추적할지' 결정하는 클래스
    """
    def __init__(self):
        self.current_target_perm_id: Optional[int] = None
        self.LOCK_BONUS = 50.0   # 한 번 문 대상은 놓치지 않도록 가산점 부여
        self.SIZE_WEIGHT = 1.0   # 가까운(큰) 물체 선호 가중치

    def select_target(self, objects: List[Dict], frame_width: int) -> Optional[Dict]:
        """
        [입력]
         - objects: ReID가 완료된 객체 리스트 (permanent_id 포함)
         - frame_width: 화면 너비 (중앙 계산용, 여기서는 크기 점수용으로 활용 가능)
        
        [출력]
         - 최우선 추적 대상 객체 1개 (없으면 None)
        """
        best_obj = None
        max_score = -1.0
        
        # 화면 전체 면적 (대략적인 값, 가로x가로 비율로 계산)
        # 실제 높이를 안 받는 구조라 너비 제곱으로 근사치 사용 혹은 단순 크기 비교
        
        for obj in objects:
            score = 0.0
            
            # 1. 크기 점수 (화면에 크게 잡힐수록 = 가까울수록 점수 높음)
            # box: [x1, y1, x2, y2]
            box = obj['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            
            # 면적이 클수록 점수 높음 (단순 비례)
            score += (area / 1000.0) * self.SIZE_WEIGHT
            
            # 2. 락킹(Locking) 보너스 (Hysteresis)
            # 이전에 추적하던 영구 ID(permanent_id)와 같으면 가산점
            # YOLO ID가 바뀌어도 영구 ID는 유지되므로 추적이 끊기지 않음
            if self.current_target_perm_id is not None:
                if obj.get('permanent_id') == self.current_target_perm_id:
                    score += self.LOCK_BONUS

            # 디버깅용 점수 기록
            obj['priority_score'] = score
            
            # 최댓값 갱신
            if score > max_score:
                max_score = score
                best_obj = obj

        # 타겟 갱신 로직
        if best_obj:
            # 타겟이 선정되면 그 ID를 기억해둠 (다음 프레임에 가산점 주려고)
            self.current_target_perm_id = best_obj.get('permanent_id')
        else:
            # 아무도 없으면 타겟 해제
            self.current_target_perm_id = None
            
        return best_obj