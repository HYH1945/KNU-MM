import cv2
import numpy as np
from typing import Dict, Tuple, List

class ReIDManager:
    """
    객체의 외형(히스토그램)을 저장하여, 화면 밖으로 나갔다 돌아와도
    기존 ID를 다시 부여해주는 재식별(Re-Identification) 관리자
    """
    def __init__(self, similarity_threshold: float = 0.70):
        # 영구 ID 관리: { permanent_id: {'hist': histogram, 'last_seen': time, 'name': 'Person X'} }
        self.known_objects: Dict[int, Dict] = {}
        
        # 현재 YOLO ID와 영구 ID 매핑: { yolo_track_id: permanent_id }
        self.id_map: Dict[int, int] = {}
        
        self.next_uid = 1  # 부여할 영구 ID 번호
        self.threshold = similarity_threshold

    def _calculate_histogram(self, image_crop):
        """이미지 조각에서 색상 분포(Fingerprint) 추출"""
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        # Hue(색상)와 Saturation(채도)만 사용 (조명 변화 영향 최소화)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update_ids(self, frame, yolo_objects: List[Dict]) -> List[Dict]:
        """
        YOLO가 감지한 객체 리스트를 받아, 영구 ID(Permanent ID)로 변환하여 반환
        """
        current_yolo_ids = set()
        processed_objects = []
        
        # 화면 크기 가져오기 (좌표 벗어남 방지용)
        frame_h, frame_w = frame.shape[:2]

        for obj in yolo_objects:
            yolo_id = obj['id']
            box = obj['box']
            current_yolo_ids.add(yolo_id)
            
            # [수정] 좌표를 정수(int)로 변환하고 화면 범위 내로 제한(Clamping)
            # 이렇게 해야 TypeError: slice indices must be integers 오류가 사라집니다.
            x1 = int(max(0, box[0]))
            y1 = int(max(0, box[1]))
            x2 = int(min(frame_w, box[2]))
            y2 = int(min(frame_h, box[3]))

            # 유효하지 않은 박스(크기가 0이거나 음수)는 건너뜀
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 이미지 자르기 (Slicing)
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0: 
                continue
            
            current_hist = self._calculate_histogram(person_roi)

            # 1. 이미 매핑된 YOLO ID인가? (화면 내에서 계속 추적 중)
            if yolo_id in self.id_map:
                perm_id = self.id_map[yolo_id]
                self.known_objects[perm_id]['hist'] = current_hist
            
            else:
                # 2. 새로운 YOLO ID 등장 -> 과거의 누군가인지 검색 (Re-ID)
                found_match = False
                best_score = -1.0
                matched_perm_id = -1

                for perm_id, data in self.known_objects.items():
                    # 현재 화면에 없는 사람하고만 비교
                    if perm_id in self.id_map.values():
                        continue
                        
                    # 히스토그램 유사도 비교
                    score = cv2.compareHist(data['hist'], current_hist, cv2.HISTCMP_CORREL)
                    
                    if score > best_score:
                        best_score = score
                        matched_perm_id = perm_id

                # 유사도가 임계값 이상이면 -> "아까 그 사람이다!"
                if best_score > self.threshold:
                    perm_id = matched_perm_id
                    self.id_map[yolo_id] = perm_id
                    self.known_objects[perm_id]['hist'] = current_hist
                    # print(f"🔄 Re-ID Success: YOLO {yolo_id} -> Person {perm_id} ({best_score:.2f})")
                else:
                    # 3. 정말 새로운 사람 -> 신규 ID 발급
                    perm_id = self.next_uid
                    self.next_uid += 1
                    self.id_map[yolo_id] = perm_id
                    self.known_objects[perm_id] = {
                        'hist': current_hist,
                        'name': f"Person {perm_id}"
                    }

            # 결과 객체에 영구 ID 정보 주입
            obj['permanent_id'] = perm_id
            obj['name'] = self.known_objects[perm_id]['name']
            
            # 박스 좌표도 정수형으로 업데이트해줌 (화면 그리기용)
            obj['box'] = [x1, y1, x2, y2]
            
            processed_objects.append(obj)

        # 화면에서 사라진 YOLO ID는 매핑에서 제거
        active_yolo_ids = list(self.id_map.keys())
        for old_id in active_yolo_ids:
            if old_id not in current_yolo_ids:
                del self.id_map[old_id]

        return processed_objects