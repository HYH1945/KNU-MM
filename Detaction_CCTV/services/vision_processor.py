from typing import List, Dict, Tuple
from ultralytics import YOLO
import numpy as np

class VisionProcessor:
    def __init__(self, model_path: str, confidence: float):
        print(f"[Vision] Loading AI Model: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print("[Vision] AI Model Loaded.")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        [수정사항]
        1. device='cpu': Mac Bus Error 방지
        2. classes=[0]: 사람만 감지 (YOLO COCO 기준 0번=Person)
        """
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.confidence, 
            verbose=False,
            classes=[0],   # 사람만 추적
            device='cpu'   # [중요] Mac 충돌 방지용 CPU 강제
        )
        
        # main.py에서 "raw_objects, _ = ..."로 받으므로 튜플 형태 유지
        annotated_frame = results[0].plot()
        
        detected_objects = []
        
        if results[0].boxes.id is not None:
            # CPU로 텐서를 이동시킨 후 Numpy 변환
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, clss):
                detected_objects.append({
                    'id': track_id,
                    'cls': cls_id,
                    'box': box.tolist(),  # [x1, y1, x2, y2]
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                })
                
        return detected_objects, annotated_frame