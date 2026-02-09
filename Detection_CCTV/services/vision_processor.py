from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def _select_device(preferred: str) -> str:
    pref = (preferred or "auto").strip().lower()
    if pref == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_tracker_cfg(name: str) -> Optional[str]:
    if not name:
        return None
    candidate = Path(name)
    if candidate.is_file():
        return str(candidate)
    try:
        pkg_dir = Path(ultralytics.__file__).resolve().parent
        fallback = pkg_dir / "cfg" / "trackers" / name
        if fallback.exists():
            return str(fallback)
    except Exception:
        pass
    return None

class VisionProcessor:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        device: str = "auto",
        tracker_cfg: str = "",
        use_tracking: bool = True,
    ):
        print(f"[Vision] Loading AI Model: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = _select_device(device)
        self.tracker_cfg = _resolve_tracker_cfg(tracker_cfg)
        self.use_tracking = bool(use_tracking)
        print("[Vision] AI Model Loaded.")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        results = None
        if self.use_tracking:
            try:
                if self.tracker_cfg:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.confidence,
                        verbose=False,
                        classes=[0],
                        device=self.device,
                        tracker=self.tracker_cfg,
                    )
                else:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.confidence,
                        verbose=False,
                        classes=[0],
                        device=self.device,
                    )
            except Exception as exc:
                print(f"[Vision] Tracking error: {exc} -> fallback to detection")
                self.use_tracking = False
                self.model.predictor = None

        if results is None:
            results = self.model(
                frame,
                conf=self.confidence,
                verbose=False,
                classes=[0],
                device=self.device,
            )
        
        # main.py에서 "raw_objects, _ = ..."로 받으므로 튜플 형태 유지
        annotated_frame = results[0].plot()
        
        detected_objects = []
        
        if results[0].boxes is not None and results[0].boxes.xyxy is not None:
            # CPU로 텐서를 이동시킨 후 Numpy 변환
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(boxes), dtype=int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, clss):
                detected_objects.append({
                    'id': track_id,
                    'cls': cls_id,
                    'box': box.tolist(),  # [x1, y1, x2, y2]
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                })
                
        return detected_objects, annotated_frame
