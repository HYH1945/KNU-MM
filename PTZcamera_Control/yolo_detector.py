import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import ultralytics
from ultralytics import YOLO


@dataclass(frozen=True)
class YoloConfig:
    model_path: str
    conf_threshold: float
    tracker_cfg_name: str


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_tracker_cfg(name: str) -> Optional[str]:
    if os.path.isfile(name):
        return name
    try:
        pkg_dir = Path(ultralytics.__file__).resolve().parent
        candidate = pkg_dir / "cfg" / "trackers" / name
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return None


class YoloDetector:
    def __init__(self, config: YoloConfig):
        self.config = config
        self.device = select_device()
        self.model = YOLO(config.model_path)
        self.tracker_cfg = resolve_tracker_cfg(config.tracker_cfg_name)
        self.use_tracking = self.tracker_cfg is not None

    def infer(self, frame):
        if self.use_tracking:
            try:
                return self.model.track(
                    frame,
                    conf=self.config.conf_threshold,
                    device=self.device,
                    tracker=self.tracker_cfg,
                    persist=True,
                    verbose=False,
                )
            except Exception as exc:
                print(f"Tracking error: {exc} -> fallback to detection")
                self.use_tracking = False
                self.model.predictor = None
        return self.model(
            frame,
            conf=self.config.conf_threshold,
            device=self.device,
            verbose=False,
        )
