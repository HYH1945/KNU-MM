from typing import List, Dict, Optional
import cv2
import numpy as np


class HeatmapOverlay:
    """
    간단한 누적 히트맵 오버레이.
    - 객체 중심점을 누적
    - 프레임마다 감쇠(decay) 적용
    - 컬러맵으로 변환 후 알파 블렌딩
    """

    def __init__(
        self,
        alpha: float = 0.35,
        decay: float = 0.94,
        radius: int = 24,
        intensity: float = 1.0,
        downscale: int = 2,
        min_value: float = 0.01,
        colormap: int = cv2.COLORMAP_JET,
    ):
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.decay = float(max(0.0, min(0.999, decay)))
        self.radius = int(max(1, radius))
        self.intensity = float(max(0.01, intensity))
        self.downscale = int(max(1, downscale))
        self.min_value = float(max(0.0, min_value))
        self.colormap = colormap

        self._heatmap: Optional[np.ndarray] = None
        self._hm_w: int = 0
        self._hm_h: int = 0

    def _ensure_shape(self, frame_shape) -> None:
        h, w = frame_shape[:2]
        hm_w = max(1, int(w // self.downscale))
        hm_h = max(1, int(h // self.downscale))
        if self._heatmap is None or hm_w != self._hm_w or hm_h != self._hm_h:
            self._hm_w = hm_w
            self._hm_h = hm_h
            self._heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)

    def update(self, frame, objects: List[Dict], add_points: bool = True) -> None:
        """누적 히트맵 갱신. add_points=False면 감쇠만 적용."""
        self._ensure_shape(frame.shape)
        if self._heatmap is None:
            return

        # 감쇠
        self._heatmap *= self.decay

        if not add_points:
            return

        radius = max(1, int(self.radius // self.downscale))
        for obj in objects:
            cx, cy = obj.get("center", (None, None))
            if cx is None or cy is None:
                continue
            x = int(cx // self.downscale)
            y = int(cy // self.downscale)
            if 0 <= x < self._hm_w and 0 <= y < self._hm_h:
                cv2.circle(self._heatmap, (x, y), radius, self.intensity, -1)

    def apply(self, frame):
        """프레임에 히트맵을 오버레이."""
        if self._heatmap is None:
            return frame
        if self._heatmap.max() <= self.min_value:
            return frame

        hm_norm = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        hm_uint8 = hm_norm.astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, self.colormap)

        if self.downscale != 1:
            hm_color = cv2.resize(
                hm_color,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return cv2.addWeighted(frame, 1.0 - self.alpha, hm_color, self.alpha, 0)

    def clear(self) -> None:
        """누적 히트맵 초기화."""
        if self._heatmap is not None:
            self._heatmap.fill(0.0)
