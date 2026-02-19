#!/usr/bin/env python3
"""
YAMNet 기반 비음성/환경음 이벤트 감지기

실시간 입력 오디오에서 위험 가능성이 높은 이벤트(비명, 유리 파손 등)를 탐지합니다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_YAMNET_AVAILABLE = True
except ImportError:
    tf = None
    hub = None
    TF_YAMNET_AVAILABLE = False


class SoundEventDetector:
    """YAMNet 기반 사운드 이벤트 감지기"""

    def __init__(
        self,
        model_url: str = "https://tfhub.dev/google/yamnet/1",
        min_confidence: float = 0.12,
        trigger_threshold: float = 0.25,
        top_k: int = 5,
        emergency_keywords: Optional[List[str]] = None,
    ):
        self.enabled = False
        self.model_url = model_url
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.trigger_threshold = max(0.0, min(1.0, float(trigger_threshold)))
        self.top_k = max(1, int(top_k))
        self.target_sample_rate = 16000
        self.class_names: List[str] = []
        self.emergency_keywords = [
            kw.lower() for kw in (emergency_keywords or [
                "scream",
                "shout",
                "yell",
                "glass",
                "breaking",
                "gunshot",
                "explosion",
                "alarm",
                "siren",
            ])
        ]

        if not TF_YAMNET_AVAILABLE:
            return

        try:
            self.model = hub.load(self.model_url)
            self.class_names = self._load_class_names()
            self.enabled = True
        except Exception:
            self.enabled = False

    def _load_class_names(self) -> List[str]:
        """YAMNet 클래스 이름 로드"""
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        class_names: List[str] = []

        with open(class_map_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                display_name = row.get("display_name", "")
                class_names.append(display_name)

        return class_names

    def _resample_linear(self, samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """가벼운 선형 보간 리샘플링"""
        if src_rate == dst_rate or len(samples) == 0:
            return samples.astype(np.float32, copy=False)

        duration = len(samples) / float(src_rate)
        dst_length = max(1, int(duration * dst_rate))
        src_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
        dst_times = np.linspace(0.0, duration, num=dst_length, endpoint=False)
        resampled = np.interp(dst_times, src_times, samples)
        return resampled.astype(np.float32, copy=False)

    def _audio_to_float_mono(self, audio: Any) -> Optional[np.ndarray]:
        """SpeechRecognition AudioData를 float32 mono waveform으로 변환"""
        if audio is None:
            return None

        try:
            raw = audio.get_raw_data()
            sample_rate = int(getattr(audio, "sample_rate", self.target_sample_rate) or self.target_sample_rate)
            sample_width = int(getattr(audio, "sample_width", 2) or 2)

            if sample_width == 2:
                waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                waveform = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                return None

            waveform = np.clip(waveform, -1.0, 1.0)
            if sample_rate != self.target_sample_rate:
                waveform = self._resample_linear(waveform, sample_rate, self.target_sample_rate)

            return waveform
        except Exception:
            return None

    def _is_emergency_label(self, label: str) -> bool:
        label_lower = (label or "").lower()
        return any(keyword in label_lower for keyword in self.emergency_keywords)

    def detect_from_audio(self, audio: Any) -> Dict[str, Any]:
        """오디오에서 이벤트 감지"""
        default_result = {
            "enabled": self.enabled,
            "model": "yamnet",
            "event_detected": False,
            "triggered": False,
            "top_event": None,
            "top_confidence": 0.0,
            "events": [],
            "emergency_events": [],
        }

        if not self.enabled:
            return default_result

        waveform = self._audio_to_float_mono(audio)
        if waveform is None or waveform.size < int(self.target_sample_rate * 0.2):
            return default_result

        try:
            scores, _, _ = self.model(tf.convert_to_tensor(waveform, dtype=tf.float32))
            score_mean = tf.reduce_mean(scores, axis=0).numpy()

            top_indices = np.argsort(score_mean)[::-1][: self.top_k]
            events: List[Dict[str, Any]] = []
            emergency_events: List[Dict[str, Any]] = []

            for idx in top_indices:
                confidence = float(score_mean[idx])
                if confidence < self.min_confidence:
                    continue

                label = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                item = {
                    "label": label,
                    "confidence": confidence,
                    "is_emergency": self._is_emergency_label(label),
                }
                events.append(item)
                if item["is_emergency"]:
                    emergency_events.append(item)

            if not events:
                return default_result

            top_event = events[0]
            triggered = any(evt["confidence"] >= self.trigger_threshold for evt in emergency_events)

            return {
                "enabled": True,
                "model": "yamnet",
                "event_detected": len(emergency_events) > 0,
                "triggered": triggered,
                "top_event": top_event["label"],
                "top_confidence": top_event["confidence"],
                "events": events,
                "emergency_events": emergency_events,
            }

        except Exception:
            return default_result
