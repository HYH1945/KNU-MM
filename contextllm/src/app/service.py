"""Integration-friendly service facade for external systems."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.settings import build_settings, load_yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ContextLLMService:
    """
    Programmatic API for coupling ContextLLM with other systems.

    Usage:
        service = ContextLLMService.from_config(Path("config/config.yaml"))
        result = service.analyze_frame("도와주세요", frame)
    """

    def __init__(self, config_path: Path, overrides: Optional[Dict[str, Any]] = None):
        self.config_path = config_path.expanduser().resolve()
        self.raw_config = load_yaml(self.config_path)
        if overrides:
            self.raw_config = _deep_merge(self.raw_config, overrides)
        self.settings = build_settings(self.raw_config)

        # Synchronize core global config to avoid drift between CLI/service execution.
        from core.config_manager import set_runtime_config

        set_runtime_config(self.raw_config, config_path=self.config_path, merge=False, load_env_file=True)

        from core.integrated_multimodal_system import DownsamplingConfig, IntegratedMultimodalSystem

        ds = self.settings.downsampling
        self.system = IntegratedMultimodalSystem(
            camera_id=self.settings.video.camera_id,
            model=self.settings.model,
            downsampling_config=DownsamplingConfig(
                max_image_size=ds.max_image_size,
                jpeg_quality=ds.jpeg_quality,
                video_fps=ds.video_fps,
                max_video_frames=ds.max_video_frames,
                video_capture_duration=ds.video_capture_duration,
            ),
            energy_threshold=self.settings.speech.energy_threshold,
            pause_threshold=self.settings.speech.pause_threshold,
            dynamic_threshold=self.settings.speech.dynamic_threshold,
            enable_speech=False,
            llm_frame_count=self.settings.analysis.llm_frame_count,
            live_prebuffer_seconds=self.settings.analysis.prebuffer_seconds,
            live_postbuffer_seconds=self.settings.analysis.postbuffer_seconds,
            live_max_buffer_seconds=self.settings.analysis.buffer_window_seconds,
        )

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "ContextLLMService":
        return cls(Path(config_path), overrides=overrides)

    def analyze_frame(
        self,
        text: str,
        frame: Any,
        additional_context: Optional[str] = None,
        sound_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        analyzer = getattr(self.system, "multimodal_analyzer", None)
        if analyzer is None:
            return {"success": False, "error": "multimodal_analyzer_not_available"}

        working_frame = frame
        downsampler = getattr(self.system, "downsampler", None)
        if downsampler is not None:
            working_frame = downsampler.downsample_image(working_frame)

        merged_context = self._merge_additional_context(
            additional_context=additional_context,
            sound_event=sound_event,
        )

        try:
            analysis = analyzer.analyze_with_image(
                audio_text=text,
                image_source=working_frame,
                additional_context=merged_context,
            )
            return {
                "success": True,
                "priority": analysis.get("priority", "LOW"),
                "urgency": analysis.get("urgency", "LOW"),
                "is_emergency": bool(analysis.get("is_emergency", False)),
                "analysis": analysis,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def analyze_image(
        self,
        text: str,
        image_path: Union[str, Path],
        additional_context: Optional[str] = None,
        sound_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        analyzer = getattr(self.system, "multimodal_analyzer", None)
        if analyzer is None:
            return {"success": False, "error": "multimodal_analyzer_not_available"}

        merged_context = self._merge_additional_context(
            additional_context=additional_context,
            sound_event=sound_event,
        )

        try:
            analysis = analyzer.analyze_with_image(
                audio_text=text,
                image_source=str(image_path),
                additional_context=merged_context,
            )
            return {
                "success": True,
                "priority": analysis.get("priority", "LOW"),
                "urgency": analysis.get("urgency", "LOW"),
                "is_emergency": bool(analysis.get("is_emergency", False)),
                "analysis": analysis,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _merge_additional_context(
        self,
        additional_context: Optional[str],
        sound_event: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        contexts = []
        if additional_context:
            contexts.append(str(additional_context))

        sound_context = self._format_sound_event_context(sound_event)
        if sound_context:
            contexts.append(sound_context)

        return "\n\n".join(contexts) if contexts else None

    def _format_sound_event_context(self, sound_event: Optional[Dict[str, Any]]) -> str:
        if not sound_event:
            return ""

        formatter = getattr(self.system, "_format_sound_event_context", None)
        if callable(formatter):
            try:
                return str(formatter(sound_event))
            except Exception:
                pass

        top_event = sound_event.get("top_event", "unknown")
        top_confidence = float(sound_event.get("top_confidence", 0.0) or 0.0)
        triggered = bool(sound_event.get("triggered", False))
        return (
            "[비음성 이벤트]\n"
            f"- 최상위 이벤트: {top_event}\n"
            f"- 신뢰도: {top_confidence:.2f}\n"
            f"- 긴급 트리거: {'예' if triggered else '아니오'}"
        )
