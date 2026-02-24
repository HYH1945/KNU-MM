"""Typed settings for config-first ContextLLM runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


VALID_MODES = {"realtime", "testset", "file", "webcam", "network"}


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


@dataclass
class VideoSettings:
    camera_id: int = 0
    testset_path: str = "testsets"
    network_url: str = ""
    file_path: str = ""


@dataclass
class AnalysisSettings:
    default_text: str = ""
    iterations: Optional[int] = None
    analyze_all_testset: bool = False
    testset_index: int = 0
    llm_frame_count: int = 4
    prebuffer_seconds: Optional[float] = None
    postbuffer_seconds: Optional[float] = None
    buffer_window_seconds: Optional[float] = None
    voice_characteristics: bool = True
    streaming: bool = False
    parallel: bool = False


@dataclass
class DownsamplingSettings:
    max_image_size: int = 640
    jpeg_quality: int = 75
    video_fps: float = 2.0
    max_video_frames: int = 10
    video_capture_duration: float = 5.0


@dataclass
class LoggingSettings:
    verbose: bool = False
    save_results: bool = True
    log_dir: str = "data/logs"


@dataclass
class DisplaySettings:
    web_enabled: bool = False
    web_port: int = 5000
    opencv_live: bool = True
    keep_open_after_run: bool = False


@dataclass
class SpeechSettings:
    enabled: bool = True
    energy_threshold: int = 400
    pause_threshold: float = 3.0
    dynamic_threshold: bool = False


@dataclass
class MediaTestSettings:
    enabled: bool = False
    image_path: str = ""
    video_path: str = ""
    audio_path: str = ""
    text_input: str = ""
    phrase_time_limit: float = 6.0


@dataclass
class ContextLLMSettings:
    mode: str = "realtime"
    model: str = "gpt-4o-mini"
    video: VideoSettings = field(default_factory=VideoSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    downsampling: DownsamplingSettings = field(default_factory=DownsamplingSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    media_test: MediaTestSettings = field(default_factory=MediaTestSettings)
    raw: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.mode = self.mode.strip().lower()
        if self.mode not in VALID_MODES:
            self.mode = "realtime"


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML이 필요합니다. 설치: pip install pyyaml")
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_settings(raw: Dict[str, Any]) -> ContextLLMSettings:
    video_raw = raw.get("video", {}) or {}
    analysis_raw = raw.get("analysis", {}) or {}
    ds_raw = raw.get("downsampling", {}) or {}
    logging_raw = raw.get("logging", {}) or {}
    display_raw = raw.get("display", {}) or {}
    speech_raw = raw.get("speech", {}) or {}
    media_test_raw = raw.get("media_test", {}) or {}

    settings = ContextLLMSettings(
        mode=_to_str(raw.get("mode"), "realtime"),
        model=_to_str(raw.get("model"), "gpt-4o-mini"),
        video=VideoSettings(
            camera_id=_to_int(video_raw.get("camera_id"), 0),
            testset_path=_to_str(video_raw.get("testset_path"), "testsets"),
            network_url=_to_str(video_raw.get("network_url"), ""),
            file_path=_to_str(video_raw.get("file_path"), ""),
        ),
        analysis=AnalysisSettings(
            default_text=_to_str(analysis_raw.get("default_text"), ""),
            iterations=None if analysis_raw.get("iterations") is None else _to_int(analysis_raw.get("iterations"), 0),
            analyze_all_testset=_to_bool(analysis_raw.get("analyze_all_testset"), False),
            testset_index=_to_int(analysis_raw.get("testset_index"), 0),
            llm_frame_count=max(1, _to_int(analysis_raw.get("llm_frame_count"), 4)),
            prebuffer_seconds=_to_optional_float(analysis_raw.get("prebuffer_seconds")),
            postbuffer_seconds=_to_optional_float(analysis_raw.get("postbuffer_seconds")),
            buffer_window_seconds=_to_optional_float(analysis_raw.get("buffer_window_seconds")),
            voice_characteristics=_to_bool(analysis_raw.get("voice_characteristics"), True),
            streaming=_to_bool(analysis_raw.get("streaming"), False),
            parallel=_to_bool(analysis_raw.get("parallel"), False),
        ),
        downsampling=DownsamplingSettings(
            max_image_size=_to_int(ds_raw.get("max_image_size"), 640),
            jpeg_quality=_to_int(ds_raw.get("jpeg_quality"), 75),
            video_fps=_to_float(ds_raw.get("video_fps"), 2.0),
            max_video_frames=_to_int(ds_raw.get("max_video_frames"), 10),
            video_capture_duration=_to_float(ds_raw.get("video_capture_duration"), 5.0),
        ),
        logging=LoggingSettings(
            verbose=_to_bool(logging_raw.get("verbose"), False),
            save_results=_to_bool(logging_raw.get("save_results"), True),
            log_dir=_to_str(logging_raw.get("log_dir"), "data/logs"),
        ),
        display=DisplaySettings(
            web_enabled=_to_bool(display_raw.get("web_enabled"), False),
            web_port=_to_int(display_raw.get("web_port"), 5000),
            opencv_live=_to_bool(display_raw.get("opencv_live"), True),
            keep_open_after_run=_to_bool(display_raw.get("keep_open_after_run"), False),
        ),
        speech=SpeechSettings(
            enabled=_to_bool(speech_raw.get("enabled"), True),
            energy_threshold=_to_int(speech_raw.get("energy_threshold"), 400),
            pause_threshold=_to_float(speech_raw.get("pause_threshold"), 3.0),
            dynamic_threshold=_to_bool(speech_raw.get("dynamic_threshold"), False),
        ),
        media_test=MediaTestSettings(
            enabled=_to_bool(media_test_raw.get("enabled"), False),
            image_path=_to_str(media_test_raw.get("image_path"), ""),
            video_path=_to_str(media_test_raw.get("video_path"), ""),
            audio_path=_to_str(media_test_raw.get("audio_path"), ""),
            text_input=_to_str(media_test_raw.get("text_input"), ""),
            phrase_time_limit=_to_float(media_test_raw.get("phrase_time_limit"), 6.0),
        ),
        raw=dict(raw),
    )
    settings.validate()
    return settings


def load_settings(config_path: Path) -> ContextLLMSettings:
    raw = load_yaml(config_path)
    return build_settings(raw)
