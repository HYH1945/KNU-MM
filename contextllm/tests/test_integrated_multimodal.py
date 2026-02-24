#!/usr/bin/env python3
"""Automated tests for the integrated multimodal system.

Interactive/hardware tests are marked as manual and skipped by default.
Set RUN_MANUAL_TESTS=1 to enable them.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.integrated_multimodal_system import (
    DownsamplingConfig,
    FileVideoSource,
    TestsetVideoSource as VideoTestsetSource,
    VideoDownsampler,
    create_video_source,
    IntegratedMultimodalSystem,
)
import core.integrated_multimodal_system as ims

ROOT = Path(__file__).resolve().parent.parent
TESTSETS = ROOT / "testsets"

manual_only = pytest.mark.skipif(
    os.getenv("RUN_MANUAL_TESTS") != "1",
    reason="manual/hardware test (set RUN_MANUAL_TESTS=1 to run)",
)


def test_downsampling_image_and_frames():
    config = DownsamplingConfig(
        max_image_size=320,
        jpeg_quality=70,
        video_fps=2.0,
        max_video_frames=5,
        video_resolution_scale=0.5,
    )
    downsampler = VideoDownsampler(config)

    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    reduced = downsampler.downsample_image(image)
    assert max(reduced.shape[:2]) <= 320

    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(20)]
    reduced_frames, timestamps = downsampler.downsample_video_frames(frames)
    assert len(reduced_frames) == 5
    assert timestamps == []


def test_file_video_source_image_capture():
    image_path = TESTSETS / "crime1.png"
    source = FileVideoSource(str(image_path))

    assert source.open() is True
    frame = source.capture_frame()
    assert frame is not None
    assert frame.ndim == 3
    source.close()


def test_testset_source_lists_files():
    source = VideoTestsetSource(str(TESTSETS))
    assert source.open() is True
    files = source.list_files()
    assert len(files) >= 1
    assert any(f.endswith((".png", ".jpg", ".jpeg", ".mp4", ".mov", ".avi")) for f in files)
    source.close()


def test_factory_creates_testset_source():
    source = create_video_source("testset", folder_path=str(TESTSETS))
    assert isinstance(source, VideoTestsetSource)


def test_integrated_system_can_init_without_speech():
    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=DownsamplingConfig(max_image_size=320),
        enable_speech=False,
    )

    assert system.speech_detector is None
    system.use_testset(str(TESTSETS))
    files = system.get_testset_files()
    assert len(files) >= 1


def test_llm_frame_sampling_uses_multi_image_analyzer():
    class FakeAnalyzer:
        def __init__(self):
            self.single_calls = 0
            self.multi_calls = 0
            self.multi_image_count = 0

        def analyze_with_image(self, **kwargs):
            self.single_calls += 1
            return {"priority": "LOW", "urgency": "LOW", "is_emergency": False}

        def analyze_with_images(self, **kwargs):
            self.multi_calls += 1
            images = kwargs.get("image_sources") or []
            self.multi_image_count = len(images)
            return {"priority": "LOW", "urgency": "LOW", "is_emergency": False}

    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=DownsamplingConfig(max_image_size=320, max_video_frames=10),
        enable_speech=False,
        llm_frame_count=3,
    )
    fake = FakeAnalyzer()
    system.multimodal_analyzer = fake

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
    llm_frames = system._select_llm_frames(frames)
    assert len(llm_frames) == 3

    result = system._analyze_frames_with_llm("테스트", llm_frames)
    assert result is not None
    assert fake.multi_calls == 1
    assert fake.single_calls == 0
    assert fake.multi_image_count == 3


def test_integrated_system_accepts_live_buffer_overrides():
    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=DownsamplingConfig(max_image_size=320),
        enable_speech=False,
        live_prebuffer_seconds=1.1,
        live_postbuffer_seconds=0.4,
        live_max_buffer_seconds=9.5,
    )
    assert system.live_prebuffer_seconds == pytest.approx(1.1)
    assert system.live_postbuffer_seconds == pytest.approx(0.4)
    assert system.live_max_buffer_seconds == pytest.approx(9.5)


def test_resolve_stt_languages_includes_config_and_defaults():
    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=DownsamplingConfig(max_image_size=320),
        enable_speech=False,
    )
    system.speech_config = {"recognition_languages": ["en-US", "ko-KR"], "language": "ja-JP"}

    langs = system._resolve_stt_languages()
    assert langs[:3] == ["en-US", "ko-KR", "ja-JP"]
    assert "ko-KR" in langs
    assert "en-US" in langs


def test_transcribe_audio_file_falls_back_language_candidates(monkeypatch):
    class FakeUnknownValueError(Exception):
        pass

    class FakeRequestError(Exception):
        pass

    calls = []

    class FakeRecognizer:
        def record(self, _source):
            return "fake-audio"

        def recognize_google(self, _audio, language="en-US"):
            calls.append(language)
            if language == "ko-KR":
                raise FakeUnknownValueError()
            if language == "en-US":
                return "help me"
            raise FakeUnknownValueError()

    class FakeAudioFile:
        def __init__(self, _path):
            self.path = _path

        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_sr = types.SimpleNamespace(
        Recognizer=FakeRecognizer,
        AudioFile=FakeAudioFile,
        UnknownValueError=FakeUnknownValueError,
        RequestError=FakeRequestError,
    )

    monkeypatch.setattr(ims, "sr", fake_sr)
    monkeypatch.setattr(ims, "SPEECH_RECOGNITION_AVAILABLE", True)

    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=DownsamplingConfig(max_image_size=320),
        enable_speech=False,
    )
    system.speech_config = {"recognition_languages": ["ko-KR", "en-US"]}

    text = system._transcribe_audio_file("dummy.wav")
    assert text == "help me"
    assert calls[:2] == ["ko-KR", "en-US"]


def test_speech_detector_recognize_audio_with_fallback(monkeypatch):
    class FakeUnknownValueError(Exception):
        pass

    class FakeRequestError(Exception):
        pass

    class FakeRecognizer:
        def __init__(self):
            self.calls = []

        def recognize_google(self, _audio, language="ko-KR"):
            self.calls.append(language)
            if language == "ko-KR":
                raise FakeUnknownValueError()
            if language == "en-US":
                return "stop now"
            raise FakeUnknownValueError()

    fake_sr = types.SimpleNamespace(
        UnknownValueError=FakeUnknownValueError,
        RequestError=FakeRequestError,
    )
    monkeypatch.setattr(ims, "sr", fake_sr)

    detector = ims.SpeechDetector.__new__(ims.SpeechDetector)
    detector.recognizer = FakeRecognizer()
    detector.recognition_languages = ["ko-KR", "en-US"]

    text, request_error = detector._recognize_audio_with_fallback(audio="dummy", language=None)
    assert text == "stop now"
    assert request_error is False
    assert detector.recognizer.calls == ["ko-KR", "en-US"]


@manual_only
def test_manual_webcam_source_opens():
    source = create_video_source("webcam", camera_id=0)
    assert source.open() is True
    frame = source.capture_frame()
    assert frame is not None
    source.close()
