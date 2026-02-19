#!/usr/bin/env python3
"""Automated tests for the integrated multimodal system.

Interactive/hardware tests are marked as manual and skipped by default.
Set RUN_MANUAL_TESTS=1 to enable them.
"""

from __future__ import annotations

import os
import sys
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


@manual_only
def test_manual_webcam_source_opens():
    source = create_video_source("webcam", camera_id=0)
    assert source.open() is True
    frame = source.capture_frame()
    assert frame is not None
    source.close()
