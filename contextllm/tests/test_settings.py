#!/usr/bin/env python3
"""Unit tests for config-first settings parsing."""

from __future__ import annotations

import sys
from pathlib import Path

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.settings import build_settings


def test_invalid_mode_falls_back_to_realtime():
    settings = build_settings({"mode": "INVALID_MODE"})
    assert settings.mode == "realtime"


def test_media_test_values_are_parsed_from_strings():
    raw = {
        "mode": "testset",
        "media_test": {
            "enabled": "true",
            "image_path": "testsets/sample.png",
            "video_path": "",
            "audio_path": "testsets/sample.mp3",
            "text_input": "보조 텍스트",
            "phrase_time_limit": "7.5",
        },
        "speech": {
            "enabled": "false",
            "dynamic_threshold": "on",
        },
    }

    settings = build_settings(raw)

    assert settings.mode == "testset"
    assert settings.media_test.enabled is True
    assert settings.media_test.image_path == "testsets/sample.png"
    assert settings.media_test.audio_path == "testsets/sample.mp3"
    assert settings.media_test.phrase_time_limit == 7.5
    assert settings.speech.enabled is False
    assert settings.speech.dynamic_threshold is True


def test_iterations_none_and_invalid_value_handling():
    settings_none = build_settings({"analysis": {"iterations": None}})
    assert settings_none.analysis.iterations is None

    settings_valid = build_settings({"analysis": {"iterations": "3"}})
    assert settings_valid.analysis.iterations == 3

    settings_invalid = build_settings({"analysis": {"iterations": "not_a_number"}})
    assert settings_invalid.analysis.iterations == 0
