#!/usr/bin/env python3
"""Unit tests for multimodal analyzer normalization guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.multimodal_analyzer import MultimodalAnalyzer


def test_intrusion_guardrail_promotes_priority_for_forced_entry():
    analyzer = MultimodalAnalyzer.__new__(MultimodalAnalyzer)

    raw_result = {
        "context": "야간 CCTV",
        "situation": "남성이 망치로 유리창을 깨고 출입문을 부수며 침입 시도 중",
        "visual_content": "유리 파손과 강제 진입 장면",
        "situation_type": "UNKNOWN",
        "priority": "MEDIUM",
        "urgency": "MEDIUM",
        "is_emergency": False,
        "emergency_reason": None,
    }

    normalized = analyzer._normalize_analysis_result(raw_result)

    assert normalized["is_emergency"] is True
    assert normalized["priority"] == "CRITICAL"
    assert normalized["urgency"] == "CRITICAL"
    assert normalized["situation_type"] == "SECURITY"
    assert normalized["emergency_reason"] is not None


def test_non_intrusion_result_keeps_low_priority():
    analyzer = MultimodalAnalyzer.__new__(MultimodalAnalyzer)

    raw_result = {
        "context": "일반 통행",
        "situation": "복도에서 사람들이 이동 중",
        "visual_content": "특이사항 없음",
        "situation_type": "NORMAL",
        "priority": "LOW",
        "urgency": "LOW",
        "is_emergency": False,
        "emergency_reason": None,
    }

    normalized = analyzer._normalize_analysis_result(raw_result)

    assert normalized["is_emergency"] is False
    assert normalized["priority"] == "LOW"
    assert normalized["urgency"] == "LOW"
    assert normalized["emergency_reason"] is None


def test_fire_guardrail_promotes_to_critical():
    analyzer = MultimodalAnalyzer.__new__(MultimodalAnalyzer)

    raw_result = {
        "context": "창고 내부",
        "situation": "연기가 빠르게 확산되고 불길이 보임",
        "visual_content": "화재 장면",
        "situation_type": "UNKNOWN",
        "priority": "MEDIUM",
        "urgency": "MEDIUM",
        "is_emergency": False,
    }

    normalized = analyzer._normalize_analysis_result(raw_result)
    assert normalized["priority"] == "CRITICAL"
    assert normalized["urgency"] == "CRITICAL"
    assert normalized["is_emergency"] is True
    assert normalized["situation_type"] == "FIRE"


def test_medical_guardrail_promotes_to_critical():
    analyzer = MultimodalAnalyzer.__new__(MultimodalAnalyzer)

    raw_result = {
        "context": "사무실 CCTV",
        "situation": "사람이 갑자기 쓰러진 뒤 의식 없음으로 보임",
        "visual_content": "주변에서 도움 요청",
        "situation_type": "UNKNOWN",
        "priority": "LOW",
        "urgency": "LOW",
        "is_emergency": False,
    }

    normalized = analyzer._normalize_analysis_result(raw_result)
    assert normalized["priority"] == "CRITICAL"
    assert normalized["urgency"] == "CRITICAL"
    assert normalized["is_emergency"] is True
    assert normalized["situation_type"] == "MEDICAL"


def test_weapon_violence_guardrail_promotes_to_critical():
    analyzer = MultimodalAnalyzer.__new__(MultimodalAnalyzer)

    raw_result = {
        "context": "주차장",
        "situation": "흉기를 든 인물이 다른 사람을 위협하며 폭행",
        "visual_content": "격한 몸싸움",
        "situation_type": "SECURITY",
        "priority": "MEDIUM",
        "urgency": "MEDIUM",
        "is_emergency": False,
    }

    normalized = analyzer._normalize_analysis_result(raw_result)
    assert normalized["priority"] == "CRITICAL"
    assert normalized["urgency"] == "CRITICAL"
    assert normalized["is_emergency"] is True
