"""
원본 모듈 경로 상수 및 import 유틸리티

각 래퍼 모듈은 이 상수를 사용하여 원본 모듈 폴더에서 직접 import합니다.
원본 폴더를 수정하면 통합 시스템에 즉시 반영됩니다.

프로젝트 구조:
    KNU-MM/
    ├── Detaction_CCTV/           → YOLO, PTZ(ONVIF), 스트림, 우선순위, ReID
    ├── mic_array_Control/        → ReSpeaker 마이크 어레이 DOA
    ├── PTZcamera_Control/        → PTZ 카메라 (ONVIF 테스트)
    ├── contextllm/               → 멀티모달 LLM 분석
    ├── 서버전송예시.py            → 서버 전송 유틸리티
    └── integrated_system/        → 이 파일이 속한 통합 시스템 (래퍼 전용)
"""

import sys
import importlib
import importlib.util
from pathlib import Path

# ── 프로젝트 루트 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── 원본 모듈 폴더 경로 ──
DETECT_DIR      = str(PROJECT_ROOT / "Detaction_CCTV")
MIC_DIR         = str(PROJECT_ROOT / "mic_array_Control")
PTZ_DIR         = str(PROJECT_ROOT / "PTZcamera_Control")
CONTEXTLLM_DIR  = str(PROJECT_ROOT / "contextllm")
CONTEXTLLM_SRC  = str(PROJECT_ROOT / "contextllm" / "src")
CONTEXTLLM_CORE = str(PROJECT_ROOT / "contextllm" / "src" / "core")


def ensure_path(path: str) -> None:
    """sys.path에 경로 추가 (중복 방지)"""
    if path not in sys.path:
        sys.path.insert(0, path)


def import_from_file(module_name: str, file_path: str):
    """
    파일 경로에서 모듈을 직접 로드 (패키지 이름 충돌 회피)
    
    이미 로드된 모듈은 캐시에서 반환합니다.
    contextllm처럼 패키지명 충돌이 있는 경우에 사용합니다.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod
