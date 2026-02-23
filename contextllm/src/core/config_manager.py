#!/usr/bin/env python3
"""
설정 관리 모듈
config/config.yaml 파일을 로드하고 전역 설정을 제공합니다.

사용법:
    from core.config_manager import config, get_config
    
    # 전체 설정
    print(config)
    
    # 특정 섹션
    prompts = get_config('prompts')
    
    # 중첩된 설정
    pitch_threshold = get_config('voice_analysis', 'pitch', 'high_threshold')
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

# 프로젝트 루트 경로 찾기
def _find_project_root() -> Path:
    """프로젝트 루트 디렉토리 찾기"""
    current = Path(__file__).resolve()
    
    # src/core/config_manager.py -> src/core -> src -> project_root
    for parent in [current.parent, current.parent.parent, current.parent.parent.parent]:
        config_path = parent / 'config' / 'config.yaml'
        if config_path.exists():
            return parent
    
    # 현재 작업 디렉토리에서도 찾기
    cwd = Path.cwd()
    if (cwd / 'config' / 'config.yaml').exists():
        return cwd
    
    # 못 찾으면 현재 파일 기준 3단계 상위
    return current.parent.parent.parent


# 프로젝트 경로들
PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / 'config'
CONFIG_PATH = CONFIG_DIR / 'config.yaml'
ENV_PATH = CONFIG_DIR / '.env'
ACTIVE_CONFIG_PATH = CONFIG_PATH
ACTIVE_ENV_PATH = ENV_PATH


def load_config(config_path: Optional[Path] = None) -> dict:
    """config.yaml 파일 로드"""
    target = Path(config_path) if config_path else ACTIVE_CONFIG_PATH
    if target.exists():
        with open(target, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    print(f"⚠️  설정 파일을 찾을 수 없습니다: {target}")
    return {}


def get_config(*keys, default=None) -> Any:
    """
    중첩된 설정값 가져오기
    
    Args:
        *keys: 중첩된 키들 (예: 'voice_analysis', 'pitch', 'high_threshold')
        default: 기본값
    
    Returns:
        설정값 또는 기본값
    
    예시:
        get_config('model')  # 'gpt-4o-mini'
        get_config('voice_analysis', 'pitch', 'high_threshold')  # 250
        get_config('없는키', default='기본값')  # '기본값'
    """
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


def reload_config(config_path: Optional[Path] = None):
    """설정 다시 로드"""
    global config
    config = load_config(config_path)
    return config


# 전역 설정 객체
config = load_config()


def set_config_path(
    config_path: Path,
    *,
    auto_reload: bool = True,
    load_env_file: bool = True,
) -> dict:
    """
    런타임 설정 파일 경로를 교체.
    외부 통합 시스템에서 custom config를 사용할 때 호출.
    """
    global ACTIVE_CONFIG_PATH, ACTIVE_ENV_PATH

    ACTIVE_CONFIG_PATH = Path(config_path).resolve()
    ACTIVE_ENV_PATH = ACTIVE_CONFIG_PATH.parent / '.env'

    if load_env_file and ACTIVE_ENV_PATH.exists():
        load_dotenv(ACTIVE_ENV_PATH, override=False)

    if auto_reload:
        return reload_config(ACTIVE_CONFIG_PATH)
    return config


def set_runtime_config(
    runtime_config: Optional[dict] = None,
    *,
    config_path: Optional[Path] = None,
    merge: bool = False,
    load_env_file: bool = True,
) -> dict:
    """
    런타임 설정 주입.
    - config_path가 있으면 ACTIVE_CONFIG_PATH/ENV_PATH를 갱신
    - runtime_config가 있으면 전역 config를 직접 주입 (merge 선택 가능)
    """
    global config

    if config_path is not None:
        set_config_path(config_path, auto_reload=False, load_env_file=load_env_file)
    elif load_env_file and ACTIVE_ENV_PATH.exists():
        load_dotenv(ACTIVE_ENV_PATH, override=False)

    if runtime_config is None:
        config = load_config(ACTIVE_CONFIG_PATH)
        return config

    if merge:
        merged = dict(config)
        merged.update(runtime_config)
        config = merged
    else:
        config = dict(runtime_config)

    return config


def get_api_key(service: str = 'openai') -> Optional[str]:
    """
    API 키 가져오기 (우선순위: 환경변수 > .env > config.yaml)
    
    Args:
        service: 서비스 이름 ('openai', 'google' 등)
    
    Returns:
        API 키 문자열 또는 None
    """
    # .env 파일 로드
    if ACTIVE_ENV_PATH.exists():
        load_dotenv(ACTIVE_ENV_PATH, override=False)
    
    # 환경변수 이름 매핑
    env_var_names = {
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
    }
    
    env_var = env_var_names.get(service, f'{service.upper()}_API_KEY')
    
    # 1. 환경변수에서 확인
    api_key = os.getenv(env_var)
    if api_key:
        return api_key
    
    # 2. config.yaml에서 확인
    config_key = get_config('api_keys', service, default='')
    if config_key:
        return config_key
    
    return None


# 편의 함수들
def get_prompt(prompt_type: str = 'system') -> str:
    """프롬프트 가져오기"""
    return get_config('prompts', prompt_type, default='')


def get_emergency_keywords() -> list:
    """긴급 키워드 목록 가져오기"""
    return get_config('prompts', 'emergency_keywords', default=[])


def get_voice_threshold(category: str, name: str) -> float:
    """음성 분석 임계값 가져오기"""
    return get_config('voice_analysis', category, name, default=0.0)


def get_openai_config(key: str, default=None) -> Any:
    """OpenAI 설정 가져오기"""
    return get_config('openai', key, default=default)


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 설정 관리자 테스트")
    print("=" * 60)
    
    print(f"\n📁 프로젝트 루트: {PROJECT_ROOT}")
    print(f"📁 설정 파일: {ACTIVE_CONFIG_PATH}")
    print(f"📁 설정 파일 존재: {ACTIVE_CONFIG_PATH.exists()}")
    
    print(f"\n📋 전체 설정 키: {list(config.keys())}")
    
    print(f"\n🎯 모델: {get_config('model')}")
    print(f"🎯 모드: {get_config('mode')}")
    
    print(f"\n🎤 음성 피치 임계값: {get_voice_threshold('pitch', 'high_threshold')}")
    print(f"🎤 음성 에너지 정규화: {get_voice_threshold('energy', 'normalization_factor')}")
    
    print(f"\n🤖 OpenAI 토큰: {get_openai_config('max_tokens')}")
    print(f"🤖 OpenAI 온도: {get_openai_config('temperature')}")
    
    print(f"\n🚨 긴급 키워드: {get_emergency_keywords()[:5]}...")
    
    print("\n📝 시스템 프롬프트 (첫 100자):")
    print(f"   {get_prompt('system')[:100]}...")
