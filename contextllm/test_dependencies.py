#!/usr/bin/env python3
"""
의존성 테스트 스크립트
모든 requirements 패키지가 정상 설치되었는지 확인합니다.
"""

import sys
import subprocess

print("🧪 의존성 테스트 시작\n")
print("=" * 70)

# 테스트할 패키지들
packages = [
    ('torch', 'PyTorch', 'ML Framework'),
    ('numpy', 'NumPy', '수치 연산'),
    ('whisper', 'OpenAI Whisper', '음성 인식'),
    ('requests', 'Requests', 'HTTP 통신'),
    ('tiktoken', 'TikToken', 'LLM 토크나이저'),
    ('tqdm', 'TQDM', '진행 표시줄'),
    ('numba', 'Numba', 'JIT 컴파일러'),
    ('regex', 'Regex', '정규표현식'),
    ('soundfile', 'SoundFile', '음성 파일 처리'),
]

failed = []
success = []

for module_name, display_name, description in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'N/A')
        status = f"✅ {display_name:20} v{str(version):15} ({description})"
        print(status)
        success.append(module_name)
    except ImportError as e:
        status = f"⚠️  {display_name:20} {'설치 필요':15} ({description})"
        print(status)
        failed.append((module_name, str(e)))
    except Exception as e:
        status = f"❌ {display_name:20} {'오류':15} ({description})"
        print(status)
        failed.append((module_name, str(e)))

print("=" * 70)
print(f"\n📊 테스트 결과:")
print(f"  ✅ 성공: {len(success)}개")
print(f"  ⚠️  경고: {len(failed)}개")

if failed:
    print(f"\n🔧 추가 설치 필요한 패키지:")
    for module_name, error in failed:
        print(f"  - {module_name}")

# pip check 실행
print(f"\n🔍 pip check (의존성 충돌 확인):")
result = subprocess.run(['pip', 'check'], capture_output=True, text=True)

if result.returncode == 0:
    print("  ✅ 의존성 충돌 없음")
else:
    print(f"  ⚠️  {result.stdout}")

# 주요 기능 테스트
print(f"\n🚀 주요 기능 테스트:")

# Whisper 테스트
try:
    import whisper
    print("  ✅ Whisper 모듈 로드 성공")
except Exception as e:
    print(f"  ❌ Whisper 오류: {e}")

# NumPy 배열 생성 테스트
try:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f"  ✅ NumPy 배열 생성 성공: {arr}")
except Exception as e:
    print(f"  ❌ NumPy 오류: {e}")

# Requests 테스트
try:
    import requests
    print("  ✅ Requests 모듈 로드 성공")
except Exception as e:
    print(f"  ❌ Requests 오류: {e}")

print("\n" + "=" * 70)
print("✨ 테스트 완료!")
print("=" * 70)

sys.exit(0 if not failed else 1)
