#!/usr/bin/env python3
"""ContextLLM CLI entrypoint (config-first)."""

import argparse
import sys
from pathlib import Path

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))


CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
VALID_MODES = {"realtime", "testset", "file", "webcam", "network"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ContextLLM - config-first 멀티모달 상황 분석",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="설정 파일 경로 (기본: config/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=None,
        help="실행 모드 override (설정 파일 mode보다 우선)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="현재 로드된 설정 출력 후 종료",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from app.runner import ContextLLMRunner
        runner = ContextLLMRunner(Path(args.config))
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    if args.show_config:
        runner.show_config()
        return 0

    try:
        runner.run(mode_override=args.mode)
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자가 중단했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
