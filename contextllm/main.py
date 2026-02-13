#!/usr/bin/env python3
"""
Context LLM - 멀티모달 상황 분석 시스템

사용법:
    python main.py --mode realtime          # 실시간 음성 + 영상 분석
    python main.py --mode testset           # 테스트셋 폴더 분석
    python main.py --mode file -f video.mp4 # 특정 파일 분석
    python main.py --mode webcam            # 웹캠만 (음성 없이)
    python main.py --mode network -u rtsp://...  # 네트워크 카메라
"""

import argparse
import sys
from pathlib import Path

import yaml

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv

# 설정 파일 경로
CONFIG_DIR = Path(__file__).parent / 'config'
CONFIG_PATH = CONFIG_DIR / 'config.yaml'
ENV_PATH = CONFIG_DIR / '.env'

# .env 파일 로드
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def load_config() -> dict:
    """config.yaml 파일 로드"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


# 전역 설정 로드
CONFIG = load_config()


def get_config(section: str, key: str, default=None):
    """설정값 가져오기 (nested key 지원)"""
    if section in CONFIG and isinstance(CONFIG[section], dict):
        return CONFIG[section].get(key, default)
    return CONFIG.get(section, default) if section == key else default


def create_system(args):
    """시스템 초기화"""
    from core.integrated_multimodal_system import (
        IntegratedMultimodalSystem,
        DownsamplingConfig
    )
    
    # 다운샘플링 설정 (CLI 인자 > config.yaml > 기본값)
    ds = CONFIG.get('downsampling', {})
    config = DownsamplingConfig(
        max_image_size=args.image_size or ds.get('max_image_size', 640),
        jpeg_quality=args.quality or ds.get('jpeg_quality', 75),
        video_fps=args.fps or ds.get('video_fps', 2.0),
        max_video_frames=args.max_frames or ds.get('max_video_frames', 10),
        video_capture_duration=args.duration or ds.get('video_capture_duration', 5.0)
    )
    
    # 비디오 설정
    video_cfg = CONFIG.get('video', {})
    camera_id = args.camera if args.camera is not None else video_cfg.get('camera_id', 0)
    
    # 모델 설정
    model = args.model or CONFIG.get('model', 'gpt-4o-mini')
    
    # 음성 인식 설정
    speech_cfg = CONFIG.get('speech', {})
    energy_threshold = args.energy_threshold or speech_cfg.get('energy_threshold', 400)
    dynamic_threshold = getattr(args, 'dynamic_threshold', speech_cfg.get('dynamic_threshold', False))
    
    system = IntegratedMultimodalSystem(
        camera_id=camera_id,
        model=model,
        downsampling_config=config,
        energy_threshold=energy_threshold,
        dynamic_threshold=dynamic_threshold
    )
    
    return system


def mode_realtime(args):
    """실시간 모드: 음성 감지 → 영상 캡처 → 분석"""
    system = create_system(args)
    
    # 반복 횟수 (CLI 인자 > config.yaml)
    analysis_cfg = CONFIG.get('analysis', {})
    logging_cfg = CONFIG.get('logging', {})
    
    iterations = args.iterations
    if iterations is None:
        iterations = analysis_cfg.get('iterations')
    
    # verbose 설정 (CLI 인자 > config.yaml)
    verbose = args.verbose if hasattr(args, 'verbose') and args.verbose else logging_cfg.get('verbose', False)
    
    system.start_monitoring(max_iterations=iterations, verbose=verbose)


def mode_testset(args):
    """테스트셋 모드: 음성 인식 후 테스트셋 이미지로 분석"""
    # 테스트셋 경로 (CLI 인자 > config.yaml > 기본값)
    video_cfg = CONFIG.get('video', {})
    logging_cfg = CONFIG.get('logging', {})
    testset_path = args.testset_path or video_cfg.get('testset_path', 'testsets')
    verbose = args.verbose if hasattr(args, 'verbose') and args.verbose else logging_cfg.get('verbose', False)
    
    if not Path(testset_path).exists():
        print(f"❌ 테스트셋 폴더를 찾을 수 없습니다: {testset_path}")
        return
    
    system = create_system(args)
    system.use_testset(testset_path)
    
    # 파일 목록 출력
    files = system.get_testset_files()
    if not files:
        print("❌ 테스트셋 폴더에 파일이 없습니다")
        return
    
    if verbose:
        print(f"\n📋 파일 목록 ({len(files)}개):")
        for i, f in enumerate(files):
            print(f"   {i}: {f}")
    
    # 파일 선택
    analysis_cfg = CONFIG.get('analysis', {})
    file_index = args.index
    if file_index is None:
        file_index = analysis_cfg.get('testset_index', 0)
    
    system.select_testset_file(file_index)
    print(f"📁 테스트 이미지: {files[file_index]}")
    
    # 반복 횟수
    iterations = args.iterations
    if iterations is None:
        iterations = analysis_cfg.get('iterations')
    
    # 실시간 모니터링 시작 (음성 인식 → 테스트 이미지 분석)
    system.start_monitoring(max_iterations=iterations, verbose=verbose)


def mode_file(args):
    """파일 모드: 음성 인식 후 지정 파일로 분석"""
    # 파일 경로 (CLI 인자 > config.yaml)
    video_cfg = CONFIG.get('video', {})
    logging_cfg = CONFIG.get('logging', {})
    file_path = args.file or video_cfg.get('file_path', '')
    verbose = args.verbose if hasattr(args, 'verbose') and args.verbose else logging_cfg.get('verbose', False)
    
    if not file_path:
        print("❌ 파일 경로를 지정하세요: -f <파일경로>")
        return
    
    if not Path(file_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    system = create_system(args)
    system.use_file(file_path)
    
    print(f"📁 파일: {file_path}")
    
    # 반복 횟수
    analysis_cfg = CONFIG.get('analysis', {})
    iterations = args.iterations
    if iterations is None:
        iterations = analysis_cfg.get('iterations')
    
    # 실시간 모니터링 시작 (음성 인식 → 파일 분석)
    system.start_monitoring(max_iterations=iterations, verbose=verbose)


def mode_webcam(args):
    """웹캠 모드: 음성 인식 후 웹캠 캡처 분석"""
    # 카메라 ID (CLI 인자 > config.yaml > 기본값)
    video_cfg = CONFIG.get('video', {})
    logging_cfg = CONFIG.get('logging', {})
    display_cfg = CONFIG.get('display', {})
    camera_id = args.camera if args.camera is not None else video_cfg.get('camera_id', 0)
    verbose = args.verbose if hasattr(args, 'verbose') and args.verbose else logging_cfg.get('verbose', False)
    
    system = create_system(args)
    system.use_webcam(camera_id)
    
    print(f"📷 웹캠: {camera_id}")
    
    # OpenCV 디스플레이 설정 (라이브 모드이므로 기본 활성화)
    use_opencv = not args.no_opencv if hasattr(args, 'no_opencv') else display_cfg.get('opencv_live', True)
    if use_opencv or args.opencv:
        system.enable_opencv_display(True)
    
    # 반복 횟수
    analysis_cfg = CONFIG.get('analysis', {})
    iterations = args.iterations
    if iterations is None:
        iterations = analysis_cfg.get('iterations')
    
    # 실시간 모니터링 시작 (음성 인식 → 웹캠 캡처 분석)
    system.start_monitoring(max_iterations=iterations, verbose=verbose)


def mode_network(args):
    """네트워크 카메라 모드: 음성 인식 후 네트워크 카메라 캡처 분석"""
    # URL (CLI 인자 > config.yaml)
    video_cfg = CONFIG.get('video', {})
    logging_cfg = CONFIG.get('logging', {})
    display_cfg = CONFIG.get('display', {})
    url = args.url or video_cfg.get('network_url', '')
    verbose = args.verbose if hasattr(args, 'verbose') and args.verbose else logging_cfg.get('verbose', False)
    
    if not url:
        print("❌ 카메라 URL을 지정하세요: -u <URL>")
        print("   예: rtsp://192.168.1.100:554/stream")
        print("   예: http://192.168.1.100:8080/video")
        return
    
    system = create_system(args)
    system.use_network_camera(url)
    
    print(f"🌐 네트워크 카메라: {url}")
    
    # OpenCV 디스플레이 설정 (라이브 모드이므로 기본 활성화)
    use_opencv = not args.no_opencv if hasattr(args, 'no_opencv') else display_cfg.get('opencv_live', True)
    if use_opencv or args.opencv:
        system.enable_opencv_display(True)
    
    # 반복 횟수
    analysis_cfg = CONFIG.get('analysis', {})
    iterations = args.iterations
    if iterations is None:
        iterations = analysis_cfg.get('iterations')
    
    # 실시간 모니터링 시작 (음성 인식 → 네트워크 캡처 분석)
    system.start_monitoring(max_iterations=iterations, verbose=verbose)


def print_result(result, verbose: bool = False):
    """분석 결과 출력 (간소화 버전)"""
    if not result.get("success"):
        print(f"❌ 실패: {result.get('error', '알 수 없는 오류')}")
        return
    
    analysis = result.get("multimodal_analysis", {})
    if not analysis:
        print("⚠️  분석 결과 없음")
        return
    
    # 핵심 정보만 출력
    situation = analysis.get('situation_type', 'N/A')
    urgency = analysis.get('urgency', 'N/A')
    is_emergency = analysis.get('is_emergency', False)
    consistency = analysis.get('audio_visual_consistency', 'N/A')
    
    icon = "🚨" if is_emergency else "✅"
    print(f"\n{icon} [{situation}] 긴급:{urgency} | 일치도:{consistency}")
    
    # verbose 모드일 때만 상세 출력
    if verbose:
        print("-" * 40)
        text = result.get("transcribed_text") or result.get("text_input", "")
        if text:
            print(f"📝 입력: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        voice = result.get("voice_characteristics")
        if voice:
            indicators = voice.get("emergency_indicators", {})
            score = indicators.get("overall_score", 0)
            print(f"🎤 음성 긴급도: {score:.0%}")
        
        print(f"🔍 상황: {analysis.get('situation', 'N/A')}")
        print(f"🚨 우선순위: {analysis.get('priority', 'N/A')}")
        
        if is_emergency:
            print(f"⚠️  긴급 사유: {analysis.get('emergency_reason', 'N/A')}")
        print("-" * 40)


def main():
    # config에서 기본값 가져오기
    video_cfg = CONFIG.get('video', {})
    ds_cfg = CONFIG.get('downsampling', {})
    analysis_cfg = CONFIG.get('analysis', {})
    
    parser = argparse.ArgumentParser(
        description='Context LLM - 멀티모달 상황 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --mode realtime              # 실시간 음성+영상 분석
  python main.py --mode realtime -n 5         # 5회 반복 후 종료
  python main.py --mode testset               # 테스트셋 전체 분석
  python main.py --mode testset --all         # 테스트셋 전체 분석
  python main.py --mode testset -i 0          # 테스트셋 첫 번째 파일
  python main.py --mode file -f video.mp4     # 특정 파일 분석
  python main.py --mode webcam                # 웹캠 영상만 분석
  python main.py --mode network -u rtsp://... # 네트워크 카메라
  
설정 파일: config/config.yaml (CLI 인자가 우선)
        """
    )
    
    # 기본 모드 (config.yaml에서 로드)
    default_mode = CONFIG.get('mode', 'realtime')
    parser.add_argument(
        '-m', '--mode',
        choices=['realtime', 'testset', 'file', 'webcam', 'network'],
        default=default_mode,
        help=f'실행 모드 (기본값: {default_mode})'
    )
    
    # 입력 소스 관련 (기본값 None으로 설정하여 config.yaml 값 사용 가능)
    parser.add_argument('-f', '--file', default=None, help='분석할 파일 경로')
    parser.add_argument('-u', '--url', default=None, help='네트워크 카메라 URL')
    parser.add_argument('-c', '--camera', type=int, default=None, help=f"웹캠 ID (기본값: {video_cfg.get('camera_id', 0)})")
    parser.add_argument('--testset-path', default=None, help=f"테스트셋 폴더 경로 (기본값: {video_cfg.get('testset_path', 'testsets')})")
    
    # 테스트셋 옵션
    parser.add_argument('-i', '--index', type=int, default=None, help='테스트셋에서 분석할 파일 인덱스')
    parser.add_argument('--all', action='store_true', help='테스트셋 전체 파일 분석')
    
    # 분석 옵션
    parser.add_argument('-t', '--text', default=None, help='분석에 사용할 텍스트 (음성 대신)')
    parser.add_argument('-n', '--iterations', type=int, default=None, help='반복 횟수 (realtime 모드)')
    parser.add_argument('--model', default=None, help=f"OpenAI 모델 (기본값: {CONFIG.get('model', 'gpt-4o-mini')})")
    parser.add_argument('-v', '--verbose', action='store_true', help='상세 출력 모드')
    
    # 음성 인식 옵션
    speech_cfg = CONFIG.get('speech', {})
    parser.add_argument('--energy-threshold', type=int, default=None, help='음성 감지 에너지 임계값 (낮을수록 민감, 기본값: 400)')
    parser.add_argument('--dynamic-threshold', action='store_true', help='동적 에너지 임계값 활성화 (실시간 마이크용)')
    parser.add_argument('--static-threshold', action='store_false', dest='dynamic_threshold', help='고정 에너지 임계값 (스피커 소리 인식용)')
    
    # 디스플레이 옵션
    display_cfg = CONFIG.get('display', {})
    parser.add_argument('--web', action='store_true', help='웹 대시보드 활성화 (localhost:5000)')
    parser.add_argument('--web-port', type=int, default=display_cfg.get('web_port', 5000), help='웹 대시보드 포트')
    parser.add_argument('--opencv', action='store_true', help='OpenCV 창 활성화 (라이브 모드만)')
    parser.add_argument('--no-opencv', action='store_true', help='OpenCV 창 비활성화')
    
    # 다운샘플링 설정 (기본값 None으로 설정하여 config.yaml 값 사용)
    parser.add_argument('--image-size', type=int, default=None, help=f"최대 이미지 크기 (기본값: {ds_cfg.get('max_image_size', 640)})")
    parser.add_argument('--quality', type=int, default=None, help=f"JPEG 품질 (기본값: {ds_cfg.get('jpeg_quality', 75)})")
    parser.add_argument('--fps', type=float, default=None, help=f"분석 FPS (기본값: {ds_cfg.get('video_fps', 2.0)})")
    parser.add_argument('--max-frames', type=int, default=None, help=f"최대 프레임 수 (기본값: {ds_cfg.get('max_video_frames', 10)})")
    parser.add_argument('--duration', type=float, default=None, help=f"캡처 시간 초 (기본값: {ds_cfg.get('video_capture_duration', 5.0)})")
    
    # 설정 파일 관련
    parser.add_argument('--config', default=None, help='사용할 설정 파일 경로')
    parser.add_argument('--show-config', action='store_true', help='현재 설정 출력')
    
    args = parser.parse_args()
    
    # 설정 출력
    if args.show_config:
        print("\n📋 현재 설정 (config/config.yaml):")
        print("-" * 40)
        import json
        print(yaml.dump(CONFIG, allow_unicode=True, default_flow_style=False))
        return
    
    # 웹 대시보드 시작
    web_dashboard = None
    if args.web:
        try:
            from web.app import start_dashboard, stop_dashboard
            start_dashboard(port=args.web_port)
            web_dashboard = True
        except ImportError as e:
            print(f"⚠️ 웹 대시보드를 시작할 수 없습니다: {e}")
            print("   flask와 flask-socketio를 설치하세요: pip install flask flask-socketio")
    
    # 모드별 실행
    try:
        if args.mode == 'realtime':
            mode_realtime(args)
        elif args.mode == 'testset':
            mode_testset(args)
        elif args.mode == 'file':
            mode_file(args)
        elif args.mode == 'webcam':
            mode_webcam(args)
        elif args.mode == 'network':
            mode_network(args)
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자가 중단했습니다")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise
    finally:
        # 웹 대시보드 정리
        if web_dashboard:
            try:
                from web.app import stop_dashboard
                stop_dashboard()
            except:
                pass


if __name__ == "__main__":
    main()
