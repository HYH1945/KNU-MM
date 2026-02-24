"""
KNU-MM 통합 멀티모달 관제 시스템 - 메인 진입점

모든 모듈(YOLO, MicArray, STT, ContextLLM, ServerReporter)을 하나의
파이프라인으로 통합 실행합니다.

데이터 흐름:
    ┌─── 영상 ───┐     ┌─── 음성 ───┐
    │ CCTV/웹캠  │     │ MicArray   │
    └─────┬──────┘     └────┬───────┘
          │                  │
     YOLO 탐지          DOA 방향 감지
          │              STT 텍스트 변환
          │                  │
          └───── ContextLLM ─┘
                (통합 분석)
                     │
              OpenCV 통합 화면

사용법:
    cd integrated_system
    python main.py                          # 기본 실행
    python main.py --no-mic                 # 마이크 어레이 없이
    python main.py --no-stt                 # 음성 인식 없이
    python main.py --no-llm                 # LLM 분석 없이
    python main.py --no-display             # 화면 표시 없이
    python main.py --config custom.yaml     # 커스텀 설정
"""

import os
import sys
import cv2
import time
import math
import signal
import logging
import argparse
import subprocess
import socket
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrated_system_process.core.event_bus import EventBus, Event
from integrated_system_process.core.orchestrator import Orchestrator
from integrated_system_process.modules.stream_manager import SharedStreamManager
from integrated_system_process.modules.ptz_controller import UnifiedPTZController
from integrated_system_process.modules.yolo_detection import YOLODetectionModule
from integrated_system_process.modules.mic_array import MicArrayModule
from integrated_system_process.modules.stt_module import STTModule
from integrated_system_process.modules.context_llm import ContextLLMModule
from integrated_system_process.modules.server_reporter import ServerReporterModule


def _get_env_str(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def _apply_camera_env_overrides(config: dict) -> None:
    """camera 설정을 .env로 오버라이드한다. (.env 값 우선)"""
    cam = config.setdefault("camera", {})

    rtsp_url = _get_env_str("CAMERA_RTSP_URL", "RTSP_URL")
    if rtsp_url:
        cam["rtsp_url"] = rtsp_url

    test_video = _get_env_str("CAMERA_TEST_VIDEO", "TEST_VIDEO")
    if test_video:
        cam["test_video"] = test_video

    ip = _get_env_str("CAMERA_IP")
    if ip:
        cam["ip"] = ip

    port = _get_env_str("CAMERA_PORT")
    if port:
        try:
            cam["port"] = int(port)
        except ValueError:
            pass

    user = _get_env_str("CAMERA_USER")
    if user:
        cam["user"] = user

    password = _get_env_str("CAMERA_PASSWORD")
    if password:
        cam["password"] = password


def load_config(config_path: str) -> dict:
    """설정 파일 로드 (YAML + 환경변수 오버라이드)"""
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    # .env 로드
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    contextllm_env = PROJECT_ROOT / "contextllm" / "config" / ".env"
    if contextllm_env.exists():
        load_dotenv(contextllm_env)

    # camera는 .env 값이 있으면 우선 적용
    _apply_camera_env_overrides(config)

    return config


def setup_logging(config: dict) -> None:
    """로깅 설정"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    log_file = log_cfg.get("file", "")
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)-18s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def _to_bool(value, default: bool = False) -> bool:
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


def _to_int(value, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    if port <= 0:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, int(port))) == 0


def _resolve_script_path(script_value: str) -> Path:
    path = Path(script_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _start_dashboard_process(
    script_path: Path,
    label: str,
    logger: logging.Logger,
    env: dict | None = None,
):
    if not script_path.exists():
        logger.warning("[%s] 자동 시작 실패: 스크립트 없음 (%s)", label, script_path)
        return None

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env or os.environ.copy(),
        )
        time.sleep(0.5)
        if proc.poll() is not None:
            logger.warning(
                "[%s] 자동 시작 실패: 프로세스 즉시 종료 (코드=%s)",
                label,
                proc.returncode,
            )
            return None
        logger.info("[%s] 자동 시작됨 (pid=%s, script=%s)", label, proc.pid, script_path)
        return proc
    except Exception as exc:
        logger.warning("[%s] 자동 시작 실패: %s", label, exc)
        return None


def _sync_server_url_with_event_port(config: dict, event_port: int, logger: logging.Logger) -> None:
    if event_port <= 0:
        return
    server_cfg = config.setdefault("server", {})
    current_url = str(server_cfg.get("url", "") or "").strip()
    if not current_url:
        server_cfg["url"] = f"http://localhost:{event_port}/event"
        return

    parsed = urlparse(current_url)
    if parsed.scheme not in {"http", "https"}:
        return
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        return
    if parsed.path not in {"", "/", "/event"}:
        return

    needs_update = (parsed.port or 80) != int(event_port)
    if not needs_update:
        return
    new_netloc = f"{parsed.hostname}:{event_port}"
    updated = parsed._replace(netloc=new_netloc, path="/event")
    new_url = urlunparse(updated)
    server_cfg["url"] = new_url
    logger.info("[EventDashboard] server.url 자동 동기화: %s", new_url)


def _start_dashboards_if_enabled(config: dict, logger: logging.Logger):
    server_cfg = config.get("server", {}) or {}
    procs = []
    started_scripts = set()

    event_port = _to_int(server_cfg.get("event_dashboard_port"), 8100)
    contextllm_port = _to_int(server_cfg.get("contextllm_dashboard_port"), 5100)

    event_dashboard_enabled = (
        _to_bool(server_cfg.get("auto_start_dashboard", False), False)
        or _to_bool(server_cfg.get("auto_start_event_dashboard", False), False)
    )
    if event_dashboard_enabled:
        _sync_server_url_with_event_port(config, event_port, logger)
        script_path = _resolve_script_path(
            str(server_cfg.get("dashboard_script", "integrated_system/modules/dashboard_server.py"))
        )
        key = str(script_path)
        if key not in started_scripts:
            if _is_port_in_use(event_port):
                logger.warning(
                    "[EventDashboard] 포트 %s 이미 사용 중입니다. 자동 시작을 건너뜁니다.",
                    event_port,
                )
            else:
                env = os.environ.copy()
                env["EVENT_DASHBOARD_PORT"] = str(event_port)
                proc = _start_dashboard_process(script_path, "EventDashboard", logger, env=env)
                if proc is not None:
                    procs.append(("EventDashboard", proc))
                    started_scripts.add(key)

    contextllm_dashboard_enabled = _to_bool(server_cfg.get("auto_start_contextllm_dashboard", False), False)
    if contextllm_dashboard_enabled:
        script_path = _resolve_script_path(
            str(server_cfg.get("contextllm_dashboard_script", "contextllm/src/web/app.py"))
        )
        key = str(script_path)
        if key in started_scripts:
            logger.info("[ContextLLMDashboard] 동일 스크립트는 중복 시작하지 않습니다: %s", script_path)
        else:
            if _is_port_in_use(contextllm_port):
                logger.warning(
                    "[ContextLLMDashboard] 포트 %s 이미 사용 중입니다. 자동 시작을 건너뜁니다.",
                    contextllm_port,
                )
            else:
                env = os.environ.copy()
                env["CONTEXTLLM_DASHBOARD_PORT"] = str(contextllm_port)
                proc = _start_dashboard_process(script_path, "ContextLLMDashboard", logger, env=env)
                if proc is not None:
                    procs.append(("ContextLLMDashboard", proc))
                    started_scripts.add(key)

    return procs


def _stop_dashboards(procs, logger: logging.Logger) -> None:
    for label, proc in reversed(procs or []):
        if proc is None or proc.poll() is not None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=3)
            logger.info("[%s] 자동 시작 프로세스 종료 완료", label)
        except Exception:
            try:
                proc.kill()
                logger.info("[%s] 자동 시작 프로세스 강제 종료", label)
            except Exception:
                pass


def build_system(config: dict, args) -> tuple:
    """
    시스템 빌드 - 모든 모듈을 생성하고 오케스트레이터에 등록
    
    Returns:
        (orchestrator, event_bus, stream_manager, ptz, mic_module, stt_module)
    """
    # 1. 이벤트 버스 생성
    event_bus = EventBus(max_workers=4, async_mode=True)

    # 2. 공유 리소스 생성
    cam_cfg = config.get("camera", {})
    
    # 테스트 영상 우선 확인
    test_video_path = cam_cfg.get("test_video", "")
    if test_video_path:
        # 상대경로면 프로젝트 루트 기준으로 변환
        if not os.path.isabs(test_video_path):
            test_video_path = str(PROJECT_ROOT / test_video_path)
        if os.path.exists(test_video_path):
            rtsp_url = test_video_path
            logging.info(f"🎬 테스트 영상 모드: {test_video_path}")
        else:
            logging.warning(f"⚠️  테스트 영상 파일 없음: {test_video_path} → 웹캠으로 폴백")
            rtsp_url = cam_cfg.get("rtsp_url", 0)
    else:
        rtsp_url = cam_cfg.get("rtsp_url", 0)
    
    # 숫자 문자열을 int로 변환 (0=웹캠, 1=두번째 카메라 등)
    if isinstance(rtsp_url, str) and rtsp_url.isdigit():
        rtsp_url = int(rtsp_url)
    
    # 테스트 모드: rtsp_url이 None이거나 -1이면 테스트 비디오 사용
    if rtsp_url is None or rtsp_url == -1:
        test_video = Path(__file__).parent / "test_video.mp4"
        if test_video.exists():
            rtsp_url = str(test_video)
            logging.warning(f"⚠️  테스트 모드: {test_video.name} 사용")
        else:
            logging.warning("⚠️  테스트 모드: 더미 스트림 사용 (영상 없음)")
            # 더미 URL - stream이 None 반환할 것
            rtsp_url = "test://dummy"
    
    stream = SharedStreamManager(rtsp_url)
    stream.start()

    # PTZ 컨트롤러
    ptz_cfg = config.get("ptz", {})
    ptz = UnifiedPTZController({
        "camera_ip": cam_cfg.get("ip", ""),
        "camera_port": cam_cfg.get("port", 80),
        "camera_user": cam_cfg.get("user", ""),
        "camera_password": cam_cfg.get("password", ""),
        "control_mode": ptz_cfg.get("control_mode", "onvif"),
    })
    ptz.initialize()

    # 3. 오케스트레이터 생성
    orch = Orchestrator(event_bus)

    # 4. 모듈 등록
    # --- YOLO ---
    yolo_cfg = config.get("yolo", {})
    if yolo_cfg.get("enabled", True) and not args.no_yolo:
        yolo_module = YOLODetectionModule(
            event_bus=event_bus,
            ptz=ptz,
            model_path=yolo_cfg.get("model_path", "yolov8n.pt"),
            confidence=yolo_cfg.get("confidence", 0.3),
            pid_kp=ptz_cfg.get("pid_kp", 0.4),
            dead_zone=ptz_cfg.get("dead_zone_pixels", 50),
            patrol_speed=ptz_cfg.get("patrol_speed", 0.2),
            target_classes=yolo_cfg.get("target_classes"),
        )
        orch.register(yolo_module)

    # --- MicArray ---
    mic_cfg = config.get("mic_array", {})
    mic_module = None
    if mic_cfg.get("enabled", True) and not args.no_mic:
        mic_module = MicArrayModule(
            event_bus=event_bus,
            ptz=ptz,
            agc_max_gain=mic_cfg.get("agc_max_gain", 15.0),
            vad_threshold=mic_cfg.get("vad_threshold", 10.0),
            confidence_threshold=mic_cfg.get("confidence_threshold", 0.6),
            zenith_confidence=mic_cfg.get("zenith_confidence", 0.4),
            zenith_gain=mic_cfg.get("zenith_gain", 10.0),
            history_size=mic_cfg.get("history_size", 10),
        )
        if orch.register(mic_module):
            mic_module.start_monitoring()

    # --- STT (Speech-to-Text) ---
    stt_cfg = config.get("stt", {})
    stt_module = None
    if stt_cfg.get("enabled", True) and not args.no_stt:
        stt_module = STTModule(
            event_bus=event_bus,
            language=stt_cfg.get("language", "ko-KR"),
            energy_threshold=stt_cfg.get("energy_threshold", 400),
            pause_threshold=stt_cfg.get("pause_threshold", 3.0),
            phrase_time_limit=stt_cfg.get("phrase_time_limit", 15.0),
            dynamic_threshold=stt_cfg.get("dynamic_threshold", True),
        )
        if orch.register(stt_module):
            stt_module.start_listening()

    # --- ContextLLM ---
    llm_cfg = config.get("context_llm", {})
    sound_cfg = config.get("sound_event", {})
    srv_cfg = config.get("server", {})
    contextllm_dashboard_port = _to_int(srv_cfg.get("contextllm_dashboard_port"), 5100)
    contextllm_dashboard_url = str(srv_cfg.get("contextllm_dashboard_url", "") or "").strip()
    if not contextllm_dashboard_url:
        contextllm_dashboard_url = f"http://127.0.0.1:{contextllm_dashboard_port}/api/push_result"
    if llm_cfg.get("enabled", True) and not args.no_llm:
        llm_module = ContextLLMModule(
            event_bus=event_bus,
            model=llm_cfg.get("model", "gpt-4o-mini"),
            config_path=llm_cfg.get("config_path", "") or None,
            ptz=ptz,
            ptz_settle_seconds=llm_cfg.get("ptz_settle_seconds", 0.6),
            max_doa_age_seconds=llm_cfg.get("max_doa_age_seconds", 5.0),
            analysis_cooldown=llm_cfg.get("analysis_cooldown", 5.0),
            non_speech_enabled=sound_cfg.get("enabled", True),
            non_speech_min_duration=sound_cfg.get("min_audio_duration", 0.2),
            non_speech_cooldown=sound_cfg.get("cooldown_seconds", 1.5),
            non_speech_analyze_on_detected=sound_cfg.get("analyze_on_detected", False),
            doa_object_mapping_enabled=llm_cfg.get("doa_object_mapping_enabled", True),
            camera_hfov_deg=llm_cfg.get("camera_hfov_deg", 60.0),
            doa_object_match_threshold_deg=llm_cfg.get("doa_object_match_threshold_deg", 12.0),
            contextllm_dashboard_push_enabled=_to_bool(srv_cfg.get("push_contextllm_dashboard", True), True),
            contextllm_dashboard_url=contextllm_dashboard_url,
            contextllm_dashboard_timeout=_to_float(
                srv_cfg.get("contextllm_dashboard_timeout", 1.2),
                1.2,
            ),
        )
        orch.register(llm_module)

    # --- ServerReporter ---
    if srv_cfg.get("enabled", False):
        server_module = ServerReporterModule(
            event_bus=event_bus,
            server_url=srv_cfg.get("url", ""),
            timeout=srv_cfg.get("timeout", 2.0),
        )
        orch.register(server_module)

    # 5. 파이프라인 정의
    # context_llm은 조건 없이 항상 실행.
    # 내부에서 STT 텍스트/비음성 이벤트 트리거 유무를 판단해 분석 여부를 결정한다.
    orch.define_pipeline("security", [
        {"module": "yolo"},
        {"module": "context_llm"},
        {"module": "server_reporter"},
    ])

    orch.define_pipeline("full_analysis", [
        {"module": "yolo"},
        {"module": "context_llm"},
        {"module": "server_reporter"},
    ])

    return orch, event_bus, stream, ptz, mic_module, stt_module


def run_main_loop(orch: Orchestrator, stream: SharedStreamManager, config: dict, args, stt_module=None) -> None:
    """
    메인 루프 - 프레임 수신 → 파이프라인 실행 → 통합 시각화

    화면 레이아웃:
    ┌──────────────────────────────────────────────┐
    │  FPS | Mode | Pipeline | 모듈 상태           │  상단 바
    ├──────────────────────────────────────────────┤
    │                                              │
    │          영상 프레임 (YOLO 박스)              │
    │                                              │
    │  ┌─DOA──┐                                    │
    │  │  ◀   │  (마이크 방향 표시)                │
    │  └──────┘                                    │
    ├──────────────────────────────────────────────┤
    │  🎤 STT: "인식된 텍스트"                     │  하단 패널
    │  🤖 LLM: [긴급도] 상황 설명                  │
    └──────────────────────────────────────────────┘
    """
    display_cfg = config.get("display", {})
    pipeline_cfg = config.get("pipeline", {})
    
    show_display = display_cfg.get("show_opencv", True) and not args.no_display
    window_name = display_cfg.get("window_name", "KNU-MM Integrated System")
    process_every = pipeline_cfg.get("process_every_n_frames", 3)
    pipeline_name = pipeline_cfg.get("default", "security")

    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    fps_frames = 0
    fps_time = time.perf_counter()

    # 표시용 상태 변수
    display_stt_text = ""
    display_stt_time = 0.0
    display_llm = {}
    display_doa_angle = -1
    display_yolo_objects = []   # YOLO 결과를 프레임 간 유지
    display_yolo_mode = "N/A"   # YOLO 모드를 프레임 간 유지

    logger = logging.getLogger("MainLoop")
    logger.info(f"━━━ 메인 루프 시작 (파이프라인: {pipeline_name}, 매 {process_every}프레임) ━━━")

    # 모듈 상태 출력
    for name, status in orch.list_modules().items():
        icon = "✅" if status["initialized"] else "⛔"
        logger.info(f"  {icon} {name}: {'활성' if status['initialized'] else '비활성'}")

    # STT 텍스트 수신용 이벤트 핸들러
    def on_stt_display(event: Event):
        nonlocal display_stt_text, display_stt_time
        display_stt_text = event.data.get("text", "")
        display_stt_time = time.time()

    def on_doa_display(event: Event):
        nonlocal display_doa_angle
        display_doa_angle = event.data.get("sector_angle", -1)

    event_bus = orch.event_bus
    event_bus.subscribe("stt.text_recognized", on_stt_display)
    event_bus.subscribe("mic.doa_detected", on_doa_display)

    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_frames += 1

            # FPS 계산
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                fps = fps_frames / (now - fps_time)
                fps_frames = 0
                fps_time = now

            # N프레임마다 파이프라인 실행
            results = {}
            if frame_count % process_every == 0:
                results = orch.run_pipeline(pipeline_name, {
                    "frame": frame,
                    "stream": stream,
                    "timestamp": time.time(),
                    "frame_count": frame_count,
                })

            # YOLO 결과 갱신 (프레임 간 유지)
            yolo_result = results.get("yolo", {})
            if "objects" in yolo_result:
                display_yolo_objects = yolo_result["objects"]
                display_yolo_mode = yolo_result.get("mode", "N/A")

            # LLM 결과 갱신
            llm_result = results.get("context_llm", {})
            if llm_result.get("analyzed"):
                display_llm = llm_result
            else:
                llm_mod = orch.get_module("context_llm")
                if llm_mod and hasattr(llm_mod, "get_display_result"):
                    latest_display = llm_mod.get_display_result()
                    if latest_display:
                        display_llm = latest_display

            # ─── 통합 시각화 ───
            if show_display:
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]

                # ── 1. YOLO 박스 그리기 (프레임 간 유지된 결과 사용) ──
                yolo_mod = orch.get_module("yolo")
                if yolo_mod and display_yolo_objects:
                    display_frame = yolo_mod.get_annotated_frame(display_frame, display_yolo_objects)

                # ── 2. 상단 정보 바 (반투명 검정) ──
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                # FPS
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # YOLO 모드 (프레임 간 유지된 값 사용)
                cv2.putText(display_frame, f"Mode: {display_yolo_mode}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                # 파이프라인 표시
                cv2.putText(display_frame, f"Pipeline: {pipeline_name}", (180, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

                # 모듈 상태 인디케이터 (우상단)
                module_status_x = w - 280
                statuses = [
                    ("YOLO", orch.get_module("yolo")),
                    ("MIC", orch.get_module("mic_array")),
                    ("STT", orch.get_module("stt")),
                    ("LLM", orch.get_module("context_llm")),
                ]
                for i, (label, mod) in enumerate(statuses):
                    sx = module_status_x + i * 70
                    is_active = mod and mod.is_ready
                    color = (0, 255, 0) if is_active else (80, 80, 80)
                    cv2.putText(display_frame, label, (sx, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    # 상태 점
                    if label == "LLM" and display_llm.get("analyzed") and display_llm.get("situation_type") == "Thinking...":
                        # LLM이 처리 중이면 점 깜빡임 (노란색)
                        if int(time.time() * 4) % 2 == 0:
                            color = (0, 255, 255)
                    cv2.circle(display_frame, (sx + 15, 40), 5, color, -1)

                # ── 3. DOA 방향 표시 (좌하단 미니 컴퍼스) ──
                mic_mod = orch.get_module("mic_array")
                if mic_mod and mic_mod.is_ready and display_doa_angle >= 0:
                    _draw_doa_compass(display_frame, display_doa_angle, x=60, y=h - 80, radius=40)
                elif display_doa_angle >= 0:
                    # DOA 텍스트만 표시 (마이크 모듈 없어도 이벤트 수신 가능)
                    cv2.putText(display_frame, f"DOA: {display_doa_angle} deg",
                                (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                # ── 4. 하단 패널 (STT + LLM 결과) ──
                panel_h = 100
                panel_y = h - panel_h

                # 반투명 하단 배경
                overlay2 = display_frame.copy()
                cv2.rectangle(overlay2, (0, panel_y), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.75, display_frame, 0.25, 0, display_frame)

                # STT 텍스트 표시 (10초간 유지)
                stt_text_display = display_stt_text if (time.time() - display_stt_time < 10) else ""
                if stt_text_display:
                    # 마이크 아이콘 + 텍스트
                    cv2.putText(display_frame, "[MIC]", (10, panel_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

                    # 텍스트가 길면 잘라서 표시
                    max_chars = w // 12
                    display_text = stt_text_display[:max_chars]
                    if len(stt_text_display) > max_chars:
                        display_text += "..."
                    cv2.putText(display_frame, display_text, (70, panel_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "[MIC] (waiting...)", (10, panel_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

                # LLM 분석 결과 표시
                if display_llm.get("analyzed"):
                    priority = display_llm.get("priority", "LOW")
                    is_emergency = display_llm.get("is_emergency", False)
                    urgency = display_llm.get("urgency", "")
                    situation = display_llm.get("situation_type", "")
                    speech_text = display_llm.get("speech_text", "")

                    # 상태별 색상 및 라벨 설정
                    if priority == "PROCESSING" or situation == "Thinking...":
                        llm_color = (0, 255, 255)  # 노랑 (처리 중)
                        priority_icon = "[PROCESSING]"
                        idx = int(time.time() * 3) % 4
                        urgency = "Thinking" + "." * idx
                        situation = "Waiting for LLM response..."
                    elif is_emergency:
                        llm_color = (0, 0, 255)   # 빨강
                        priority_icon = "[EMERGENCY]"
                    elif priority in ("CRITICAL", "HIGH"):
                        llm_color = (0, 128, 255)  # 주황
                        priority_icon = f"[{priority}]"
                    elif priority == "MEDIUM":
                        llm_color = (0, 255, 255)  # 노랑
                        priority_icon = f"[{priority}]"
                    else:
                        llm_color = (0, 255, 0)    # 초록
                        priority_icon = f"[{priority}]"

                    cv2.putText(display_frame, "[LLM]", (10, panel_y + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1)
                    cv2.putText(display_frame, f"{priority_icon} {urgency}", (70, panel_y + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, llm_color, 2)

                    # 상황 요약 (2번째 줄)
                    if situation:
                        cv2.putText(display_frame, f"Situation: {situation}", (70, panel_y + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                    # 긴급 시 화면 테두리 깜빡임
                    if is_emergency and int(time.time() * 2) % 2 == 0:
                        cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
                else:
                    cv2.putText(display_frame, "[LLM] (no analysis yet)", (10, panel_y + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

                # 구분선
                cv2.line(display_frame, (0, panel_y), (w, panel_y), (80, 80, 80), 1)
                cv2.line(display_frame, (0, 80), (w, 80), (80, 80, 80), 1)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    # 파이프라인 전환
                    pipeline_name = "full_analysis" if pipeline_name == "security" else "security"
                    logger.info(f"파이프라인 전환 → {pipeline_name}")

    except KeyboardInterrupt:
        logger.info("\n사용자 중단 (Ctrl+C)")


def _draw_doa_compass(frame, angle_deg: float, x: int, y: int, radius: int = 40):
    """
    DOA 방향 미니 컴퍼스 그리기

    Args:
        frame: OpenCV 프레임
        angle_deg: DOA 각도 (0=전방, 시계방향)
        x, y: 컴퍼스 중심 좌표
        radius: 컴퍼스 반지름
    """
    # 배경 원 (반투명)
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), radius + 5, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 외곽 원
    cv2.circle(frame, (x, y), radius, (100, 100, 100), 2)

    # 방위 표시
    cv2.putText(frame, "N", (x - 5, y - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 2)

    # 방향 화살표 (angle_deg → 라디안, OpenCV 좌표계)
    # DOA 0° = 전방(위), 시계방향 증가
    rad = math.radians(angle_deg - 90)  # OpenCV: 0°=오른쪽이므로 -90
    end_x = int(x + radius * 0.8 * math.cos(rad))
    end_y = int(y + radius * 0.8 * math.sin(rad))

    cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 200, 255), 2, tipLength=0.3)

    # 각도 텍스트
    cv2.putText(frame, f"{int(angle_deg)}", (x - 12, y + radius + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)


def main():
    parser = argparse.ArgumentParser(
        description='KNU-MM 통합 멀티모달 관제 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                      # 전체 모듈 실행
  python main.py --no-mic             # 마이크 없이 (YOLO + STT + LLM)
  python main.py --no-stt             # 음성 인식 없이 (YOLO + LLM만)
  python main.py --no-llm             # LLM 없이 (YOLO + 마이크만)
  python main.py --no-display         # 화면 없이 (서버 전송만)
  python main.py --config my.yaml     # 커스텀 설정

실행 중 키:
  Q: 종료
  P: 파이프라인 전환 (security ↔ full_analysis)
        """
    )
    parser.add_argument('--config', default=str(Path(__file__).parent / 'config.yaml'), help='설정 파일 경로')
    parser.add_argument('--no-mic', action='store_true', help='마이크 어레이 비활성화')
    parser.add_argument('--no-stt', action='store_true', help='음성 인식(STT) 비활성화')
    parser.add_argument('--no-llm', action='store_true', help='ContextLLM 비활성화')
    parser.add_argument('--no-yolo', action='store_true', help='YOLO 비활성화')
    parser.add_argument('--no-display', action='store_true', help='OpenCV 디스플레이 비활성화')
    parser.add_argument('--debug', action='store_true', help='DEBUG 로깅 활성화')

    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.config)
    
    if args.debug:
        config.setdefault("logging", {})["level"] = "DEBUG"
    
    setup_logging(config)
    logger = logging.getLogger("Main")

    logger.info("=" * 60)
    logger.info("  KNU-MM 통합 멀티모달 관제 시스템")
    logger.info("  시각(YOLO) + 청각(MicArray+STT) → LLM 통합 분석")
    logger.info("=" * 60)

    dashboard_procs = _start_dashboards_if_enabled(config, logger)

    # 시스템 빌드
    orch, event_bus, stream, ptz, mic_module, stt_module = build_system(config, args)

    # Graceful Shutdown
    def signal_handler(sig, frame):
        logger.info("\n종료 신호 수신...")
        if stt_module:
            stt_module.stop_listening()
        orch.shutdown_all()
        stream.release()
        ptz.shutdown()
        _stop_dashboards(dashboard_procs, logger)
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 메인 루프 실행
    try:
        run_main_loop(orch, stream, config, args, stt_module)
    finally:
        logger.info("시스템 종료 중...")
        if stt_module:
            stt_module.stop_listening()
        orch.shutdown_all()
        stream.release()
        ptz.shutdown()
        _stop_dashboards(dashboard_procs, logger)
        cv2.destroyAllWindows()
        logger.info("시스템 종료 완료")


if __name__ == "__main__":
    main()
