#!/usr/bin/env python3
"""
Mic Array + ContextLLM fusion runner.

Workflow:
1) Mic array detects speech direction (DOA).
2) PTZ moves to the detected direction.
3) STT captures speech text OR emits non-speech audio candidate.
4) After PTZ settle delay, current camera frame is captured.
5) ContextLLM analyzes text + frame (+ non-speech context, if available).
"""

import argparse
import logging
import os
import signal
import subprocess
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse, urlunparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_logging(config: Dict[str, Any]) -> None:
    runtime_cfg = config.get("runtime", {})
    level_name = runtime_cfg.get("log_level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)-18s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def apply_openai_api_key(config: Dict[str, Any], cli_key: Optional[str]) -> None:
    """
    OPENAI_API_KEY 주입 우선순위:
    1) CLI --openai-api-key
    2) mic_context_fusion/config.yaml 의 openai.api_key
    3) 기존 환경변수 / .env 값 유지
    """
    if cli_key:
        os.environ["OPENAI_API_KEY"] = cli_key.strip()
        return

    cfg_key = (config.get("openai", {}) or {}).get("api_key", "")
    if isinstance(cfg_key, str) and cfg_key.strip():
        os.environ["OPENAI_API_KEY"] = cfg_key.strip()


def parse_camera_source(value: Any) -> Union[int, str]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def parse_bool(value: Any, default: bool = False) -> bool:
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


def parse_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    if port <= 0:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, int(port))) == 0


def _resolve_script_path(script_value: str) -> Path:
    script_path = Path(script_value)
    if not script_path.is_absolute():
        script_path = (PROJECT_ROOT / script_path).resolve()
    return script_path


def _start_dashboard_process(
    script_path: Path,
    label: str,
    logger: logging.Logger,
    env: Optional[Dict[str, str]] = None,
):
    if not script_path.exists():
        logger.warning("[%s] auto-start failed: script not found (%s)", label, script_path)
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
                "[%s] auto-start failed: process exited early (code=%s)",
                label,
                proc.returncode,
            )
            return None
        logger.info("[%s] auto-started (pid=%s, script=%s)", label, proc.pid, script_path)
        return proc
    except Exception as exc:
        logger.warning("[%s] auto-start failed: %s", label, exc)
        return None


def sync_server_url_with_event_port(config: Dict[str, Any], event_port: int, logger: logging.Logger) -> None:
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
    if (parsed.port or 80) == int(event_port):
        return

    new_netloc = f"{parsed.hostname}:{event_port}"
    updated = parsed._replace(netloc=new_netloc, path="/event")
    new_url = urlunparse(updated)
    server_cfg["url"] = new_url
    logger.info("[EventDashboard] server.url auto-synced to %s", new_url)


def start_dashboards_if_enabled(config: Dict[str, Any], logger: logging.Logger):
    server_cfg = config.get("server", {}) or {}
    procs = []
    started_scripts = set()
    event_port = parse_int(server_cfg.get("event_dashboard_port"), 8100)
    contextllm_port = parse_int(server_cfg.get("contextllm_dashboard_port"), 5100)

    event_dashboard_enabled = (
        parse_bool(server_cfg.get("auto_start_dashboard", False), False)
        or parse_bool(server_cfg.get("auto_start_event_dashboard", False), False)
    )
    if event_dashboard_enabled:
        sync_server_url_with_event_port(config, event_port, logger)
        script_path = _resolve_script_path(
            str(server_cfg.get("dashboard_script", "integrated_system/modules/dashboard_server.py"))
        )
        key = str(script_path)
        if key not in started_scripts:
            if is_port_in_use(event_port):
                logger.warning(
                    "[EventDashboard] port %s already in use. Skip auto-start.",
                    event_port,
                )
            else:
                env = os.environ.copy()
                env["EVENT_DASHBOARD_PORT"] = str(event_port)
                proc = _start_dashboard_process(script_path, "EventDashboard", logger, env=env)
                if proc is not None:
                    procs.append(("EventDashboard", proc))
                    started_scripts.add(key)

    contextllm_dashboard_enabled = parse_bool(server_cfg.get("auto_start_contextllm_dashboard", False), False)
    if contextllm_dashboard_enabled:
        script_path = _resolve_script_path(
            str(server_cfg.get("contextllm_dashboard_script", "contextllm/src/web/app.py"))
        )
        key = str(script_path)
        if key in started_scripts:
            logger.info("[ContextLLMDashboard] duplicate script skipped: %s", script_path)
        else:
            if is_port_in_use(contextllm_port):
                logger.warning(
                    "[ContextLLMDashboard] port %s already in use. Skip auto-start.",
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


def stop_dashboards(procs, logger: logging.Logger) -> None:
    for label, proc in reversed(procs or []):
        if proc is None or proc.poll() is not None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=3)
            logger.info("[%s] auto-start process stopped.", label)
        except Exception:
            try:
                proc.kill()
                logger.info("[%s] auto-start process killed.", label)
            except Exception:
                pass


class MicContextFusionApp:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("MicContextFusion")

        try:
            from integrated_system.core.event_bus import EventBus
            from integrated_system.core.orchestrator import Orchestrator
            from integrated_system.modules.mic_array import MicArrayModule
            from integrated_system.modules.ptz_controller import PTZPriority, UnifiedPTZController
            from integrated_system.modules.server_reporter import ServerReporterModule
            from integrated_system.modules.stt_module import STTModule
            from integrated_system.modules.stream_manager import SharedStreamManager
        except ImportError as exc:
            raise RuntimeError(
                "integrated_system dependencies are missing. "
                "Install project requirements first: pip install -r integrated_system/requirements.txt"
            ) from exc

        self._ptz_priority_mic_doa = PTZPriority.MIC_DOA

        self.event_bus = EventBus(max_workers=6, async_mode=True)
        self.orchestrator = Orchestrator(self.event_bus)

        camera_cfg = config.get("camera", {})
        ptz_cfg = config.get("ptz", {})
        mic_cfg = config.get("mic_array", {})
        stt_cfg = config.get("stt", {})
        llm_cfg = config.get("context_llm", {})
        non_speech_cfg = config.get("non_speech", {}) or {}
        server_cfg = config.get("server", {}) or {}

        source = parse_camera_source(camera_cfg.get("source", 0))
        self.stream = SharedStreamManager(source).start()

        self.ptz = UnifiedPTZController(
            {
                "camera_ip": ptz_cfg.get("ip", ""),
                "camera_port": ptz_cfg.get("port", 80),
                "camera_user": ptz_cfg.get("user", ""),
                "camera_password": ptz_cfg.get("password", ""),
                "control_mode": ptz_cfg.get("control_mode", "onvif"),
            }
        )
        self.ptz.initialize()

        self.mic_module = MicArrayModule(
            event_bus=self.event_bus,
            ptz=self.ptz,
            agc_max_gain=mic_cfg.get("agc_max_gain", 15.0),
            vad_threshold=mic_cfg.get("vad_threshold", 10.0),
            confidence_threshold=mic_cfg.get("confidence_threshold", 0.6),
            zenith_confidence=mic_cfg.get("zenith_confidence", 0.4),
            zenith_gain=mic_cfg.get("zenith_gain", 10.0),
            history_size=mic_cfg.get("history_size", 10),
        )
        self.stt_module = STTModule(
            event_bus=self.event_bus,
            language=stt_cfg.get("language", "ko-KR"),
            energy_threshold=stt_cfg.get("energy_threshold", 400),
            pause_threshold=stt_cfg.get("pause_threshold", 3.0),
            phrase_time_limit=stt_cfg.get("phrase_time_limit", 15.0),
            dynamic_threshold=stt_cfg.get("dynamic_threshold", True),
        )
        self.server_module = None
        if parse_bool(server_cfg.get("enabled", False), False):
            self.server_module = ServerReporterModule(
                event_bus=self.event_bus,
                server_url=str(server_cfg.get("url", "") or ""),
                timeout=parse_float(server_cfg.get("timeout", 2.0), 2.0),
                enabled=True,
            )

        # ContextLLM app/service 계층을 직접 사용해 결합 (config-first 구조 이식).
        contextllm_src = PROJECT_ROOT / "contextllm" / "src"
        if str(contextllm_src) not in sys.path:
            sys.path.insert(0, str(contextllm_src))

        try:
            from app.service import ContextLLMService
        except ImportError as exc:
            raise RuntimeError(
                "contextllm app/service import failed. "
                "Ensure contextllm/src is available and dependencies are installed."
            ) from exc

        contextllm_config = llm_cfg.get("config_path", "") or str(
            PROJECT_ROOT / "contextllm" / "config" / "config.yaml"
        )
        llm_overrides: Dict[str, Any] = {}
        if llm_cfg.get("model"):
            llm_overrides["model"] = llm_cfg["model"]

        self.llm_service = ContextLLMService.from_config(
            contextllm_config,
            overrides=llm_overrides or None,
        )
        if not self._is_llm_available():
            self.logger.warning(
                "ContextLLM analyzer is unavailable. "
                "Set OPENAI_API_KEY (or mic_context_fusion/config.yaml: openai.api_key) "
                "to enable multimodal analysis."
            )

        self.non_speech_enabled = parse_bool(non_speech_cfg.get("enabled", True), True)
        self.non_speech_min_duration = parse_float(non_speech_cfg.get("min_audio_duration", 0.2), 0.2)
        self.non_speech_cooldown_seconds = parse_float(non_speech_cfg.get("cooldown_seconds", 1.5), 1.5)
        self.non_speech_analyze_on_detected = parse_bool(
            non_speech_cfg.get("analyze_on_detected", False),
            False,
        )
        self._last_non_speech_time = 0.0
        self._non_speech_state_lock = threading.Lock()
        self.sound_event_detector = None
        self._init_sound_event_detector(non_speech_cfg)

        self.ptz_tilt = ptz_cfg.get("tilt_on_doa", -15)
        self.ptz_settle_seconds = float(ptz_cfg.get("settle_seconds", 1.2))
        self.max_doa_age_seconds = float(ptz_cfg.get("max_doa_age_seconds", 5.0))

        self._running = False
        self._analysis_lock = threading.Lock()
        self._last_doa_sector: Optional[float] = None
        self._last_doa_time: float = 0.0
        self._state_lock = threading.Lock()

        self.event_bus.subscribe("mic.doa_detected", self._on_doa_detected)
        self.event_bus.subscribe("stt.text_recognized", self._on_stt_text)
        self.event_bus.subscribe("stt.non_speech_audio", self._on_non_speech_audio)
        self.event_bus.subscribe("llm.emergency", self._on_emergency)

    def _is_llm_available(self) -> bool:
        if self.llm_service is None:
            return False
        system = getattr(self.llm_service, "system", None)
        analyzer = getattr(system, "multimodal_analyzer", None) if system is not None else None
        return analyzer is not None

    def _init_sound_event_detector(self, non_speech_cfg: Dict[str, Any]) -> None:
        if not self.non_speech_enabled:
            self.logger.info("Non-speech path disabled by config.")
            return

        try:
            from core.sound_event_detector import SoundEventDetector
        except ImportError:
            self.logger.warning(
                "SoundEventDetector import failed. "
                "Install TensorFlow + TensorFlow Hub to enable non-speech path."
            )
            return

        base_cfg = (getattr(self.llm_service, "raw_config", {}) or {}).get("sound_event", {}) or {}
        detector_cfg = dict(base_cfg)
        for key in ("model_url", "min_confidence", "trigger_threshold", "top_k", "emergency_keywords"):
            if key in non_speech_cfg:
                detector_cfg[key] = non_speech_cfg[key]

        self.sound_event_detector = SoundEventDetector(
            model_url=detector_cfg.get("model_url", "https://tfhub.dev/google/yamnet/1"),
            min_confidence=detector_cfg.get("min_confidence", 0.12),
            trigger_threshold=detector_cfg.get("trigger_threshold", 0.25),
            top_k=detector_cfg.get("top_k", 5),
            emergency_keywords=detector_cfg.get("emergency_keywords", []),
        )

        if getattr(self.sound_event_detector, "enabled", False):
            self.logger.info("Non-speech detector ready (YAMNet).")
        else:
            self.logger.warning("Non-speech detector is disabled (dependency/model unavailable).")

    def _register_modules(self) -> None:
        mic_ready = self.orchestrator.register(self.mic_module)
        if mic_ready:
            self.mic_module.start_monitoring()
        else:
            self.logger.warning("MicArray module is not ready.")

        stt_ready = self.orchestrator.register(self.stt_module)
        if stt_ready:
            self.stt_module.start_listening()
        else:
            self.logger.warning("STT module is not ready.")

        server_ready = False
        if self.server_module is not None:
            server_ready = self.orchestrator.register(self.server_module)
            if not server_ready:
                self.logger.warning("Server reporter module is not ready.")

        llm_ready = self._is_llm_available()

        self.logger.info(
            "Module status - mic: %s, stt: %s, llm: %s, web: %s",
            "ready" if mic_ready else "disabled",
            "ready" if stt_ready else "disabled",
            "ready" if llm_ready else "disabled",
            "ready" if server_ready else ("off" if self.server_module is None else "disabled"),
        )

    def start(self) -> None:
        self._register_modules()
        self._running = True
        self.logger.info("Fusion workflow started.")
        self.logger.info(
            "Flow: mic.doa_detected -> PTZ move -> (stt.text_recognized | stt.non_speech_audio) -> frame capture -> ContextLLM"
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.logger.info("Stopping fusion workflow...")

        try:
            self.orchestrator.shutdown_all()
        finally:
            try:
                self.stream.release()
            except Exception:
                pass
            try:
                self.ptz.shutdown()
            except Exception:
                pass

        self.logger.info("Fusion workflow stopped.")

    def run_forever(self) -> None:
        while self._running:
            time.sleep(0.2)

    def _on_doa_detected(self, event) -> None:
        sector = event.data.get("sector_angle")
        if sector is None:
            return

        with self._state_lock:
            self._last_doa_sector = float(sector)
            self._last_doa_time = time.time()

        self.ptz.request_move(
            pan=float(sector),
            tilt=self.ptz_tilt,
            priority=self._ptz_priority_mic_doa,
            owner="mic_context_fusion",
            move_type="absolute",
        )
        self.logger.info("DOA detected: %.1f deg -> PTZ move requested.", float(sector))

    def _on_stt_text(self, event) -> None:
        text = (event.data.get("text") or "").strip()
        if not text:
            return

        if self._analysis_lock.locked():
            self.logger.info("Analysis in progress, ignored STT text=%s", text)
            return

        # Keep event callbacks non-blocking.
        threading.Thread(
            target=self._run_multimodal_analysis,
            args=(text,),
            kwargs={"trigger_source": "speech"},
            daemon=True,
            name="FusionAnalysisWorker",
        ).start()

    def _on_non_speech_audio(self, event) -> None:
        if not self.non_speech_enabled or self.sound_event_detector is None:
            return

        if self._analysis_lock.locked():
            return

        audio = event.data.get("audio")
        if audio is None:
            return

        duration = parse_float(event.data.get("duration", 0.0), 0.0)
        if duration < self.non_speech_min_duration:
            return

        now = time.time()
        with self._non_speech_state_lock:
            if now - self._last_non_speech_time < self.non_speech_cooldown_seconds:
                return
            self._last_non_speech_time = now

        doa_hint = event.data.get("doa_angle")
        threading.Thread(
            target=self._run_non_speech_analysis,
            args=(audio, doa_hint),
            daemon=True,
            name="FusionNonSpeechWorker",
        ).start()

    def _run_non_speech_analysis(self, audio: Any, doa_hint: Optional[float]) -> None:
        if self.sound_event_detector is None:
            return

        detection = self.sound_event_detector.detect_from_audio(audio)
        has_detected_event = bool(detection.get("event_detected", False))
        has_trigger = bool(detection.get("triggered", False))

        if not has_detected_event:
            return
        if not has_trigger and not self.non_speech_analyze_on_detected:
            return

        top_event = detection.get("top_event") or "unknown_sound"
        top_conf = parse_float(detection.get("top_confidence", 0.0), 0.0)
        self.logger.info(
            "Non-speech event detected: %s (%.2f), triggered=%s",
            top_event,
            top_conf,
            has_trigger,
        )

        analysis_text = f"[음성 텍스트 없음] 비음성 이벤트 감지: {top_event}"
        self._run_multimodal_analysis(
            text=analysis_text,
            sound_event=detection,
            trigger_source="sound_event",
            doa_hint=doa_hint,
        )

    def _run_multimodal_analysis(
        self,
        text: str,
        sound_event: Optional[Dict[str, Any]] = None,
        trigger_source: str = "speech",
        doa_hint: Optional[float] = None,
    ) -> None:
        if not self._running:
            return

        if not self._is_llm_available():
            self.logger.warning("ContextLLM analyzer unavailable; skip trigger=%s", trigger_source)
            return

        if not self._analysis_lock.acquire(blocking=False):
            self.logger.info("Analysis in progress, skipped trigger=%s text=%s", trigger_source, text)
            return

        try:
            doa_hint_float = parse_optional_float(doa_hint)
            if doa_hint_float is not None:
                doa = doa_hint_float
                doa_age = 0.0
            else:
                now = time.time()
                with self._state_lock:
                    doa = self._last_doa_sector
                    doa_age = now - self._last_doa_time

            if doa is not None and doa_age <= self.max_doa_age_seconds:
                self.ptz.request_move(
                    pan=float(doa),
                    tilt=self.ptz_tilt,
                    priority=self._ptz_priority_mic_doa,
                    owner="mic_context_fusion",
                    move_type="absolute",
                )
                time.sleep(self.ptz_settle_seconds)

            frame = self._capture_frame_with_retry(max_retry=10, retry_interval=0.1)
            if frame is None:
                self.logger.warning("Frame capture failed. ContextLLM analysis skipped.")
                return

            additional_context = self._build_additional_context(doa=doa)
            service_result = self.llm_service.analyze_frame(
                text=text,
                frame=frame,
                additional_context=additional_context,
                sound_event=sound_event,
            )
            if service_result.get("success"):
                analysis = service_result.get("analysis", {}) or {}
                priority = service_result.get("priority", "LOW")
                urgency = service_result.get("urgency", "LOW")
                situation = analysis.get("situation_type", "N/A")
                self.logger.info(
                    'ContextLLM analyzed: trigger=%s priority=%s urgency=%s situation=%s text="%s"',
                    trigger_source,
                    priority,
                    urgency,
                    situation,
                    text,
                )
                self._print_contextllm_style_result(
                    text=text,
                    trigger_source=trigger_source,
                    sound_event=sound_event,
                    analysis=analysis,
                    priority=priority,
                    urgency=urgency,
                    is_emergency=bool(service_result.get("is_emergency", False)),
                )
                self.event_bus.publish_simple(
                    "fusion.analysis_complete",
                    {
                        "text": text,
                        "doa_sector": doa,
                        "trigger_source": trigger_source,
                        "sound_event": sound_event,
                        "priority": priority,
                        "urgency": urgency,
                        "situation_type": situation,
                        "analysis": analysis,
                    },
                    source="mic_context_fusion",
                )
                self.event_bus.publish_simple(
                    "llm.analysis_complete",
                    {
                        "result": analysis,
                        "priority": priority,
                        "is_emergency": bool(service_result.get("is_emergency", False)),
                        "speech_text": text,
                        "doa_sector": doa,
                        "trigger_source": trigger_source,
                        "sound_event": sound_event,
                    },
                    source="mic_context_fusion",
                )

                if service_result.get("is_emergency"):
                    self.event_bus.publish_simple(
                        "llm.emergency",
                        {"result": analysis, "urgency": urgency, "situation": situation},
                        source="mic_context_fusion",
                        priority=2,
                    )
            else:
                self.logger.info(
                    "ContextLLM failed: trigger=%s reason=%s text=%s",
                    trigger_source,
                    service_result.get("error", "unknown"),
                    text,
                )
        except Exception as exc:
            self.logger.error("Fusion analysis error: %s", exc)
        finally:
            self._analysis_lock.release()

    def _build_additional_context(self, doa: Optional[float]) -> Optional[str]:
        contexts = []
        contexts.append("doa_sector=unknown" if doa is None else f"doa_sector={doa:.1f}")
        return "\n\n".join([ctx for ctx in contexts if ctx]) or None

    def _capture_frame_with_retry(self, max_retry: int, retry_interval: float):
        for _ in range(max_retry):
            frame = self.stream.get_frame()
            if frame is not None:
                return frame
            time.sleep(retry_interval)
        return None

    def _on_emergency(self, event) -> None:
        result = event.data.get("result", {})
        self.logger.warning(
            "EMERGENCY detected: urgency=%s summary=%s",
            result.get("urgency", "N/A"),
            result.get("summary", "N/A"),
        )

    def _print_contextllm_style_result(
        self,
        text: str,
        trigger_source: str,
        sound_event: Optional[Dict[str, Any]],
        analysis: Dict[str, Any],
        priority: str,
        urgency: str,
        is_emergency: bool,
    ) -> None:
        print("\n" + "=" * 50)
        print("📊 분석 결과")
        print("=" * 50)
        print(f'📝 음성 입력: "{text}"')

        if sound_event and sound_event.get("top_event"):
            top_conf = float(sound_event.get("top_confidence", 0.0) or 0.0)
            print("\n🔊 사운드 이벤트 분석:")
            print(f"   - 최상위 이벤트: {sound_event.get('top_event')} ({top_conf:.2f})")
            emergency_events = sound_event.get("emergency_events", []) or []
            if emergency_events:
                labels = ", ".join(
                    f"{e.get('label', 'unknown')}({float(e.get('confidence', 0.0) or 0.0):.2f})"
                    for e in emergency_events[:3]
                )
                print(f"   - 위험 후보: {labels}")
            print(f"   - 트리거 소스: {trigger_source}")

        print("\n🔍 상황 분석:")
        print(f"   - 상황 유형: {analysis.get('situation_type', 'N/A')}")
        print(f"   - 상황 설명: {analysis.get('situation', 'N/A')}")
        print(f"   - 감정 상태: {analysis.get('emotional_state', 'N/A')}")
        print(f"   - 영상 내용: {analysis.get('visual_content', 'N/A')}")

        print("\n⚠️  긴급도 판단:")
        print("   - 긴급 여부: 🚨 YES - 즉시 대응 필요!" if is_emergency else "   - 긴급 여부: ✅ 아니오")
        print(f"   - 우선순위: {priority}")
        print(f"   - 위급도: {urgency}")
        print(f"   - 긴급 판단 근거: {analysis.get('emergency_reason', 'N/A')}")

        print("\n🎯 음성-영상 일치도:")
        print(f"   - 일치 여부: {analysis.get('audio_visual_consistency', 'N/A')}")

        print("\n💡 권장 조치:")
        action = analysis.get("action", "N/A")
        print(f"   - {'🚨 긴급: ' if is_emergency else ''}{action}")

        if is_emergency:
            print("🚨" * 50 + "\n")
        else:
            print("=" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mic Array + ContextLLM fusion workflow runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config path",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="",
        help="OpenAI API key override (highest priority)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("MicContextFusion")
    dashboard_procs = start_dashboards_if_enabled(config, logger)

    # Load API keys and camera envs, if present.
    root_env = PROJECT_ROOT / ".env"
    ctx_env = PROJECT_ROOT / "contextllm" / "config" / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    if ctx_env.exists():
        load_dotenv(ctx_env)
    apply_openai_api_key(config, args.openai_api_key)

    app = MicContextFusionApp(config)

    def _shutdown_handler(signum, _frame):
        logger.info("Signal received: %s", signum)
        app.stop()
        stop_dashboards(dashboard_procs, logger)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    app.start()
    try:
        app.run_forever()
    finally:
        app.stop()
        stop_dashboards(dashboard_procs, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
