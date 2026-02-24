"""
Context LLM 모듈 - 음성+영상 멀티모달 LLM 분석을 BaseModule로 래핑

원본 모듈 (직접 import — 수정 시 즉시 반영):
    - contextllm/src/core/integrated_multimodal_system.py → IntegratedMultimodalSystem
    - contextllm/src/core/config_manager.py               → get_config
    - contextllm/src/core/voice_characteristics.py         → VoiceCharacteristicsAnalyzer
    - contextllm/src/core/multimodal_analyzer.py           → MultimodalAnalyzer

    ※ contextllm/src/core/ 는 integrated_system/core/ 와 패키지명이 충돌하므로
      importlib.util.spec_from_file_location 으로 직접 파일 경로 기반 로딩합니다.

통합 레이어 (이 파일에만 존재):
    - 쿨다운 제어 (API 호출 빈도 제한)
    - EventBus 이벤트 발행 / 구독
    - 긴급도 판정 및 긴급 이벤트 발행
    - STT 텍스트 수신 → 영상+음성 통합 분석 트리거
    - STT 비음성 오디오 수신 → YAMNet 판별 후 멀티모달 분석 트리거
    - YOLO 객체 목록 요약을 LLM 추가 컨텍스트로 전달

이벤트 발행:
    - llm.analysis_complete : 분석 완료 시
    - llm.emergency         : 긴급 상황 판정 시 (priority=2)

이벤트 구독:
    - yolo.person_detected  : 사람 감지 시 자동 분석 트리거
    - mic.speech_detected   : 음성 감지 시 분석 트리거
    - stt.text_recognized   : STT 텍스트 수신 → 음성+영상 통합 분석
    - stt.non_speech_audio  : STT 인식 실패 오디오 수신 → 비음성 이벤트 분석
"""

import sys
import os
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import requests
except ImportError:
    requests = None

from integrated_system_process.core.base_module import BaseModule
from integrated_system_process.core.event_bus import EventBus, Event
from integrated_system_process.core.module_loader import (
    CONTEXTLLM_DIR, CONTEXTLLM_SRC, CONTEXTLLM_CORE,
    ensure_path, import_from_file,
)

logger = logging.getLogger(__name__)


class ContextLLMModule(BaseModule):
    """
    OpenAI GPT 기반 멀티모달 상황 분석 모듈

    ★ contextllm/src/core/ 의 원본 시스템을 importlib로 직접 로드합니다.
    원본 파일을 수정하면 즉시 반영됩니다.

    분석 트리거:
        1. STT 텍스트 수신 시 → 해당 프레임을 캡처하여 음성+영상 통합 분석
        2. STT 비음성 오디오 수신 시 → YAMNet 위험 이벤트 판별 후 통합 분석
        (사람 감지만으로는 분석하지 않음 — 원본 contextllm 동작 유지)
    """

    def __init__(
        self,
        event_bus: EventBus,
        model: str = "gpt-4o-mini",
        config_path: Optional[str] = None,
        ptz: Optional[Any] = None,
        ptz_settle_seconds: float = 0.6,
        max_doa_age_seconds: float = 5.0,
        analysis_cooldown: float = 5.0,
        non_speech_enabled: bool = True,
        non_speech_min_duration: float = 0.2,
        non_speech_cooldown: float = 1.5,
        non_speech_analyze_on_detected: bool = False,
        doa_object_mapping_enabled: bool = True,
        camera_hfov_deg: float = 60.0,
        doa_object_match_threshold_deg: float = 12.0,
        contextllm_dashboard_push_enabled: bool = True,
        contextllm_dashboard_url: str = "http://127.0.0.1:5100/api/push_result",
        contextllm_dashboard_timeout: float = 1.2,
    ):
        super().__init__(event_bus)

        def _safe_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        self.model = model
        self.config_path = config_path or str(Path(CONTEXTLLM_DIR) / "config" / "config.yaml")
        self.ptz = ptz
        self.ptz_settle_seconds = max(0.0, _safe_float(ptz_settle_seconds, 0.6))
        self.max_doa_age_seconds = max(0.0, _safe_float(max_doa_age_seconds, 5.0))
        self.analysis_cooldown = max(0.0, _safe_float(analysis_cooldown, 5.0))
        self.non_speech_enabled = bool(non_speech_enabled)
        self.non_speech_min_duration = max(0.0, _safe_float(non_speech_min_duration, 0.2))
        self.non_speech_cooldown = max(0.0, _safe_float(non_speech_cooldown, 1.5))
        self.non_speech_analyze_on_detected = bool(non_speech_analyze_on_detected)
        self.doa_object_mapping_enabled = bool(doa_object_mapping_enabled)
        self.camera_hfov_deg = max(1.0, _safe_float(camera_hfov_deg, 60.0))
        self.doa_object_match_threshold_deg = max(0.0, _safe_float(doa_object_match_threshold_deg, 12.0))
        self.contextllm_dashboard_push_enabled = bool(contextllm_dashboard_push_enabled)
        self.contextllm_dashboard_url = str(contextllm_dashboard_url or "").strip()
        if not self.contextllm_dashboard_url:
            self.contextllm_dashboard_url = "http://127.0.0.1:5100/api/push_result"
        self.contextllm_dashboard_timeout = max(0.1, _safe_float(contextllm_dashboard_timeout, 1.2))
        self._dashboard_push_warned = False

        self._system = None          # IntegratedMultimodalSystem 인스턴스
        self._multimodal_analyzer = None  # MultimodalAnalyzer 직접 참조
        self._last_analysis_time = 0.0
        self._last_result: Dict = {}

        # STT 연동 상태
        self._pending_text: Optional[str] = None
        self._pending_text_time: float = 0.0
        self._text_lock = threading.Lock()

        # 비음성(YAMNet) 연동 상태
        self._pending_sound_event: Optional[Dict[str, Any]] = None
        self._pending_sound_event_time: float = 0.0
        self._last_non_speech_time: float = 0.0
        self._sound_lock = threading.Lock()

        # DOA / YOLO 스냅샷 (mic_context_fusion 방식 통합)
        self._last_doa_sector: Optional[float] = None
        self._last_doa_time: float = 0.0
        self._state_lock = threading.Lock()
        self._latest_yolo_result: Dict[str, Any] = {}
        self._latest_yolo_time: float = 0.0
        self._yolo_lock = threading.Lock()

        # 최신 분석 결과 (OpenCV 오버레이용)
        self._display_result: Dict = {}
        self._display_lock = threading.Lock()

        # 비동기 LLM 처리를 위한 워커 스레드 및 큐 (메인 루프 프리징 방지)
        self._analysis_queue = queue.Queue(maxsize=5) # 큐 크기 제한으로 메모리/요청 폭주 방지
        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self._worker_thread.start()


    @property
    def name(self) -> str:
        return "context_llm"

    def initialize(self) -> bool:
        """
        ContextLLM 시스템 초기화

        ★ importlib 직접 로딩으로 패키지 충돌(integrated_system/core vs contextllm/src/core) 회피
        """
        try:
            # contextllm/src 를 sys.path에 추가 (하위 의존성용)
            ensure_path(CONTEXTLLM_SRC)
            ensure_path(CONTEXTLLM_DIR)

            # ★ 의존 모듈을 파일 경로로 직접 로드 (core 패키지 충돌 방지) ★
            config_mgr_path = os.path.join(CONTEXTLLM_CORE, "config_manager.py")
            if os.path.exists(config_mgr_path):
                import_from_file("core.config_manager", config_mgr_path)

            voice_char_path = os.path.join(CONTEXTLLM_CORE, "voice_characteristics.py")
            if os.path.exists(voice_char_path):
                import_from_file("core.voice_characteristics", voice_char_path)

            multimodal_path = os.path.join(CONTEXTLLM_CORE, "multimodal_analyzer.py")
            if os.path.exists(multimodal_path):
                import_from_file("core.multimodal_analyzer", multimodal_path)

            # 메인 모듈 로드
            ims_path = os.path.join(CONTEXTLLM_CORE, "integrated_multimodal_system.py")
            ims_mod = import_from_file("core.integrated_multimodal_system", ims_path)
            IntegratedMultimodalSystem = ims_mod.IntegratedMultimodalSystem
            DownsamplingConfig = ims_mod.DownsamplingConfig

            # config.yaml 로드
            config = self._load_config()
            ds = config.get('downsampling', {})

            ds_config = DownsamplingConfig(
                max_image_size=ds.get('max_image_size', 640),
                jpeg_quality=ds.get('jpeg_quality', 75),
                video_fps=ds.get('video_fps', 2.0),
                max_video_frames=ds.get('max_video_frames', 10),
                video_capture_duration=ds.get('video_capture_duration', 5.0),
            )

            self._system = IntegratedMultimodalSystem(
                camera_id=None,  # 외부 프레임 사용
                model=self.model,
                downsampling_config=ds_config,
            )

            # MultimodalAnalyzer 직접 참조 (analyze_with_image 호출용)
            self._multimodal_analyzer = getattr(self._system, 'multimodal_analyzer', None)

            # 이벤트 구독
            self._event_bus.subscribe("yolo.person_detected", self._on_person_detected)
            self._event_bus.subscribe("yolo.objects_detected", self._on_yolo_objects_detected)
            self._event_bus.subscribe("mic.speech_detected", self._on_speech_detected)
            self._event_bus.subscribe("mic.doa_detected", self._on_doa_detected)
            self._event_bus.subscribe("stt.text_recognized", self._on_stt_text)
            self._event_bus.subscribe("stt.non_speech_audio", self._on_non_speech_audio)

            logger.info("[ContextLLM] 초기화 완료 (원본 IntegratedMultimodalSystem 사용)")
            if self._multimodal_analyzer:
                logger.info("[ContextLLM] MultimodalAnalyzer 연결됨 (음성+영상 통합 분석 가능)")
            else:
                logger.warning("[ContextLLM] MultimodalAnalyzer 없음 (openai 미설치?)")

            if self.non_speech_enabled:
                logger.info(
                    "[ContextLLM] 비음성 경로 활성화 "
                    f"(min_duration={self.non_speech_min_duration:.2f}s, cooldown={self.non_speech_cooldown:.2f}s, "
                    f"analyze_on_detected={self.non_speech_analyze_on_detected})"
                )
            else:
                logger.info("[ContextLLM] 비음성 경로 비활성화")
            logger.info(
                "[ContextLLM] ContextLLM dashboard push: enabled=%s, url=%s",
                self.contextllm_dashboard_push_enabled,
                self.contextllm_dashboard_url,
            )
            return True

        except ImportError as e:
            logger.warning(f"[ContextLLM] 의존성 없음: {e}")
            return False
        except Exception as e:
            logger.error(f"[ContextLLM] 초기화 실패: {e}")
            return False

    def _load_config(self) -> dict:
        """config.yaml 로드"""
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    # ─── 이벤트 핸들러 ───

    def _on_person_detected(self, event: Event) -> None:
        """YOLO 사람 감지 이벤트 핸들러"""
        logger.debug(f"[ContextLLM] 사람 감지 이벤트 수신 (count: {event.data.get('count', 0)})")
        with self._yolo_lock:
            self._latest_yolo_result = {
                "person_detected": True,
                "count": event.data.get("count", 0),
                "objects": event.data.get("objects", []),
                "target": event.data.get("target"),
                "mode": event.data.get("mode", ""),
            }
            self._latest_yolo_time = time.time()

    def _on_yolo_objects_detected(self, event: Event) -> None:
        """YOLO 객체 감지 이벤트 스냅샷 저장."""
        with self._yolo_lock:
            self._latest_yolo_result = {
                "person_detected": any(
                    str(obj.get("name", "")).lower() == "person"
                    for obj in (event.data.get("objects") or [])
                ),
                "count": event.data.get("count", 0),
                "objects": event.data.get("objects", []),
                "target": event.data.get("target"),
                "mode": event.data.get("mode", ""),
            }
            self._latest_yolo_time = time.time()

    def _on_speech_detected(self, event: Event) -> None:
        """마이크 음성 감지 이벤트 핸들러 (DOA 정보)"""
        logger.debug(f"[ContextLLM] 음성 감지 이벤트 수신 (angle: {event.data.get('raw_angle', '?')})")

    def _on_doa_detected(self, event: Event) -> None:
        """Mic DOA 최신 방향 저장."""
        sector = event.data.get("sector_angle")
        if sector is None:
            return
        with self._state_lock:
            self._last_doa_sector = float(sector)
            self._last_doa_time = time.time()

    def _on_stt_text(self, event: Event) -> None:
        """
        STT 텍스트 수신 → 대기 텍스트에 저장

        다음 파이프라인 실행 시 이 텍스트와 현재 프레임을 함께 분석합니다.
        """
        text = event.data.get("text", "")
        if text:
            with self._text_lock:
                self._pending_text = text
                self._pending_text_time = time.time()
            logger.info(f'[ContextLLM] STT 텍스트 수신: "{text}"')

    def _on_non_speech_audio(self, event: Event) -> None:
        """
        STT 비음성 오디오 이벤트 수신.
        오디오를 YAMNet으로 판별해 위험 후보일 때 멀티모달 분석을 트리거한다.
        """
        if not self.non_speech_enabled:
            return

        audio = event.data.get("audio")
        if audio is None:
            return

        duration = float(event.data.get("duration", 0.0) or 0.0)
        if duration < self.non_speech_min_duration:
            return

        now = time.time()
        with self._sound_lock:
            if now - self._last_non_speech_time < self.non_speech_cooldown:
                return
            self._last_non_speech_time = now

        sound_event = self._detect_sound_event(audio)
        if not sound_event:
            return

        has_detected_event = bool(sound_event.get("event_detected", False))
        has_trigger = bool(sound_event.get("triggered", False))
        if not has_detected_event:
            return
        if not has_trigger and not self.non_speech_analyze_on_detected:
            return

        top_event = sound_event.get("top_event") or "unknown_sound"
        top_conf = float(sound_event.get("top_confidence", 0.0) or 0.0)
        synthetic_text = f"[음성 텍스트 없음] 비음성 이벤트 감지: {top_event}"

        with self._text_lock:
            self._pending_text = synthetic_text
            self._pending_text_time = now

        with self._sound_lock:
            self._pending_sound_event = sound_event
            self._pending_sound_event_time = now

        logger.info(
            "[ContextLLM] 비음성 이벤트 수신: %s (%.2f), triggered=%s",
            top_event,
            top_conf,
            has_trigger,
        )

    def _detect_sound_event(self, audio: Any) -> Optional[Dict[str, Any]]:
        """YAMNet 비음성 이벤트 감지 래퍼."""
        if self._system is None:
            return None

        # 1) IntegratedMultimodalSystem helper 우선 사용
        detect_fn = getattr(self._system, "_analyze_sound_event", None)
        if callable(detect_fn):
            try:
                return detect_fn(audio)
            except Exception as e:
                logger.debug("[ContextLLM] _analyze_sound_event failed: %s", e)

        # 2) detector 직접 사용 폴백
        detector = getattr(self._system, "sound_event_detector", None)
        detect_from_audio = getattr(detector, "detect_from_audio", None) if detector else None
        if callable(detect_from_audio):
            try:
                return detect_from_audio(audio)
            except Exception as e:
                logger.debug("[ContextLLM] sound_event_detector.detect_from_audio failed: %s", e)

        return None

    # ─── 파이프라인 실행 ───

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        멀티모달 분석 실행

        트리거 조건:
            - STT 텍스트가 감지되었을 때
            - 비음성 오디오가 YAMNet 위험 이벤트로 판정되었을 때
            - 사람 감지만으로는 분석하지 않음 (원본 contextllm 정책 유지)

        분석 방식:
            - analyze_with_image(텍스트, 프레임) → 음성+영상 통합 분석
        """
        if self._system is None:
            return {"analyzed": False, "reason": "system_not_ready"}

        frame = shared_data.get("frame")
        if frame is None:
            return {"analyzed": False, "reason": "no_frame"}
        stream = shared_data.get("stream")

        now = time.time()

        # 대기 중인 STT 텍스트/비음성 이벤트 조회
        speech_candidate = None
        with self._text_lock:
            if self._pending_text and (now - self._pending_text_time < 30):
                speech_candidate = self._pending_text

        sound_event_candidate = None
        with self._sound_lock:
            if self._pending_sound_event and (now - self._pending_sound_event_time < 30):
                sound_event_candidate = self._pending_sound_event

        has_speech = speech_candidate is not None
        has_sound_event = sound_event_candidate is not None
        yolo_result = shared_data.get("results", {}).get("yolo", {}) or {}
        if not yolo_result:
            with self._yolo_lock:
                yolo_result = dict(self._latest_yolo_result) if self._latest_yolo_result else {}
        has_person = yolo_result.get("person_detected", False)

        if not has_speech and not has_sound_event:
            return {
                "analyzed": False,
                "reason": "no_trigger",
                "display_result": self.get_display_result(),
            }

        # 쿨다운 체크 (비음성 triggered 이벤트는 즉시 처리)
        urgent_sound_trigger = bool(has_sound_event and sound_event_candidate.get("triggered", False))
        if now - self._last_analysis_time < self.analysis_cooldown and not urgent_sound_trigger:
            return {
                "analyzed": False,
                "reason": "cooldown",
                "last_result": self._last_result,
                "display_result": self.get_display_result(),
            }

        if self._analysis_queue.full():
            logger.warning("[ContextLLM] 분석 큐가 가득 찼습니다. 이번 프레임 분석은 건너뜁니다.")
            return {
                "analyzed": False,
                "reason": "queue_full",
                "display_result": self.get_display_result(),
            }

        # 소비 후보 추출 (큐 적재 실패 시 복구 가능하도록 임시 보관)
        speech_text = None
        if has_speech:
            with self._text_lock:
                if self._pending_text and (now - self._pending_text_time < 30):
                    speech_text = self._pending_text
                    self._pending_text = None

        sound_event = None
        if has_sound_event:
            with self._sound_lock:
                if self._pending_sound_event and (now - self._pending_sound_event_time < 30):
                    sound_event = self._pending_sound_event
                    self._pending_sound_event = None

        had_real_speech = bool(speech_text)
        has_sound_source = sound_event is not None

        if not speech_text and sound_event:
            top_event = sound_event.get("top_event") or "unknown_sound"
            speech_text = f"[음성 텍스트 없음] 비음성 이벤트 감지: {top_event}"

        trigger_source = (
            "speech+sound_event" if (had_real_speech and has_sound_source)
            else ("speech" if had_real_speech else "sound_event")
        )

        try:
            # mic_context_fusion 방식: DOA가 최신이면 PTZ를 먼저 정렬
            doa = self._resolve_recent_doa(now)

            # 비동기 큐에 작업 넣기
            frame_snapshot = frame
            try:
                if hasattr(frame, "copy"):
                    frame_snapshot = frame.copy()
            except Exception:
                frame_snapshot = frame

            task = {
                "frame": frame_snapshot, # 프레임 복사본 사용 (메인 스레드에서 변경될 수 있음)
                "stream": stream,
                "speech_text": speech_text,
                "has_person": has_person,
                "yolo_result": yolo_result,
                "sound_event": sound_event,
                "trigger_source": trigger_source,
                "doa": doa,
                "timestamp": now
            }
            
            self._analysis_queue.put(task, block=False)
            self._last_analysis_time = now
            
            # 처리 중 상태 바로 반환 (동기 블로킹 해제)
            with self._display_lock:
                self._display_result["analyzed"] = True
                self._display_result["priority"] = self._display_result.get("priority", "PROCESSING")
                self._display_result["situation_type"] = "Thinking..."
                self._display_result["urgency"] = "Analyzing Context"
            
            logger.info(f"[ContextLLM] 분석 작업 큐에 추가됨 (현재 대기열: {self._analysis_queue.qsize()})")

            return {
                "analyzed": True,
                "reason": "queued",
                "display_result": self.get_display_result(),
            }

        except queue.Full:
            # 큐 포화 경쟁 상황에서 소비한 이벤트를 최대한 복구
            if speech_text:
                with self._text_lock:
                    if not self._pending_text:
                        self._pending_text = speech_text
                        self._pending_text_time = now
            if sound_event:
                with self._sound_lock:
                    if self._pending_sound_event is None:
                        self._pending_sound_event = sound_event
                        self._pending_sound_event_time = now
            logger.warning("[ContextLLM] 분석 큐가 가득 차서 요청을 거절했습니다.")
            return {"analyzed": False, "reason": "queue_full"}
        except Exception as e:
            logger.error(f"[ContextLLM] 분석 오류 (큐 삽입 실패): {e}")
            return {"analyzed": False, "error": str(e)}

    def _analysis_worker(self):
        """백그라운드에서 LLM 분석을 수행하는 워커 스레드"""
        logger.info("[ContextLLM] 비동기 LLM 분석 워커 스레드 시작")
        while True:
            try:
                task = self._analysis_queue.get()
                if task is None: # 종료 시그널
                    break
                    
                frame = task["frame"]
                stream = task["stream"]
                speech_text = task["speech_text"]
                has_person = task["has_person"]
                yolo_result = task["yolo_result"]
                sound_event = task["sound_event"]
                trigger_source = task["trigger_source"]
                doa = task["doa"]
                now = task["timestamp"]
                
                # PTZ 정렬 및 프레임 캡처를 백그라운드 스레드에서 수행하여 메인 루프 블로킹을 막음
                if doa is not None and self.ptz is not None:
                    try:
                        from integrated_system_process.modules.ptz_controller import PTZPriority

                        self.ptz.request_move(
                            pan=float(doa),
                            tilt=-15,
                            priority=PTZPriority.YOLO_TRACKING,
                            owner=self.name,
                            move_type="absolute",
                        )
                        if self.ptz_settle_seconds > 0:
                            time.sleep(self.ptz_settle_seconds)
                    except Exception as e:
                        logger.debug("[ContextLLM] PTZ pre-align failed: %s", e)

                # PTZ 정렬 후 최신 프레임 재캡처 (가능할 때)
                fresh_frame = self._capture_frame_with_retry(stream, max_retry=10, retry_interval=0.1)
                if fresh_frame is not None:
                    frame = fresh_frame

                doa_object_mapping = self._compute_doa_object_mapping(
                    doa=doa,
                    frame=frame,
                    yolo_result=yolo_result,
                )

                # ★ 핵심: MultimodalAnalyzer로 직접 영상+텍스트 통합 분석 (동기 호출이지만 스레드 내부라 메인 스레드 안전) ★
                logger.info(f"[ContextLLM] 백그라운드 멀티모달 분석 시작 (trigger: {trigger_source})")
                
                result = self._analyze_multimodal(
                    frame,
                    speech_text,
                    has_person,
                    yolo_result=yolo_result,
                    sound_event=sound_event,
                    doa_object_mapping=doa_object_mapping,
                )

                if result and result.get("success"):
                    self._last_result = result
                    analysis = result.get("multimodal_analysis", {})
                    priority = analysis.get("priority", "LOW")
                    is_emergency = analysis.get("is_emergency", False)
                    situation = analysis.get("situation_type", analysis.get("situation", "N/A"))
                    urgency = analysis.get("urgency", "N/A")

                    # 디스플레이용 결과 갱신
                    display = {
                        "analyzed": True,
                        "priority": priority,
                        "is_emergency": is_emergency,
                        "situation": situation,
                        "situation_type": situation,
                        "urgency": urgency,
                        "speech_text": speech_text,
                        "has_person": has_person,
                        "yolo_count": yolo_result.get("count", 0),
                        "yolo_result": yolo_result,
                        "doa_sector": doa,
                        "doa_object_mapping": doa_object_mapping,
                        "trigger_source": trigger_source,
                        "sound_event": sound_event,
                        "timestamp": time.time(),
                        "summary": analysis.get("summary", ""),
                    }
                    with self._display_lock:
                        self._display_result = display

                    self._push_contextllm_dashboard(
                        speech_text=speech_text,
                        analysis=analysis,
                        priority=priority,
                        urgency=urgency,
                        is_emergency=is_emergency,
                        trigger_source=trigger_source,
                        sound_event=sound_event,
                    )

                    # 이벤트 발행
                    self.emit("llm.analysis_complete", {
                        "result": analysis,
                        "priority": priority,
                        "is_emergency": is_emergency,
                        "speech_text": speech_text,
                        "doa_sector": doa,
                        "doa_object_mapping": doa_object_mapping,
                        "yolo_result": yolo_result,
                        "trigger_source": trigger_source,
                        "sound_event": sound_event,
                    })

                    logger.info(
                        "[ContextLLM] 분석 완료: trigger=%s, priority=%s, urgency=%s, situation=%s",
                        trigger_source,
                        priority,
                        urgency,
                        situation,
                    )

                    if is_emergency:
                        self.emit("llm.emergency", {
                            "result": analysis,
                            "urgency": urgency,
                            "situation": situation,
                        }, priority=2)
                else:
                    logger.warning("[ContextLLM] API 분석 결과가 올바르지 않거나 실패했습니다.")

            except Exception as e:
                logger.error(f"[ContextLLM] 백그라운드 워커 실행 오류: {e}")
            finally:
                self._analysis_queue.task_done()

    def _push_contextllm_dashboard(
        self,
        speech_text: Optional[str],
        analysis: Dict[str, Any],
        priority: str,
        urgency: str,
        is_emergency: bool,
        trigger_source: str,
        sound_event: Optional[Dict[str, Any]],
    ) -> None:
        if not self.contextllm_dashboard_push_enabled:
            return
        if requests is None:
            if not self._dashboard_push_warned:
                logger.warning("[ContextLLM] requests 미설치로 dashboard push 비활성화")
                self._dashboard_push_warned = True
            return

        analysis_payload = dict(analysis or {})
        analysis_payload.setdefault("priority", priority)
        analysis_payload.setdefault("urgency", urgency)
        analysis_payload.setdefault("is_emergency", bool(is_emergency))

        payload = {
            "success": True,
            "transcribed_text": speech_text or "",
            "trigger_source": trigger_source,
            "sound_event": sound_event,
            "multimodal_analysis": analysis_payload,
            "voice_characteristics": {},
        }

        try:
            response = requests.post(
                self.contextllm_dashboard_url,
                json=payload,
                timeout=self.contextllm_dashboard_timeout,
            )
            if response.status_code >= 400:
                if not self._dashboard_push_warned:
                    logger.warning(
                        "[ContextLLM] dashboard push 실패(status=%s): %s",
                        response.status_code,
                        self.contextllm_dashboard_url,
                    )
                    self._dashboard_push_warned = True
            elif self._dashboard_push_warned:
                logger.info("[ContextLLM] dashboard push 복구됨")
                self._dashboard_push_warned = False
        except Exception as exc:
            if not self._dashboard_push_warned:
                logger.warning("[ContextLLM] dashboard push 오류: %s", exc)
                self._dashboard_push_warned = True

    def _analyze_multimodal(
        self,
        frame,
        speech_text: Optional[str],
        has_person: bool,
        yolo_result: Optional[Dict[str, Any]] = None,
        sound_event: Optional[Dict[str, Any]] = None,
        doa_object_mapping: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """
        ★ MultimodalAnalyzer.analyze_with_image() 직접 호출 ★

        기존의 _analyze_frame()은 IntegratedMultimodalSystem.analyze_once()를 호출했는데,
        이는 자체 마이크로 음성을 대기하는 블로킹 함수라 파이프라인에 부적합했음.

        수정: multimodal_analyzer.analyze_with_image(text, frame)을 직접 호출하여
        외부에서 전달된 프레임 + STT 텍스트로 즉시 분석.
        """
        try:
            if not self._multimodal_analyzer:
                logger.warning("[ContextLLM] MultimodalAnalyzer 없음")
                return None

            # 분석 텍스트 결정
            if speech_text:
                analysis_text = speech_text
                context = f"[통합 분석] 음성 입력과 CCTV 영상을 종합하여 상황을 판단하세요. 사람 감지: {'예' if has_person else '아니오'}"
            else:
                analysis_text = "현재 상황을 분석해 주세요. 위험하거나 긴급한 상황인지 판단해 주세요."
                context = "[영상 분석] 실제 음성 입력 없이 영상만 분석. 영상에서 보이는 상황을 객관적으로 분석하세요."

            context_parts = [context]
            yolo_ctx = self._format_yolo_context(yolo_result)
            if yolo_ctx:
                context_parts.append(yolo_ctx)
            sound_ctx = self._format_sound_event_context(sound_event)
            if sound_ctx:
                context_parts.append(sound_ctx)
            doa_obj_ctx = self._format_doa_object_context(doa_object_mapping)
            if doa_obj_ctx:
                context_parts.append(doa_obj_ctx)

            # Downsampling 적용
            if hasattr(self._system, 'downsampler'):
                frame = self._system.downsampler.downsample_image(frame)

            # ★ analyze_with_image 호출 ★
            multimodal_result = self._multimodal_analyzer.analyze_with_image(
                audio_text=analysis_text,
                image_source=frame,
                additional_context="\n\n".join(context_parts),
            )

            return {
                "success": True,
                "multimodal_analysis": multimodal_result,
                "speech_text": speech_text,
                "sound_event": sound_event,
                "doa_object_mapping": doa_object_mapping,
            }

        except Exception as e:
            logger.error(f"[ContextLLM] 멀티모달 분석 오류: {e}")
            return None

    @staticmethod
    def _format_sound_event_context(sound_event: Optional[Dict[str, Any]]) -> str:
        if not sound_event:
            return ""

        top_event = sound_event.get("top_event")
        top_conf = float(sound_event.get("top_confidence", 0.0) or 0.0)
        triggered = bool(sound_event.get("triggered", False))
        emergency_events = sound_event.get("emergency_events", []) or []

        lines = ["[비음성 사운드 이벤트(YAMNet)]"]
        if top_event:
            lines.append(f"- 최상위 이벤트: {top_event} ({top_conf:.2f})")
        if emergency_events:
            compact = ", ".join(
                f"{evt.get('label', 'unknown')}({float(evt.get('confidence', 0.0) or 0.0):.2f})"
                for evt in emergency_events[:3]
            )
            lines.append(f"- 위험 후보: {compact}")
        lines.append(f"- 트리거 여부: {'예' if triggered else '아니오'}")
        return "\n".join(lines)

    def _resolve_recent_doa(self, now: Optional[float] = None) -> Optional[float]:
        current_time = now if now is not None else time.time()
        with self._state_lock:
            doa = self._last_doa_sector
            doa_age = current_time - self._last_doa_time
        if doa is None:
            return None
        if doa_age > self.max_doa_age_seconds:
            return None
        return float(doa)

    @staticmethod
    def _capture_frame_with_retry(stream: Any, max_retry: int = 10, retry_interval: float = 0.1):
        if stream is None:
            return None
        get_frame = getattr(stream, "get_frame", None)
        if not callable(get_frame):
            return None
        for _ in range(max_retry):
            frame = get_frame()
            if frame is not None:
                return frame
            time.sleep(retry_interval)
        return None

    @staticmethod
    def _extract_center_x(obj: Dict[str, Any]) -> Optional[float]:
        center = obj.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 1:
            try:
                return float(center[0])
            except (TypeError, ValueError):
                pass

        box = obj.get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                x1 = float(box[0])
                x2 = float(box[2])
                return (x1 + x2) / 2.0
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float(angle) % 360.0

    @staticmethod
    def _angular_distance(a: float, b: float) -> float:
        diff = (a - b + 180.0) % 360.0 - 180.0
        return abs(diff)

    def _compute_doa_object_mapping(
        self,
        doa: Optional[float],
        frame: Any,
        yolo_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.doa_object_mapping_enabled or doa is None:
            return None
        if frame is None or not hasattr(frame, "shape"):
            return None

        frame_width = int(frame.shape[1]) if len(frame.shape) >= 2 else 0
        if frame_width <= 0:
            return None

        objects = (yolo_result or {}).get("objects", []) or []
        if not objects:
            return {
                "enabled": True,
                "doa_angle": float(doa),
                "camera_hfov_deg": float(self.camera_hfov_deg),
                "object_count": 0,
                "near_object_detected": False,
                "nearest_object": None,
            }

        nearest: Optional[Dict[str, Any]] = None
        half_w = frame_width / 2.0
        for obj in objects:
            center_x = self._extract_center_x(obj)
            if center_x is None:
                continue

            relative_ratio = (center_x - half_w) / float(frame_width)
            relative_angle = relative_ratio * float(self.camera_hfov_deg)
            estimated_angle = self._normalize_angle(float(doa) + relative_angle)
            angle_diff = self._angular_distance(estimated_angle, float(doa))

            obj_entry = {
                "name": obj.get("name", "unknown"),
                "id": obj.get("permanent_id"),
                "confidence": float(obj.get("confidence", 0.0) or 0.0),
                "center_x": float(center_x),
                "estimated_angle": float(estimated_angle),
                "angle_diff_from_doa": float(angle_diff),
            }
            if nearest is None or obj_entry["angle_diff_from_doa"] < nearest["angle_diff_from_doa"]:
                nearest = obj_entry

        if nearest is None:
            return None

        near_object_detected = nearest["angle_diff_from_doa"] <= float(self.doa_object_match_threshold_deg)
        return {
            "enabled": True,
            "doa_angle": float(doa),
            "camera_hfov_deg": float(self.camera_hfov_deg),
            "object_count": len(objects),
            "threshold_deg": float(self.doa_object_match_threshold_deg),
            "near_object_detected": bool(near_object_detected),
            "nearest_object": nearest,
        }

    def _format_doa_object_context(self, mapping: Optional[Dict[str, Any]]) -> str:
        if not mapping:
            return ""

        lines: List[str] = ["[DOA-객체 각도 매핑]"]
        lines.append(f"- DOA 기준 각도: {float(mapping.get('doa_angle', 0.0)):.1f}도")
        lines.append(f"- 카메라 수평 시야각(FOV): {float(mapping.get('camera_hfov_deg', self.camera_hfov_deg)):.1f}도")
        lines.append(f"- 객체 수: {int(mapping.get('object_count', 0) or 0)}")

        nearest = mapping.get("nearest_object")
        if isinstance(nearest, dict):
            obj_name = nearest.get("name", "unknown")
            obj_id = nearest.get("id", "N/A")
            est_angle = float(nearest.get("estimated_angle", 0.0) or 0.0)
            diff = float(nearest.get("angle_diff_from_doa", 0.0) or 0.0)
            lines.append(f"- 최근접 객체: {obj_name} (ID:{obj_id})")
            lines.append(f"- 최근접 객체 추정 각도: {est_angle:.1f}도")
            lines.append(f"- DOA와 각도 차: {diff:.1f}도")
            lines.append(
                f"- DOA 근접 객체 탐지 여부: {'예' if mapping.get('near_object_detected', False) else '아니오'}"
            )
        else:
            lines.append("- 최근접 객체: 없음")
            lines.append("- DOA 근접 객체 탐지 여부: 아니오")

        return "\n".join(lines)

    @staticmethod
    def _format_yolo_context(yolo_result: Optional[Dict[str, Any]]) -> str:
        if not yolo_result:
            return ""

        objects = yolo_result.get("objects", []) or []
        person_detected = bool(yolo_result.get("person_detected", False))
        count = int(yolo_result.get("count", len(objects) if objects else 0) or 0)
        mode = yolo_result.get("mode", "N/A")

        lines: List[str] = ["[YOLO 객체 탐지 컨텍스트]"]
        lines.append(f"- 사람 감지: {'예' if person_detected else '아니오'}")
        lines.append(f"- 총 객체 수: {count}")
        lines.append(f"- 추적 모드: {mode}")

        if objects:
            top_items = objects[:5]
            compact = ", ".join(
                f"{obj.get('name', 'unknown')}({float(obj.get('confidence', 0.0) or 0.0):.2f})"
                for obj in top_items
            )
            lines.append(f"- 주요 객체: {compact}")

        target = yolo_result.get("target")
        if isinstance(target, dict):
            target_name = target.get("display_name") or target.get("name") or "unknown"
            target_id = target.get("permanent_id", "N/A")
            lines.append(f"- 추적 대상: {target_name} (ID:{target_id})")

        return "\n".join(lines)

    def get_display_result(self) -> Dict:
        """현재 디스플레이용 분석 결과 반환 (스레드 안전)"""
        with self._display_lock:
            return dict(self._display_result)

    def analyze_with_text(self, frame, text: str) -> Dict:
        """텍스트와 함께 수동 분석 (외부 호출용)"""
        with self._text_lock:
            self._pending_text = text
            self._pending_text_time = time.time()
        result = self.process({"frame": frame, "results": {"yolo": {"person_detected": True}}})
        return result

    def shutdown(self) -> None:
        if hasattr(self, '_analysis_queue'):
            # 큐가 가득 찬 상태에서도 종료가 블로킹되지 않도록 처리
            try:
                self._analysis_queue.put_nowait(None)  # 워커 스레드 종료 시그널
            except queue.Full:
                try:
                    self._analysis_queue.get_nowait()
                    self._analysis_queue.task_done()
                except Exception:
                    pass
                try:
                    self._analysis_queue.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
                
        self._system = None
        self._multimodal_analyzer = None
        logger.info("[ContextLLM] 종료")
