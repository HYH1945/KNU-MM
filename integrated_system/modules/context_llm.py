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

이벤트 발행:
    - llm.analysis_complete : 분석 완료 시
    - llm.emergency         : 긴급 상황 판정 시 (priority=2)

이벤트 구독:
    - yolo.person_detected  : 사람 감지 시 자동 분석 트리거
    - mic.speech_detected   : 음성 감지 시 분석 트리거
    - stt.text_recognized   : STT 텍스트 수신 → 음성+영상 통합 분석
"""

import sys
import os
import time
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from integrated_system.core.base_module import BaseModule
from integrated_system.core.event_bus import EventBus, Event
from integrated_system.core.module_loader import (
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
        (사람 감지만으로는 분석하지 않음 — 원본 contextllm과 동일한 동작)
    """

    ANALYSIS_COOLDOWN = 5.0  # 분석 쿨다운 (초)

    def __init__(
        self,
        event_bus: EventBus,
        model: str = "gpt-4o-mini",
        config_path: Optional[str] = None,
    ):
        super().__init__(event_bus)
        self.model = model
        self.config_path = config_path or str(Path(CONTEXTLLM_DIR) / "config" / "config.yaml")

        self._system = None          # IntegratedMultimodalSystem 인스턴스
        self._multimodal_analyzer = None  # MultimodalAnalyzer 직접 참조
        self._last_analysis_time = 0.0
        self._last_result: Dict = {}

        # STT 연동 상태
        self._pending_text: Optional[str] = None
        self._pending_text_time: float = 0.0
        self._text_lock = threading.Lock()

        # 최신 분석 결과 (OpenCV 오버레이용)
        self._display_result: Dict = {}
        self._display_lock = threading.Lock()

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
            self._event_bus.subscribe("mic.speech_detected", self._on_speech_detected)
            self._event_bus.subscribe("stt.text_recognized", self._on_stt_text)

            logger.info("[ContextLLM] 초기화 완료 (원본 IntegratedMultimodalSystem 사용)")
            if self._multimodal_analyzer:
                logger.info("[ContextLLM] MultimodalAnalyzer 연결됨 (음성+영상 통합 분석 가능)")
            else:
                logger.warning("[ContextLLM] MultimodalAnalyzer 없음 (openai 미설치?)")
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

    def _on_speech_detected(self, event: Event) -> None:
        """마이크 음성 감지 이벤트 핸들러 (DOA 정보)"""
        logger.debug(f"[ContextLLM] 음성 감지 이벤트 수신 (angle: {event.data.get('raw_angle', '?')})")

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

    # ─── 파이프라인 실행 ───

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        멀티모달 분석 실행

        트리거 조건:
            - STT에서 텍스트(대화)가 감지되었을 때만 분석 실행
            - 사람 감지만으로는 분석하지 않음 (원본 contextllm과 동일)

        분석 방식:
            - analyze_with_image(텍스트, 프레임) → 음성+영상 통합 분석
        """
        if self._system is None:
            return {"analyzed": False, "reason": "system_not_ready"}

        # 쿨다운 체크
        now = time.time()
        if now - self._last_analysis_time < self.ANALYSIS_COOLDOWN:
            return {
                "analyzed": False,
                "reason": "cooldown",
                "last_result": self._last_result,
                "display_result": self.get_display_result(),
            }

        frame = shared_data.get("frame")
        if frame is None:
            return {"analyzed": False, "reason": "no_frame"}

        # 대기 중인 STT 텍스트 꺼내기
        speech_text = None
        with self._text_lock:
            if self._pending_text and (now - self._pending_text_time < 30):
                speech_text = self._pending_text
                self._pending_text = None  # 소비

        # ★ 트리거: 대화(STT 텍스트)가 감지된 경우에만 분석 ★
        has_speech = speech_text is not None
        yolo_result = shared_data.get("results", {}).get("yolo", {})
        has_person = yolo_result.get("person_detected", False)

        if not has_speech:
            return {
                "analyzed": False,
                "reason": "no_speech",
                "display_result": self.get_display_result(),
            }

        try:
            self._last_analysis_time = now

            # ★ 핵심: MultimodalAnalyzer로 직접 영상+텍스트 통합 분석 ★
            result = self._analyze_multimodal(frame, speech_text, has_person)

            if result and result.get("success"):
                self._last_result = result
                analysis = result.get("multimodal_analysis", {})
                priority = analysis.get("priority", "LOW")
                is_emergency = analysis.get("is_emergency", False)
                situation = analysis.get("situation_type", analysis.get("situation", "N/A"))
                urgency = analysis.get("urgency", "N/A")

                # 디스플레이용 결과 갱신
                display = {
                    "priority": priority,
                    "is_emergency": is_emergency,
                    "situation": situation,
                    "urgency": urgency,
                    "speech_text": speech_text,
                    "has_person": has_person,
                    "timestamp": now,
                    "summary": analysis.get("summary", ""),
                }
                with self._display_lock:
                    self._display_result = display

                # 이벤트 발행
                self.emit("llm.analysis_complete", {
                    "result": analysis,
                    "priority": priority,
                    "is_emergency": is_emergency,
                    "speech_text": speech_text,
                })

                if is_emergency:
                    self.emit("llm.emergency", {
                        "result": analysis,
                        "urgency": urgency,
                        "situation": situation,
                    }, priority=2)

                return {
                    "analyzed": True,
                    "priority": priority,
                    "is_emergency": is_emergency,
                    "situation_type": situation,
                    "urgency": urgency,
                    "speech_text": speech_text,
                    "analysis": analysis,
                    "display_result": display,
                }

            return {"analyzed": False, "reason": "analysis_failed"}

        except Exception as e:
            logger.error(f"[ContextLLM] 분석 오류: {e}")
            return {"analyzed": False, "error": str(e)}

    def _analyze_multimodal(
        self,
        frame,
        speech_text: Optional[str],
        has_person: bool,
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

            # Downsampling 적용
            if hasattr(self._system, 'downsampler'):
                frame = self._system.downsampler.downsample_image(frame)

            # ★ analyze_with_image 호출 ★
            multimodal_result = self._multimodal_analyzer.analyze_with_image(
                audio_text=analysis_text,
                image_source=frame,
                additional_context=context,
            )

            return {
                "success": True,
                "multimodal_analysis": multimodal_result,
                "speech_text": speech_text,
            }

        except Exception as e:
            logger.error(f"[ContextLLM] 멀티모달 분석 오류: {e}")
            return None

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
        self._system = None
        self._multimodal_analyzer = None
        logger.info("[ContextLLM] 종료")
