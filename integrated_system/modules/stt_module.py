"""
STT (Speech-to-Text) 모듈 - 음성 인식 및 텍스트 변환

마이크에서 음성을 감지하고, Google Speech API를 통해 텍스트로 변환합니다.
변환된 텍스트는 EventBus를 통해 ContextLLM에 전달됩니다.

원본 참조:
    - contextllm/src/core/integrated_multimodal_system.py → SpeechDetector (알고리즘 참조)
    - mic_array_Control/test.py                           → recognizer 설정 참조

※ ReSpeaker 마이크 어레이가 있으면 해당 디바이스를, 없으면 시스템 기본 마이크를 사용합니다.

이벤트 발행:
    - stt.text_recognized   : 음성→텍스트 변환 완료 시
    - stt.listening_started  : 음성 대기 시작
    - stt.listening_stopped  : 음성 대기 종료

이벤트 구독:
    - mic.speech_detected    : (선택) MicArray 음성 감지 시 DOA 정보 수집
"""

import time
import threading
import logging
from typing import Dict, Any, Optional

from integrated_system.core.base_module import BaseModule
from integrated_system.core.event_bus import EventBus, Event

logger = logging.getLogger(__name__)

# speech_recognition 가용성 확인
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


class STTModule(BaseModule):
    """
    Speech-to-Text 모듈

    백그라운드에서 마이크 음성을 지속적으로 감지하고,
    Google Speech API를 통해 한국어 텍스트로 변환합니다.

    변환된 텍스트는 `stt.text_recognized` 이벤트로 발행되어
    ContextLLM 등 다른 모듈에서 활용할 수 있습니다.
    """

    def __init__(
        self,
        event_bus: EventBus,
        language: str = "ko-KR",
        energy_threshold: int = 400,
        pause_threshold: float = 3.0,
        phrase_time_limit: float = 15.0,
        dynamic_threshold: bool = True,
        min_audio_duration: float = 0.3,
        device_index: Optional[int] = None,
    ):
        """
        Args:
            event_bus: 이벤트 버스
            language: 인식 언어 (기본: 한국어)
            energy_threshold: 음성 감지 에너지 임계값 (낮을수록 민감)
            pause_threshold: 문장 끝 판단 무음 시간 (초)
            phrase_time_limit: 최대 발화 시간 (초)
            dynamic_threshold: 동적 에너지 임계값 (True=주변소음 적응)
            min_audio_duration: 최소 오디오 길이 (초, 노이즈 필터)
        """
        super().__init__(event_bus)
        self.language = language
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.phrase_time_limit = phrase_time_limit
        self.dynamic_threshold = dynamic_threshold
        self.min_audio_duration = min_audio_duration
        self.device_index = device_index

        # 런타임 상태
        self._recognizer = None
        self._microphone = None
        self._listen_thread: Optional[threading.Thread] = None
        self._running = False

        # 최근 인식 결과
        self._latest_text: Optional[str] = None
        self._latest_time: float = 0.0
        self._text_lock = threading.Lock()

        # MicArray DOA 정보 (선택적)
        self._current_doa: Optional[float] = None

    @property
    def name(self) -> str:
        return "stt"

    def initialize(self) -> bool:
        """마이크 초기화 및 주변 소음 보정"""
        if not SR_AVAILABLE:
            logger.warning(
                "[STT] speech_recognition 패키지 미설치 (비활성화)\n"
                "     설치: pip install SpeechRecognition"
            )
            return False

        try:
            self._recognizer = sr.Recognizer()
            self._recognizer.pause_threshold = self.pause_threshold
            self._recognizer.dynamic_energy_threshold = self.dynamic_threshold

            # 마이크 오픈 테스트 + 주변 소음 보정
            if self.device_index is not None:
                self._microphone = sr.Microphone(device_index=self.device_index)
            else:
                self._microphone = sr.Microphone()
            with self._microphone as source:
                logger.info("[STT] 주변 소음 보정 중 (1초)...")
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
                self._recognizer.energy_threshold = max(
                    self.energy_threshold,
                    self._recognizer.energy_threshold,
                )

            logger.info(
                f"[STT] 초기화 완료 "
                f"(언어: {self.language}, "
                f"에너지 임계값: {self._recognizer.energy_threshold:.0f}, "
                f"device_index: {self.device_index if self.device_index is not None else 'default'})"
            )

            # MicArray DOA 이벤트 구독 (선택)
            self._event_bus.subscribe("mic.doa_detected", self._on_doa_detected)

            return True

        except OSError as e:
            logger.warning(f"[STT] 마이크를 열 수 없습니다 (비활성화): {e}")
            return False
        except Exception as e:
            logger.error(f"[STT] 초기화 실패: {e}")
            return False

    def start_listening(self) -> None:
        """백그라운드 음성 인식 시작"""
        if not self._initialized or self._running:
            return

        self._running = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="STT-Listener"
        )
        self._listen_thread.start()

        self.emit("stt.listening_started", {})
        logger.info("[STT] 백그라운드 음성 인식 시작")

    def stop_listening(self) -> None:
        """백그라운드 음성 인식 중지"""
        self._running = False
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=3)

        self.emit("stt.listening_stopped", {})
        logger.info("[STT] 음성 인식 중지")

    def _listen_loop(self) -> None:
        """
        음성 인식 백그라운드 루프

        알고리즘 참조:
            - contextllm SpeechDetector.start_background_listening()
            - mic_array_Control/test.py recognizer 패턴
        """
        while self._running:
            try:
                with self._microphone as source:
                    # 음성 대기 (timeout=5초마다 재시도)
                    try:
                        audio = self._recognizer.listen(
                            source,
                            timeout=5,
                            phrase_time_limit=self.phrase_time_limit,
                        )
                    except sr.WaitTimeoutError:
                        continue

                    # 오디오 길이 확인 (노이즈 필터)
                    audio_data = audio.get_raw_data()
                    duration = len(audio_data) / (audio.sample_rate * audio.sample_width)
                    if duration < self.min_audio_duration:
                        continue

                    # Google Speech API로 텍스트 변환
                    try:
                        text = self._recognizer.recognize_google(
                            audio, language=self.language
                        )
                    except sr.UnknownValueError:
                        # 음성은 감지됐지만 인식 불가
                        logger.debug("[STT] 음성 감지됨 (인식 불가)")
                        continue
                    except sr.RequestError as e:
                        logger.error(f"[STT] Google API 오류: {e}")
                        time.sleep(2)
                        continue

                    if not text or not text.strip():
                        continue

                    # 인식 성공 → 이벤트 발행
                    now = time.time()
                    with self._text_lock:
                        self._latest_text = text
                        self._latest_time = now

                    logger.info(f'[STT] 인식: "{text}"')

                    self.emit("stt.text_recognized", {
                        "text": text,
                        "timestamp": now,
                        "duration": duration,
                        "doa_angle": self._current_doa,
                    })

            except Exception as e:
                logger.error(f"[STT] 인식 루프 오류: {e}")
                time.sleep(1)

    def _on_doa_detected(self, event: Event) -> None:
        """MicArray DOA 이벤트 수신 → 현재 방향 갱신"""
        self._current_doa = event.data.get("sector_angle")

    def get_latest_text(self) -> Optional[str]:
        """최근 인식된 텍스트 반환 (30초 이내)"""
        with self._text_lock:
            if self._latest_text and (time.time() - self._latest_time < 30):
                return self._latest_text
            return None

    def consume_text(self) -> Optional[str]:
        """최근 인식 텍스트를 가져오고 소비 (1회성)"""
        with self._text_lock:
            if self._latest_text and (time.time() - self._latest_time < 30):
                text = self._latest_text
                self._latest_text = None
                return text
            return None

    def process(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        파이프라인 호출 시 최근 STT 상태 반환.
        STT는 자체 백그라운드 스레드에서 동작합니다.
        """
        return {
            "listening": self._running,
            "latest_text": self.get_latest_text(),
            "latest_time": self._latest_time,
            "doa_angle": self._current_doa,
        }

    def shutdown(self) -> None:
        self.stop_listening()
        self._recognizer = None
        self._microphone = None
        logger.info("[STT] 종료")
