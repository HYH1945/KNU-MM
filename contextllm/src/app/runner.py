"""Application runner for config-first ContextLLM."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

try:
    import yaml
except ImportError:
    yaml = None

from app.settings import ContextLLMSettings, VALID_MODES, load_settings, load_yaml


class ContextLLMRunner:
    """High-level runner used by CLI or external orchestrators."""

    def __init__(self, config_path: Path):
        self.config_path = config_path.expanduser().resolve()
        self.raw_config = load_yaml(self.config_path)
        self.settings: ContextLLMSettings = load_settings(self.config_path)
        self._sync_core_config()

    def _sync_core_config(self) -> None:
        # Keep core modules and runner on the same config snapshot.
        from core.config_manager import set_runtime_config

        set_runtime_config(
            self.raw_config,
            config_path=self.config_path,
            merge=False,
            load_env_file=True,
        )

    def show_config(self) -> None:
        print(f"\n📋 현재 설정 ({self.config_path}):")
        print("-" * 60)
        if yaml is not None:
            print(yaml.dump(self.raw_config, allow_unicode=True, sort_keys=False, default_flow_style=False))
        else:
            print(self.raw_config)

    def run(self, mode_override: str | None = None) -> None:
        mode = (mode_override or self.settings.mode).strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"지원하지 않는 mode: {mode}")

        web_dashboard_started = self._maybe_start_web_dashboard()
        handlers: Dict[str, Callable[[], None]] = {
            "realtime": self._run_realtime,
            "testset": self._run_testset,
            "file": self._run_file,
            "webcam": self._run_webcam,
            "network": self._run_network,
        }

        try:
            handlers[mode]()
        finally:
            if web_dashboard_started:
                self._stop_web_dashboard()

    def _create_system(self, enable_speech: bool | None = None):
        from core.integrated_multimodal_system import DownsamplingConfig, IntegratedMultimodalSystem

        ds = self.settings.downsampling
        downsampling_config = DownsamplingConfig(
            max_image_size=ds.max_image_size,
            jpeg_quality=ds.jpeg_quality,
            video_fps=ds.video_fps,
            max_video_frames=ds.max_video_frames,
            video_capture_duration=ds.video_capture_duration,
        )

        speech_enabled = self.settings.speech.enabled if enable_speech is None else enable_speech
        return IntegratedMultimodalSystem(
            camera_id=self.settings.video.camera_id,
            model=self.settings.model,
            downsampling_config=downsampling_config,
            energy_threshold=self.settings.speech.energy_threshold,
            pause_threshold=self.settings.speech.pause_threshold,
            dynamic_threshold=self.settings.speech.dynamic_threshold,
            enable_speech=speech_enabled,
        )

    def _run_monitoring(self, system) -> None:
        analysis = self.settings.analysis
        if self.settings.speech.enabled:
            system.start_monitoring(
                max_iterations=analysis.iterations,
                verbose=self.settings.logging.verbose,
                parallel=analysis.parallel,
            )
            return

        self._run_video_only_loop(system)

    def _run_video_only_loop(self, system) -> None:
        """
        speech.enabled=false 환경용 폴백.
        음성 트리거 대신 analyze_video_only()를 반복 실행한다.
        """
        iterations = self.settings.analysis.iterations
        if iterations is None:
            iterations = 1
            print("⚠️ speech.enabled=false && analysis.iterations=null -> 1회 분석으로 자동 제한")

        text_input = self.settings.analysis.default_text.strip() or None
        verbose = self.settings.logging.verbose

        for idx in range(iterations):
            result = system.analyze_video_only(text_input=text_input)
            if verbose:
                self._print_video_only_result(result, idx + 1, iterations)

    @staticmethod
    def _print_video_only_result(result: dict, idx: int, total: int) -> None:
        ok = result.get("success", False)
        analysis = result.get("multimodal_analysis", {}) or {}
        if ok:
            print(
                f"[{idx}/{total}] ✅ video-only "
                f"priority={analysis.get('priority', 'N/A')} "
                f"urgency={analysis.get('urgency', 'N/A')} "
                f"situation={analysis.get('situation_type', 'N/A')}"
            )
        else:
            print(f"[{idx}/{total}] ❌ video-only failed: {result.get('error', 'unknown')}")

    def _run_realtime(self) -> None:
        system = self._create_system()
        self._run_monitoring(system)

    def _run_testset(self) -> None:
        if self.settings.media_test.enabled:
            self._run_media_test()
            return

        testset_path = Path(self.settings.video.testset_path)
        if not testset_path.exists():
            print(f"❌ 테스트셋 폴더를 찾을 수 없습니다: {testset_path}")
            return

        system = self._create_system()
        system.use_testset(str(testset_path))
        files = system.get_testset_files()
        if not files:
            print(f"❌ 테스트셋 폴더에 파일이 없습니다: {testset_path}")
            return

        if self.settings.logging.verbose:
            print(f"\n📋 파일 목록 ({len(files)}개):")
            for idx, filename in enumerate(files):
                print(f"   {idx}: {filename}")

        if self.settings.analysis.analyze_all_testset:
            results = system.analyze_testset_all(self.settings.analysis.default_text or None)
            success_count = sum(1 for item in results if item.get("success"))
            print(f"\n📊 테스트셋 전체 분석 완료: 성공 {success_count}/{len(results)}")
            for item in results:
                filename = item.get("filename", "N/A")
                if item.get("success"):
                    analysis = item.get("multimodal_analysis", {}) or {}
                    print(f"   ✅ {filename}: {analysis.get('priority', 'N/A')}/{analysis.get('urgency', 'N/A')}")
                else:
                    print(f"   ❌ {filename}: {item.get('error', '알 수 없는 오류')}")
            return

        index = self.settings.analysis.testset_index
        if index < 0 or index >= len(files):
            print(f"❌ 잘못된 testset_index: {index} (0~{len(files)-1})")
            return

        system.select_testset_file(index)
        selected_file = files[index]
        print(f"📁 테스트 파일: {selected_file}")

        media_extensions = {
            ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
        }
        if Path(selected_file).suffix.lower() in media_extensions:
            result = system.analyze_video_only(self.settings.analysis.default_text or None)
            if result.get("success"):
                analysis = result.get("multimodal_analysis", {}) or {}
                print(
                    "✅ testset-media "
                    f"priority={analysis.get('priority', 'N/A')} "
                    f"urgency={analysis.get('urgency', 'N/A')} "
                    f"situation={analysis.get('situation_type', 'N/A')}"
                )
            else:
                print(f"❌ testset-media failed: {result.get('error', 'unknown')}")
            return

        self._run_monitoring(system)

    def _run_media_test(self) -> None:
        media = self.settings.media_test
        system = self._create_system()
        result = system.analyze_configured_media_inputs(
            image_path=media.image_path,
            video_path=media.video_path,
            audio_path=media.audio_path,
            text_input=media.text_input or None,
            phrase_time_limit=media.phrase_time_limit,
        )
        if not result.get("success"):
            print(f"❌ media_test failed: {result.get('error', 'unknown')}")
            return

        if hasattr(system, "_print_result_summary"):
            system._print_result_summary(result, verbose=self.settings.logging.verbose)

    def _run_file(self) -> None:
        file_path = self.settings.video.file_path.strip()
        if not file_path:
            print("❌ video.file_path가 비어 있습니다. config에서 설정하세요.")
            return
        if not Path(file_path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return

        system = self._create_system()
        system.use_file(file_path)
        print(f"📁 파일: {file_path}")
        self._run_monitoring(system)

    def _run_webcam(self) -> None:
        system = self._create_system()
        system.use_webcam(self.settings.video.camera_id)
        print(f"📷 웹캠: {self.settings.video.camera_id}")
        if self.settings.display.opencv_live:
            system.enable_opencv_display(True)
        self._run_monitoring(system)

    def _run_network(self) -> None:
        url = self.settings.video.network_url.strip()
        if not url:
            print("❌ video.network_url이 비어 있습니다. config에서 설정하세요.")
            print("   예: rtsp://192.168.1.100:554/stream")
            print("   예: http://192.168.1.100:8080/video")
            return

        system = self._create_system()
        system.use_network_camera(url)
        print(f"🌐 네트워크 카메라: {url}")
        if self.settings.display.opencv_live:
            system.enable_opencv_display(True)
        self._run_monitoring(system)

    def _maybe_start_web_dashboard(self) -> bool:
        if not self.settings.display.web_enabled:
            return False

        try:
            from web.app import start_dashboard

            start_dashboard(port=self.settings.display.web_port)
            print(f"🌐 웹 대시보드 시작: http://localhost:{self.settings.display.web_port}")
            return True
        except ImportError as exc:
            print(f"⚠️ 웹 대시보드를 시작할 수 없습니다: {exc}")
            print("   설치: pip install flask flask-socketio")
            return False

    @staticmethod
    def _stop_web_dashboard() -> None:
        try:
            from web.app import stop_dashboard

            stop_dashboard()
        except Exception:
            pass
