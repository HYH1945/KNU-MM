#!/usr/bin/env python3
"""
통합 멀티모달 시스템 테스트
음성 감지 → 음성 특성 분석 + 영상 캡처/분석 동시 수행

사용법:
    python tests/test_integrated_multimodal.py
    
테스트 항목:
    1. 다운샘플링 기능 테스트
    2. 단발성 분석 테스트 (음성 감지 → 영상 캡처)
    3. 연속 모니터링 테스트
    4. 비디오 소스 테스트 (웹캠, 파일, 네트워크, 테스트셋)
    5. 테스트셋 분석 테스트
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv

# .env 파일 로드 (config 폴더에서)
env_path = Path(__file__).parent.parent / 'config' / '.env'
load_dotenv(env_path)

import cv2
import numpy as np


def test_downsampling():
    """다운샘플링 기능 테스트"""
    print("\n" + "=" * 60)
    print("1️⃣  다운샘플링 기능 테스트")
    print("=" * 60)
    
    from core.integrated_multimodal_system import (
        VideoDownsampler, 
        DownsamplingConfig
    )
    
    # 설정
    config = DownsamplingConfig(
        max_image_size=320,
        jpeg_quality=70,
        video_fps=2.0,
        max_video_frames=5,
        video_resolution_scale=0.5
    )
    
    downsampler = VideoDownsampler(config)
    
    # 테스트 이미지 생성 (1920x1080)
    print("\n📸 테스트 이미지 다운샘플링...")
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"   원본 크기: {test_image.shape}")
    
    downsampled = downsampler.downsample_image(test_image)
    print(f"   다운샘플링 후: {downsampled.shape}")
    
    # JPEG 인코딩 테스트
    jpeg_bytes = downsampler.encode_frame_to_jpeg(downsampled)
    print(f"   JPEG 크기: {len(jpeg_bytes) / 1024:.1f} KB")
    
    # 여러 프레임 다운샘플링 테스트
    print("\n📹 테스트 비디오 프레임 다운샘플링...")
    test_frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(20)]
    print(f"   원본: {len(test_frames)}개 프레임, 각 {test_frames[0].shape}")
    
    downsampled_frames, _ = downsampler.downsample_video_frames(test_frames)
    print(f"   다운샘플링 후: {len(downsampled_frames)}개 프레임, 각 {downsampled_frames[0].shape}")
    
    print("\n✅ 다운샘플링 테스트 완료!")


def test_video_capture():
    """비디오 캡처 테스트"""
    print("\n" + "=" * 60)
    print("2️⃣  비디오 캡처 테스트")
    print("=" * 60)
    
    from core.integrated_multimodal_system import VideoCaptureManager
    
    manager = VideoCaptureManager(camera_id=0)
    
    if not manager.open():
        print("❌ 카메라를 열 수 없습니다. 테스트 건너뜀.")
        return False
    
    # 단일 프레임 캡처
    print("\n📷 단일 프레임 캡처...")
    frame = manager.capture_frame()
    if frame is not None:
        print(f"   ✅ 프레임 캡처 성공: {frame.shape}")
    else:
        print("   ❌ 프레임 캡처 실패")
    
    # 비디오 세그먼트 캡처 (3초)
    print("\n📹 비디오 세그먼트 캡처 (3초, 2fps)...")
    frames, timestamps = manager.capture_video_segment(duration=3.0, target_fps=2.0)
    print(f"   ✅ {len(frames)}개 프레임 캡처됨")
    print(f"   타임스탬프: {[f'{t:.2f}s' for t in timestamps]}")
    
    manager.close()
    
    print("\n✅ 비디오 캡처 테스트 완료!")
    return True


def test_speech_detection():
    """음성 감지 테스트"""
    print("\n" + "=" * 60)
    print("3️⃣  음성 감지 테스트")
    print("=" * 60)
    
    from core.integrated_multimodal_system import SpeechDetector
    
    try:
        detector = SpeechDetector()
        
        print("\n🎤 음성 감지 테스트 (5초 타임아웃)...")
        print("   말씀해 주세요!")
        
        audio, detected = detector.listen_for_speech(timeout=5)
        
        if detected:
            print("   ✅ 음성 감지됨!")
            
            # 음성 인식
            text = detector.recognize_speech(audio)
            if text:
                print(f"   📝 인식된 텍스트: {text}")
            
            # 오디오 저장
            audio_path = f"recordings/test_audio_{datetime.now().strftime('%H%M%S')}.wav"
            detector.save_audio_to_wav(audio, audio_path)
            print(f"   💾 오디오 저장: {audio_path}")
        else:
            print("   ⚠️  음성이 감지되지 않았습니다 (타임아웃)")
        
        print("\n✅ 음성 감지 테스트 완료!")
        return True
    
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False


def test_single_analysis():
    """단발성 분석 테스트"""
    print("\n" + "=" * 60)
    print("4️⃣  단발성 멀티모달 분석 테스트")
    print("=" * 60)
    
    from core.integrated_multimodal_system import (
        IntegratedMultimodalSystem,
        DownsamplingConfig
    )
    
    # 성능 최적화 설정
    config = DownsamplingConfig(
        max_image_size=640,
        jpeg_quality=75,
        video_fps=2.0,
        max_video_frames=10,
        video_resolution_scale=0.5,
        video_capture_duration=5.0
    )
    
    system = IntegratedMultimodalSystem(
        camera_id=0,
        model="gpt-4o-mini",
        downsampling_config=config
    )
    
    print("\n🎙️  음성이 감지되면 자동으로 영상을 캡처하고 분석합니다.")
    print("   말씀해 주세요! (최대 30초)")
    
    result = system.analyze_once(phrase_time_limit=30.0)
    
    # 결과 출력
    print("\n" + "-" * 40)
    print("📊 분석 결과:")
    print("-" * 40)
    
    if result.get("success"):
        print(f"✅ 성공!")
        print(f"   음성 텍스트: {result.get('transcribed_text', 'N/A')}")
        
        voice = result.get("voice_characteristics")
        if voice:
            indicators = voice.get("emergency_indicators", {})
            print(f"   음성 긴급도: {indicators.get('overall_score', 0):.0%}")
        
        analysis = result.get("multimodal_analysis")
        if analysis:
            print(f"   상황: {analysis.get('situation', 'N/A')[:50]}...")
            print(f"   위급도: {analysis.get('urgency', 'N/A')}")
            print(f"   우선순위: {analysis.get('priority', 'N/A')}")
    else:
        print(f"❌ 실패: {result.get('error', '알 수 없는 오류')}")
    
    # 카메라 정리
    system.video_manager.close()
    
    print("\n✅ 단발성 분석 테스트 완료!")


def test_continuous_monitoring():
    """연속 모니터링 테스트 (3회)"""
    print("\n" + "=" * 60)
    print("5️⃣  연속 모니터링 테스트 (3회)")
    print("=" * 60)
    
    from core.integrated_multimodal_system import (
        IntegratedMultimodalSystem,
        DownsamplingConfig
    )
    
    config = DownsamplingConfig(
        max_image_size=640,
        video_fps=2.0,
        max_video_frames=10,
        video_capture_duration=5.0
    )
    
    system = IntegratedMultimodalSystem(
        camera_id=0,
        model="gpt-4o-mini",
        downsampling_config=config
    )
    
    # 결과 콜백
    results = []
    def on_result(result):
        results.append(result)
        print(f"   📊 분석 완료 ({len(results)}회차)")
    
    print("\n🔄 3회 연속 모니터링 시작...")
    print("   각 회차마다 음성을 말씀해 주세요!")
    
    try:
        system.start_monitoring(on_result=on_result, max_iterations=3)
    except KeyboardInterrupt:
        print("\n   ⏹️  사용자가 중단함")
    
    # 최종 결과 요약
    print("\n" + "=" * 40)
    print("📊 최종 결과 요약")
    print("=" * 40)
    print(f"   총 분석 횟수: {len(results)}")
    print(f"   성공: {sum(1 for r in results if r.get('success'))}")
    print(f"   실패: {sum(1 for r in results if not r.get('success'))}")
    
    print("\n✅ 연속 모니터링 테스트 완료!")


def test_video_sources():
    """다양한 비디오 소스 테스트"""
    print("\n" + "=" * 60)
    print("6️⃣  비디오 소스 테스트")
    print("=" * 60)
    
    from core.integrated_multimodal_system import (
        WebcamVideoSource,
        FileVideoSource,
        NetworkVideoSource,
        TestsetVideoSource,
        create_video_source
    )
    
    # 1. 웹캠 소스
    print("\n📹 1. 웹캠 소스 테스트")
    webcam = create_video_source("webcam", camera_id=0)
    if webcam.open():
        frame = webcam.capture_frame()
        if frame is not None:
            print(f"   ✅ 프레임 캡처 성공: {frame.shape}")
        info = webcam.get_info()
        print(f"   ℹ️  정보: {info}")
        webcam.close()
    else:
        print("   ⚠️  웹캠을 열 수 없습니다")
    
    # 2. 파일 소스 (테스트 이미지/비디오가 있는 경우)
    print("\n📁 2. 파일 소스 테스트")
    testsets_dir = Path(__file__).parent.parent / "testsets"
    test_files = list(testsets_dir.glob("*.*"))
    
    if test_files:
        test_file = test_files[0]
        print(f"   테스트 파일: {test_file.name}")
        
        file_source = create_video_source("file", file_path=str(test_file))
        if file_source.open():
            frame = file_source.capture_frame()
            if frame is not None:
                print(f"   ✅ 프레임 캡처 성공: {frame.shape}")
            info = file_source.get_info()
            print(f"   ℹ️  정보: {info}")
            file_source.close()
    else:
        print("   ⚠️  testsets/ 폴더에 테스트 파일이 없습니다")
        print("      테스트할 이미지나 비디오를 testsets/ 폴더에 추가해주세요")
    
    # 3. 테스트셋 소스
    print("\n📂 3. 테스트셋 소스 테스트")
    if test_files:
        testset = create_video_source("testset", folder_path=str(testsets_dir))
        if testset.open():
            files = testset.list_files()
            print(f"   📋 파일 목록: {files}")
            
            frame = testset.capture_frame()
            if frame is not None:
                print(f"   ✅ 프레임 캡처 성공: {frame.shape}")
            
            info = testset.get_info()
            print(f"   ℹ️  현재 파일: {info.get('current_file')}")
            
            testset.close()
    else:
        print("   ⚠️  testsets/ 폴더에 테스트 파일이 없습니다")
    
    # 4. 네트워크 카메라 (실제 URL이 있는 경우만)
    print("\n🌐 4. 네트워크 카메라 테스트")
    print("   ℹ️  네트워크 카메라 테스트는 실제 URL이 필요합니다")
    print("   예시: rtsp://192.168.1.100:554/stream")
    print("   예시: http://192.168.1.100:8080/video")
    
    test_url = input("   URL 입력 (건너뛰려면 Enter): ").strip()
    if test_url:
        network = create_video_source("network", url=test_url)
        if network.open():
            frame = network.capture_frame()
            if frame is not None:
                print(f"   ✅ 프레임 캡처 성공: {frame.shape}")
            network.close()
        else:
            print("   ❌ 연결 실패")
    
    print("\n✅ 비디오 소스 테스트 완료!")


def test_testset_analysis():
    """테스트셋 분석 테스트 (영상만, 음성 없이)"""
    print("\n" + "=" * 60)
    print("7️⃣  테스트셋 분석 테스트 (음성 입력 없이)")
    print("=" * 60)
    
    from core.integrated_multimodal_system import (
        IntegratedMultimodalSystem,
        DownsamplingConfig
    )
    
    testsets_dir = Path(__file__).parent.parent / "testsets"
    test_files = list(testsets_dir.glob("*.*"))
    
    if not test_files:
        print("\n❌ testsets/ 폴더에 테스트 파일이 없습니다")
        print("   테스트할 이미지나 비디오를 testsets/ 폴더에 추가해주세요")
        return
    
    config = DownsamplingConfig(
        max_image_size=640,
        video_fps=2.0,
        max_video_frames=10,
        video_capture_duration=5.0
    )
    
    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=config
    )
    
    # 테스트셋 설정
    system.use_testset(str(testsets_dir))
    
    # 파일 목록 출력
    files = system.get_testset_files()
    print(f"\n📋 테스트셋 파일 목록 ({len(files)}개):")
    for i, f in enumerate(files):
        print(f"   {i}: {f}")
    
    print("\n테스트 옵션:")
    print("  1. 특정 파일 하나만 분석")
    print("  2. 모든 파일 순차 분석")
    
    choice = input("\n선택 (1/2): ").strip()
    
    if choice == "1":
        idx = input(f"파일 번호 (0-{len(files)-1}): ").strip()
        try:
            idx = int(idx)
            if 0 <= idx < len(files):
                system.select_testset_file(idx)
                
                print("\n💡 프롬프트 예시:")
                print("   - 현재 상황을 분석해 주세요.")
                print("   - 폭행이나 위험한 상황인지 확인해 주세요.")
                print("   - 화재나 긴급 상황인지 판단해 주세요.")
                print("   - 도와주세요! (긴급 상황 시뮬레이션)")
                text = input("\n분석에 사용할 텍스트 (Enter = 기본값): ").strip()
                result = system.analyze_video_only(text if text else None)
                
                print("\n" + "-" * 40)
                print("📊 분석 결과:")
                print("-" * 40)
                
                if result.get("success"):
                    analysis = result.get("multimodal_analysis", {})
                    print(f"✅ 성공!")
                    print(f"   상황: {analysis.get('situation', 'N/A')}")
                    print(f"   상황 유형: {analysis.get('situation_type', 'N/A')}")
                    print(f"   위급도: {analysis.get('urgency', 'N/A')}")
                    print(f"   우선순위: {analysis.get('priority', 'N/A')}")
                    print(f"   긴급 상황: {analysis.get('is_emergency', False)}")
                    if analysis.get('is_emergency'):
                        print(f"   긴급 사유: {analysis.get('emergency_reason', 'N/A')}")
                else:
                    print(f"❌ 실패: {result.get('error', '알 수 없는 오류')}")
            else:
                print("잘못된 번호입니다")
        except ValueError:
            print("숫자를 입력해주세요")
    
    elif choice == "2":
        print("\n💡 프롬프트 예시:")
        print("   - 현재 상황을 분석해 주세요.")
        print("   - 폭행이나 위험한 상황인지 확인해 주세요.")
        print("   - 화재나 긴급 상황인지 판단해 주세요.")
        print("   - 도움이 필요한 상황인가요?")
        text = input("\n분석에 사용할 텍스트 (Enter = 기본값): ").strip()
        results = system.analyze_testset_all(text if text else None)
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 전체 결과 요약")
        print("=" * 60)
        
        for r in results:
            filename = r.get("filename", "N/A")
            success = r.get("success", False)
            
            if success:
                analysis = r.get("multimodal_analysis", {})
                priority = analysis.get("priority", "N/A")
                urgency = analysis.get("urgency", "N/A")
                is_emergency = analysis.get("is_emergency", False)
                
                status = "🚨 긴급" if is_emergency else "✅"
                print(f"   {status} {filename}: {priority} / {urgency}")
            else:
                print(f"   ❌ {filename}: 실패 - {r.get('error', '?')}")
    
    system.video_manager.close()
    print("\n✅ 테스트셋 분석 테스트 완료!")


def test_with_specific_file():
    """특정 파일로 테스트 (경로 직접 입력)"""
    print("\n" + "=" * 60)
    print("8️⃣  특정 파일 테스트")
    print("=" * 60)
    
    from core.integrated_multimodal_system import (
        IntegratedMultimodalSystem,
        DownsamplingConfig
    )
    
    file_path = input("\n파일 경로를 입력하세요: ").strip()
    
    if not file_path or not Path(file_path).exists():
        print("❌ 파일을 찾을 수 없습니다")
        return
    
    config = DownsamplingConfig(
        max_image_size=640,
        video_fps=2.0,
        max_video_frames=10,
        video_capture_duration=5.0
    )
    
    system = IntegratedMultimodalSystem(
        model="gpt-4o-mini",
        downsampling_config=config
    )
    
    # 파일 소스 설정
    system.use_file(file_path)
    
    print("\n💡 프롬프트 예시:")
    print("   - 현재 상황을 분석해 주세요.")
    print("   - 폭행이나 위험한 상황인지 확인해 주세요.")
    print("   - 살려주세요! 도와주세요! (긴급 상황 시뮬레이션)")
    text = input("\n분석에 사용할 텍스트 (Enter = 기본값): ").strip()
    result = system.analyze_video_only(text if text else None)
    
    print("\n" + "-" * 40)
    print("📊 분석 결과:")
    print("-" * 40)
    
    if result.get("success"):
        analysis = result.get("multimodal_analysis", {})
        print(f"✅ 성공!")
        print(f"   상황: {analysis.get('situation', 'N/A')}")
        print(f"   상황 유형: {analysis.get('situation_type', 'N/A')}")
        print(f"   위급도: {analysis.get('urgency', 'N/A')}")
        print(f"   우선순위: {analysis.get('priority', 'N/A')}")
        print(f"   긴급 상황: {analysis.get('is_emergency', False)}")
        print(f"   권장 조치: {analysis.get('action', 'N/A')}")
    else:
        print(f"❌ 실패: {result.get('error', '알 수 없는 오류')}")
    
    system.video_manager.close()
    print("\n✅ 파일 테스트 완료!")


def main():
    """메인 테스트 함수"""
    print("=" * 70)
    print("🚀 통합 멀티모달 시스템 테스트")
    print("=" * 70)
    print("\n이 테스트는 다음을 검증합니다:")
    print("  1. 이미지/비디오 다운샘플링 기능")
    print("  2. 비디오 캡처 기능")
    print("  3. 음성 감지 및 인식 기능")
    print("  4. 단발성 멀티모달 분석")
    print("  5. 연속 모니터링")
    print("  6. 비디오 소스 (웹캠/파일/네트워크/테스트셋)")
    print("  7. 테스트셋 분석 (음성 없이 영상만)")
    print("  8. 특정 파일 테스트")
    
    print("\n테스트 선택:")
    print("  1. 다운샘플링 테스트만")
    print("  2. 비디오 캡처 테스트만")
    print("  3. 음성 감지 테스트만")
    print("  4. 단발성 분석 테스트 (음성 + 영상)")
    print("  5. 연속 모니터링 테스트 (3회)")
    print("  6. 비디오 소스 테스트")
    print("  7. 테스트셋 분석 테스트 ⭐")
    print("  8. 특정 파일 테스트 ⭐")
    print("  9. 전체 테스트 (1-3)")
    
    choice = input("\n선택 (1-9): ").strip()
    
    if choice == "1":
        test_downsampling()
    
    elif choice == "2":
        test_video_capture()
    
    elif choice == "3":
        test_speech_detection()
    
    elif choice == "4":
        test_single_analysis()
    
    elif choice == "5":
        test_continuous_monitoring()
    
    elif choice == "6":
        test_video_sources()
    
    elif choice == "7":
        test_testset_analysis()
    
    elif choice == "8":
        test_with_specific_file()
    
    elif choice == "9":
        test_downsampling()
        if test_video_capture():
            test_speech_detection()
    
    else:
        print("잘못된 선택입니다. 1-9 중 선택하세요.")
    
    print("\n" + "=" * 70)
    print("🎉 테스트 종료")
    print("=" * 70)


if __name__ == "__main__":
    main()
