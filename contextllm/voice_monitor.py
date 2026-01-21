#!/usr/bin/env python3
"""
독립 실행형 실시간 음성 인식 데몬
10초마다 자동으로 음성을 녹음하고 변환합니다.
다른 프로그램에서도 접근 가능한 파일 기반 인터페이스 제공
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# 설정 (현재 폴더 기준 상대 경로)
VENV_PYTHON = "./.venv/bin/python3"
WHISPER_SCRIPT = "./whisper_service.py"
RECORDING_DIR = "./recordings"
RESULTS_DIR = "./transcriptions"
INTERVAL = 10  # 10초마다 녹음

# 디렉토리 생성
Path(RECORDING_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def record_audio(duration=10):
    """Sox를 사용하여 음성 녹음"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"{RECORDING_DIR}/audio_{timestamp}.wav"
    
    try:
        # sox 명령어로 10초 녹음
        subprocess.run([
            'sox', '-d', audio_file,
            'rate', '16000',
            'channels', '1',
            'trim', '0', str(duration)
        ], check=True, capture_output=True)
        
        return audio_file if os.path.getsize(audio_file) > 1000 else None
    except Exception as e:
        print(f"❌ 녹음 오류: {e}")
        return None

def transcribe_audio(audio_file):
    """Whisper로 변환"""
    try:
        result = subprocess.run(
            [VENV_PYTHON, WHISPER_SCRIPT, audio_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"❌ Whisper 오류: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ 변환 오류: {e}")
        return None

def save_results(text):
    """결과를 파일에 저장"""
    timestamp = datetime.now().isoformat()
    date_folder = f"{RESULTS_DIR}/{datetime.now().strftime('%Y-%m-%d')}"
    Path(date_folder).mkdir(exist_ok=True)
    
    # TXT 파일에 누적
    txt_file = f"{date_folder}/transcriptions.txt"
    with open(txt_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {text}\n")
    
    # JSON에 누적
    json_file = f"{date_folder}/transcriptions.json"
    entry = {
        "timestamp": timestamp,
        "text": text,
        "model": "whisper-base",
        "language": "ko"
    }
    
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.append(entry)
    else:
        data = [entry]
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 최신 결과를 latest.json에 저장 (다른 프로그램에서 쉽게 읽을 수 있도록)
    latest_file = f"{date_folder}/latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    
    print(f"✅ [{timestamp}] {text[:50]}...")

def main():
    """메인 루프"""
    print(f"🎤 음성 인식 데몬 시작 (10초 간격)")
    print(f"📁 결과 저장: {RESULTS_DIR}")
    print(f"⏹️  Ctrl+C로 종료\n")
    
    try:
        while True:
            print(f"🔴 녹음 중... ({datetime.now().strftime('%H:%M:%S')})")
            
            audio_file = record_audio(duration=INTERVAL)
            
            if audio_file:
                print(f"⚙️  변환 중...")
                text = transcribe_audio(audio_file)
                
                if text:
                    save_results(text)
                else:
                    print("⚠️  변환 실패")
                
                # 정리
                try:
                    os.remove(audio_file)
                except:
                    pass
            else:
                print("⚠️  녹음 실패")
            
            time.sleep(1)  # 1초 대기 후 다시
            
    except KeyboardInterrupt:
        print("\n\n⛔ 데몬 종료")
        sys.exit(0)

if __name__ == "__main__":
    main()
