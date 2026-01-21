#!/usr/bin/env python3
"""
REST API 서버로 실행 가능한 음성 인식 엔진
다른 프로그램에서 HTTP 요청으로 접근 가능
"""

from flask import Flask, jsonify, request
import os
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# 설정 (현재 폴더 기준 상대 경로)
VENV_PYTHON = "./.venv/bin/python3"
WHISPER_SCRIPT = "./whisper_service.py"
RECORDING_DIR = "./recordings"
RESULTS_DIR = "./transcriptions"

# 상태
is_monitoring = False
latest_result = None

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """음성 파일 변환"""
    data = request.get_json()
    audio_file = data.get('audio_file')
    
    if not audio_file or not os.path.exists(audio_file):
        return jsonify({'error': '파일을 찾을 수 없음'}), 400
    
    try:
        result = subprocess.run(
            [VENV_PYTHON, WHISPER_SCRIPT, audio_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            text = result.stdout.strip()
            save_to_file(text)
            return jsonify({'success': True, 'text': text})
        else:
            return jsonify({'error': result.stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitor/start', methods=['POST'])
def start_monitor():
    """자동 모니터링 시작"""
    global is_monitoring
    
    if is_monitoring:
        return jsonify({'error': '이미 실행 중'}), 400
    
    is_monitoring = True
    
    def monitor_loop():
        while is_monitoring:
            try:
                # sox로 10초 녹음
                audio_file = f"{RECORDING_DIR}/auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                subprocess.run([
                    'sox', '-d', audio_file,
                    'rate', '16000',
                    'channels', '1',
                    'trim', '0', '10'
                ], check=True, capture_output=True)
                
                # 변환
                result = subprocess.run(
                    [VENV_PYTHON, WHISPER_SCRIPT, audio_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    text = result.stdout.strip()
                    save_to_file(text)
                
                os.remove(audio_file)
            except:
                pass
    
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': '모니터링 시작'})

@app.route('/api/monitor/stop', methods=['POST'])
def stop_monitor():
    """자동 모니터링 중지"""
    global is_monitoring
    is_monitoring = False
    return jsonify({'success': True, 'message': '모니터링 중지'})

@app.route('/api/monitor/status', methods=['GET'])
def monitor_status():
    """모니터링 상태"""
    return jsonify({'is_monitoring': is_monitoring, 'latest_result': latest_result})

@app.route('/api/results/latest', methods=['GET'])
def get_latest():
    """최신 결과"""
    if latest_result:
        return jsonify(latest_result)
    return jsonify({'error': '결과 없음'}), 404

def save_to_file(text):
    """결과 저장"""
    global latest_result
    
    timestamp = datetime.now().isoformat()
    date_folder = f"{RESULTS_DIR}/{datetime.now().strftime('%Y-%m-%d')}"
    Path(date_folder).mkdir(exist_ok=True)
    
    entry = {
        "timestamp": timestamp,
        "text": text,
        "model": "whisper-base",
        "language": "ko"
    }
    
    # 누적
    json_file = f"{date_folder}/transcriptions.json"
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.append(entry)
    else:
        data = [entry]
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    latest_result = entry

if __name__ == '__main__':
    Path(RECORDING_DIR).mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    print("🚀 REST API 서버 시작 (http://localhost:5000)")
    app.run(host='127.0.0.1', port=5000, debug=False)
