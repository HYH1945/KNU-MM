#!/usr/bin/env python3
"""
웹 대시보드 서버
분석 결과를 실시간으로 웹 브라우저에서 확인할 수 있습니다.
"""

import os
import sys
import threading
import time
import cv2
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, 
            template_folder=str(PROJECT_ROOT / 'src' / 'web' / 'templates'),
            static_folder=str(PROJECT_ROOT / 'src' / 'web' / 'static'))

# SECRET_KEY 환경변수에서 로드 (기본값: 개발용)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# CORS 설정: localhost만 허용 (보안)
cors_origins = ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost", "http://127.0.0.1"]
socketio = SocketIO(app, cors_allowed_origins=cors_origins, async_mode='threading', 
                   ping_timeout=120, ping_interval=25)

# 전역 변수: 최근 분석 결과들
analysis_results = []
MAX_RESULTS = 50  # 최대 저장 결과 수

# 비디오 스트리밍 관련
video_frame = None
video_frame_lock = threading.Lock()
video_streaming_enabled = False


class DashboardServer:
    """웹 대시보드 서버 관리"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.server_thread = None
        self.running = False
        self.port = 5000
        self.host = '127.0.0.1'
    
    def start(self, port: int = 5000, host: str = None):
        """백그라운드에서 서버 시작 (localhost에만 바인드)"""
        if self.running:
            print(f"   ⚠️ 웹 대시보드가 이미 실행 중입니다: http://{self.host}:{self.port}")
            return
        
        # 보안: 항상 localhost(127.0.0.1)에만 바인드
        self.host = '127.0.0.1'
        self.port = port
        self.running = True
        
        def run_server():
            # 로깅 최소화
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            print(f"⚠️ 보안 알림: 대시보드는 localhost({self.host}:{self.port})에만 접근 가능합니다")
            socketio.run(app, host=self.host, port=self.port, 
                        debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        print(f"\n🌐 웹 대시보드 시작: http://{self.host}:{self.port}")
        print("   브라우저에서 열어서 분석 결과를 확인하세요!\n")
    
    def stop(self):
        """서버 중지"""
        self.running = False
        # Flask-SocketIO는 daemon thread라서 자동 종료됨
    
    def push_result(self, result: dict):
        """분석 결과를 대시보드에 푸시"""
        global analysis_results
        
        # 결과 정리 (웹 전송용)
        formatted = self._format_result(result)
        
        # 저장
        analysis_results.insert(0, formatted)
        if len(analysis_results) > MAX_RESULTS:
            analysis_results = analysis_results[:MAX_RESULTS]
        
        # 웹소켓으로 실시간 전송
        socketio.emit('new_result', formatted)
    
    def _format_result(self, result: dict) -> dict:
        """결과를 웹 표시용으로 포맷"""
        analysis = result.get('multimodal_analysis', {})
        voice = result.get('voice_characteristics', {})
        
        # 긴급도 레벨 결정 (urgency 필드 사용)
        is_emergency = analysis.get('is_emergency', False)
        urgency = analysis.get('urgency', 'LOW')  # 프롬프트는 'urgency' 필드 사용
        
        if is_emergency:
            level = 'critical'
            level_color = '#dc3545'  # 빨강
        elif urgency in ['HIGH', '높음', '긴급']:
            level = 'high'
            level_color = '#fd7e14'  # 주황
        elif urgency in ['MEDIUM', '중간']:
            level = 'medium'
            level_color = '#ffc107'  # 노랑
        else:
            level = 'low'
            level_color = '#28a745'  # 초록
        
        return {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'transcribed_text': result.get('transcribed_text', ''),
            'situation_type': analysis.get('situation_type', 'N/A'),
            # 프롬프트 필드명과 매칭
            'situation_description': analysis.get('situation', 'N/A'),  # situation -> situation_description
            'emotion': analysis.get('emotional_state', 'N/A'),  # emotional_state -> emotion
            'video_description': analysis.get('visual_content', 'N/A'),  # visual_content -> video_description
            'is_emergency': is_emergency,
            'urgency_level': urgency,
            'priority': analysis.get('priority', 'LOW'),
            'emergency_reason': analysis.get('emergency_reason', ''),
            'recommended_action': analysis.get('action', 'N/A'),  # action -> recommended_action
            'voice_video_match': analysis.get('audio_visual_consistency', 'N/A'),  # audio_visual_consistency -> voice_video_match
            'level': level,
            'level_color': level_color,
            'voice_urgency': voice.get('urgency_score', 0) if voice else 0,
            'voice_speed': voice.get('speaking_rate', 'N/A') if voice else 'N/A',
        }


# 싱글톤 인스턴스
dashboard = DashboardServer()


# Flask 라우트
@app.before_request
def add_cors_headers():
    """CORS 헤더 추가"""
    pass

@app.after_request
def set_cors_headers(response):
    """응답에 CORS 헤더 추가"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('dashboard.html')


@app.route('/api/results')
def get_results():
    """최근 분석 결과 API"""
    return jsonify(analysis_results)


@app.route('/api/clear')
def clear_results():
    """결과 초기화"""
    global analysis_results
    analysis_results = []
    socketio.emit('clear_results')
    return jsonify({'status': 'cleared'})


@app.route('/api/video_status')
def video_status():
    """비디오 스트리밍 상태 확인"""
    return jsonify({'enabled': video_streaming_enabled})


def generate_frames():
    """MJPEG 스트림 생성"""
    global video_frame
    while True:
        with video_frame_lock:
            if video_frame is not None:
                # JPEG 인코딩
                ret, buffer = cv2.imencode('.jpg', video_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30fps


@app.route('/video_feed')
def video_feed():
    """비디오 스트림 엔드포인트"""
    if not video_streaming_enabled:
        # 스트리밍 비활성화 시 빈 응답
        return Response('', status=204)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 시"""
    emit('init_results', analysis_results)
    emit('video_status', {'enabled': video_streaming_enabled})


# 외부에서 사용할 함수
def start_dashboard(port: int = 5000):
    """대시보드 시작"""
    dashboard.start(port=port)


def push_result(result: dict):
    """결과 푸시"""
    dashboard.push_result(result)


def push_frame(frame):
    """비디오 프레임 업데이트"""
    global video_frame
    with video_frame_lock:
        video_frame = frame.copy() if frame is not None else None


def enable_video_stream(enable: bool = True):
    """비디오 스트리밍 활성화/비활성화"""
    global video_streaming_enabled
    video_streaming_enabled = enable
    socketio.emit('video_status', {'enabled': enable})


def stop_dashboard():
    """대시보드 중지"""
    dashboard.stop()


if __name__ == '__main__':
    # 직접 실행 시 테스트
    print("🚀 웹 대시보드 테스트 모드")
    dashboard.start(port=5000)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n중지됨")
