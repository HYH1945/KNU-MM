import cv2
import time
import threading
import requests
from requests.auth import HTTPDigestAuth
import usb.core
from tuning import Tuning
import speech_recognition as sr

# --- 설정 구간 ---
RTSP_URL = "rtsp://admin:saloris4321@192.168.0.60:554/Streaming/Channels/101"
CAM_IP = "192.168.0.60"
CAM_USER = "admin"
CAM_PASS = "saloris4321"
AUTH = HTTPDigestAuth(CAM_USER, CAM_PASS)

# ReSpeaker 설정
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic_tuning = Tuning(dev)
recognizer = sr.Recognizer()

# 전역 변수 (이벤트 텍스트 표시용)
status_text = "Monitoring..."

def control_ptz_absolute(angle):
    """
    ReSpeaker의 0~359도 값을 Hikvision의 절대 좌표(Azimuth)로 매칭
    하이크비전은 보통 0.1도 단위를 사용하므로 각도에 10을 곱함
    """
    url = f"http://{CAM_IP}/ISAPI/PTZCtrl/channels/1/absolute"
    # ReSpeaker와 카메라의 0도 방향이 일치하도록 offset을 조절할 수 있습니다.
    target_azimuth = int(angle * 10) 
    
    xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
    <PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
        <AbsoluteHigh>
            <elevation>0</elevation>
            <azimuth>{target_azimuth}</azimuth>
            <absoluteZoom>10</absoluteZoom>
        </AbsoluteHigh>
    </PTZData>"""

    try:
        requests.put(url, data=xml_data, auth=AUTH, timeout=2)
        print(f"[명령] 카메라를 {angle}도 방향으로 이동")
    except Exception as e:
        print(f"PTZ 제어 실패: {e}")

def mic_monitoring_thread():
    """음성 감지 및 STT를 처리하는 별도 스레드"""
    global status_text
    print(">>> 음성 감지 스레드 시작")
    
    while True:
        try:
            # 목소리 감지
            if mic_tuning.read('SPEECHDETECTED') == 1:
                angle = mic_tuning.read('DOAANGLE')
                status_text = f"Voice Detected at {angle} deg!"
                
                # 즉시 카메라 회전
                control_ptz_absolute(angle)
                
                # STT 분석 (녹음 중에는 영상이 멈추지 않음)
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, phrase_time_limit=3)
                    
                try:
                    result = recognizer.recognize_google(audio, language='ko-KR')
                    status_text = f"STT: {result}"
                    
                    if any(word in result for word in ["불", "화재", "도와", "살려"]):
                        status_text = f"!! EMERGENCY: {result} !!"
                except:
                    status_text = "STT Failed"
                
                time.sleep(1) 
            time.sleep(0.1)
        except Exception as e:
            print(f"마이크 에러: {e}")

def start_system():
    global status_text
    
    # 1. 마이크 감시 스레드 실행
    mic_thread = threading.Thread(target=mic_monitoring_thread, daemon=True)
    mic_thread.start()

    # 2. 메인 스레드: 영상 스트리밍
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("카메라 연결 실패")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.open(RTSP_URL)
            continue

        # 영상 리사이즈 및 상태 텍스트 출력
        display_frame = cv2.resize(frame, (800, 600))
        cv2.putText(display_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("AI Security Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mic_tuning.close()

if __name__ == "__main__":
    start_system()