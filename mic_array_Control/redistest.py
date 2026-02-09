import cv2, time, threading, requests, redis, json, usb.core, collections
import numpy as np
import os
from requests.auth import HTTPDigestAuth
from tuning import Tuning
import datetime

# --- 설정 ---
RTSP_URL = "rtsp://admin:saloris4321@192.168.0.60:554/Streaming/Channels/101"
CAM_IP = "192.168.0.60"
AUTH = HTTPDigestAuth("admin", "saloris4321")

# Redis 연결
r = redis.Redis(host='localhost', port=6379, db=0)

# ReSpeaker 설정
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic_tuning = Tuning(dev)

# 전역 
status_text = "System Initializing..."
angle_history = collections.deque(maxlen=10)

def control_ptz_absolute(pan, tilt=-15):
    """카메라 제어 및 상태 발행"""
    url = f"http://{CAM_IP}/ISAPI/PTZCtrl/channels/1/absolute"
    azimuth = int(pan * 10) if pan is not None else 0
    xml_data = f"""<PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
        <AbsoluteHigh><elevation>{int(tilt*10)}</elevation><azimuth>{azimuth}</azimuth><absoluteZoom>10</absoluteZoom></AbsoluteHigh>
    </PTZData>"""
    try:
        requests.put(url, data=xml_data, auth=AUTH, timeout=1)
        # 카메라 이동 이벤트를 Redis에 알림 (대시보드용)
        r.publish('security_events', json.dumps({
            "source": "CAMERA_NODE",
            "type": "CAMERA_MOVE",
            "time_str": datetime.datetime.now().strftime("%H:%M:%S"),
            "target_pan": pan,
            "timestamp": time.time()
        }))
    except: pass

def mic_monitoring_thread():
    global status_text
    last_sector = -1
    while True:
        try:
            if mic_tuning.read('SPEECHDETECTED') == 1:
                raw_angle   = mic_tuning.read('DOAANGLE')
                angle_history.append(raw_angle)
                
                if len(angle_history) >= 5:
                    rad_angles = np.deg2rad(list(angle_history))
                    confidence = np.sqrt(np.mean(np.sin(rad_angles))**2 + np.mean(np.cos(rad_angles))**2)

                    if confidence > 0.6:
                        smooth_angle = np.rad2deg(np.arctan2(np.mean(np.sin(rad_angles)), np.mean(np.cos(rad_angles)))) % 360
                        target_sector = ((int(smooth_angle) + 15) // 30) * 30 % 360
                        
                        # 1. Redis로 데이터 전송 (대시보드행)
                        payload = {
                            "source": "MIC_NODE",
                            "type": "AUDIO_DETECTION",
                            "angle": round(smooth_angle, 1),
                            "confidence": round(float(confidence), 2),
                            "time_str": datetime.datetime.now().strftime("%H:%M:%S"),
                            "timestamp": time.time()
                        }
                        r.publish('security_events', json.dumps(payload))
                        
                        # 2. 카메라 이동 제어
                        if target_sector != last_sector:
                            status_text = f"Moving to {target_sector} deg"
                            control_ptz_absolute(pan=target_sector)
                            last_sector = target_sector
            time.sleep(0.05)
        except: pass

def start_system():
    # 영점 조절
    control_ptz_absolute(pan=0, tilt=0)
    
    # 마이크 감시 시작
    threading.Thread(target=mic_monitoring_thread, daemon=True).start()

    # 영상 스트리밍
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" # 환경 변수 설정
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = cv2.resize(frame, (800, 600))
        cv2.putText(display_frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Smart AI Security Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_system()