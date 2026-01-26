import cv2
import time
import threading
import requests
from requests.auth import HTTPDigestAuth
import usb.core
from tuning import Tuning
import speech_recognition as sr
import collections
import numpy as np

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

# 전역 변수
status_text = "System Initializing..."
angle_history = collections.deque(maxlen=10)

def initialize_system():
    """
    초기 셋업 함수:
    마이크가 인식하는 각도와 카메라가 인식하는 각도를 맞추기 위한
    카메라 영점 조절
    & 초기 파라미터 설정 (민감도 설정)
    """
    global status_text
    print(">>> [Step 1] 카메라 영점 조절 시작 ")
    status_text = "Calibrating: Moving to 0 deg..."
    
    # 카메라를 물리적 0도, 수평으로 이동
    control_ptz_absolute(pan=0, tilt=0)

    # mic array 민감도 설정
    mic_tuning.write('AGCMAXGAIN', 15.0)    # 증폭을 줄여서 잡음 유입 차단
    mic_tuning.write('GAMMAVAD_SR', 10.0)   # VAD 문턱값을 높여서 확실한 소리에만 반응


    print(">>> [Step 2] 마이크 정면(MIC1)을 카메라 렌즈와 일직선으로 맞춰주세요.")
    time.sleep(3) # 정렬할 시간 부여
    status_text = "System Ready: Monitoring..."

def get_sector_angle(raw_angle):
    """
    360도를 12개 구역(30도 단위)으로 나누는 함수
    """
    sector_angle = ((raw_angle + 15) // 30) * 30
    return sector_angle % 360

def control_ptz_absolute(pan, tilt=-15):
    """
    Hikvision ptz 카메라 제어 함수
    """
    url = f"http://{CAM_IP}/ISAPI/PTZCtrl/channels/1/absolute"
    
    # pan이 None이면(사각지대) Azimuth를 0으로 보내거나, 
    # 모델에 따라 현재 위치를 유지하도록 XML 구조를 조정해야 합니다.
    # 여기서는 안전하게 0으로 처리하거나, 필요시 Azimuth 태그를 제외할 수 있습니다.

    # 방위각 : 수평회전
    azimuth = int(pan * 10) if pan is not None else 0

    # 앙각 : 수직회전
    elevation = int(tilt * 10)
    
    # 배율 (10 : 1배율)
    absoluteZoom = 10 

    xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
    <PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
        <AbsoluteHigh>
            <elevation>{elevation}</elevation>
            <azimuth>{azimuth}</azimuth>
            <absoluteZoom>{absoluteZoom}</absoluteZoom>
        </AbsoluteHigh>
    </PTZData>"""

    try:
        requests.put(url, data=xml_data, auth=AUTH, timeout=1)
    except Exception as e:
        print(f"PTZ 제어 실패: {e}")

def mic_monitoring_thread():
    global status_text, last_camera_angle # last_camera_angle 전역 변수 필요
    last_camera_angle = -1 
    
    while True:
        try:
            if mic_tuning.read('SPEECHDETECTED') == 1:
                raw_angle = mic_tuning.read('DOAANGLE')
                gain = mic_tuning.read('AGCGAIN')
                angle_history.append(raw_angle)
                
                if len(angle_history) >= 5:
                    rad_angles = np.deg2rad(list(angle_history))
                    sin_mean = np.mean(np.sin(rad_angles))
                    cos_mean = np.mean(np.cos(rad_angles))
                    confidence = np.sqrt(sin_mean**2 + cos_mean**2)

                    # 1. 사각지대 판정
                    if confidence < 0.4 and gain < 10.0:
                        status_text = f"!! ZENITH !! (Conf:{confidence:.2f})"
                        control_ptz_absolute(pan=None, tilt=-90)
                    
                    # 2. 일반 방향 판정 (confidence: 신뢰도, 감지 음성 각도가 무작위할수록 값이 낮아짐)
                    elif confidence > 0.6:
                        # 평균 각도 계산
                        smooth_angle = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
                        target_sector = get_sector_angle(smooth_angle)
                        
                        # 다른 구역 각도 나왔을 때만 이동
                        if target_sector != last_camera_angle:
                            status_text = f"Moving to Sector {target_sector} (Original:{smooth_angle:.1f})"
                            control_ptz_absolute(pan=target_sector, tilt=-15)
                            last_camera_angle = target_sector
                        else:
                            status_text = f"Monitoring Sector {target_sector}"
                    
                    else:
                        status_text = f"Low Confidence: {confidence:.2f}"
                
            time.sleep(0.05)
        except Exception as e:
            print(f"\n마이크 스레드 에러: {e}")


def start_system():
    # 1. 카메라 영점세팅
    initialize_system()

    # 2. 마이크 감시 스레드 시작
    mic_thread = threading.Thread(target=mic_monitoring_thread, daemon=True)
    mic_thread.start()

    # 3. 영상 스트리밍 시작
    cap = cv2.VideoCapture(RTSP_URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.open(RTSP_URL)
            continue

        display_frame = cv2.resize(frame, (800, 600))
        cv2.rectangle(display_frame, (10, 10), (450, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Smart AI Security Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mic_tuning.close()

if __name__ == "__main__":
    start_system()