import requests
from requests.auth import HTTPDigestAuth
import time

# --- 카메라 접속 정보 (사용자 환경에 맞게 수정) ---
CAM_IP = "192.168.0.60"
CAM_USER = "admin"
CAM_PASS = "saloris4321"
AUTH = HTTPDigestAuth(CAM_USER, CAM_PASS)

def test_camera_ptz(pan_angle, tilt_angle=0):
    """
    카메라를 입력받은 Pan/Tilt 각도로 즉시 이동
    """
    url = f"http://{CAM_IP}/ISAPI/PTZCtrl/channels/1/absolute"
    
    # [가설] 하이크비전은 0.1도 단위를 사용함 (예: 90도 이동 시 900 입력)
    azimuth = int(pan_angle * 10)
    elevation = int(tilt_angle * 10)
    
    xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
    <PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
        <AbsoluteHigh>
            <elevation>{elevation}</elevation>
            <azimuth>{azimuth}</azimuth>
            <absoluteZoom>10</absoluteZoom>
        </AbsoluteHigh>
    </PTZData>"""

    try:
        response = requests.put(url, data=xml_data, auth=AUTH, timeout=3)
        if response.status_code == 200:
            print(f">>> [성공] {pan_angle}도 방향으로 명령 전송 완료")
        else:
            print(f">>> [실패] 응답 코드: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f">>> [에러] 연결 실패: {e}")

if __name__ == "__main__":
    print("--- Hikvision PTZ Calibration Test ---")
    print("0을 입력하면 정면(영점)으로 이동합니다.")
    print("'q'를 입력하면 종료합니다.\n")
    
    while True:
        user_input = input("이동할 Pan 각도 입력 (0~359): ")
        
        if user_input.lower() == 'q':
            break
            
        try:
            target_pan = float(user_input)
            test_camera_ptz(target_pan)
        except ValueError:
            print("숫자를 입력해 주세요.")