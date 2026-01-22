import cv2
import time

# 카메라 설정
RTSP_URL = "rtsp://admin:saloris4321@192.168.0.60:554/Streaming/Channels/101"

def start_camera_monitoring():
    # RTSP 연결
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("!!! [오류] 카메라에 연결할 수 없습니다. IP 주소나 계정 정보를 확인하세요.")
        return

    print(">>> 카메라 연결 성공. 모니터링을 시작합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("!!! [오류] 프레임을 수신할 수 없습니다. 재연결 시도 중...")
            cap.open(RTSP_URL)
            continue

        # --- 영상 처리 로직 (여기에 객체 감지나 화재 감지 모델을 넣을 수 있습니다) ---
        resized_frame = cv2.resize(frame, (800, 600))
        # 화면 출력
        cv2.imshow("Smart Security Camera", resized_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 비상 상황 시 카메라 제어 예시 함수 ---
def on_emergency_detected(event_type, angle):
    """
    음성 인식 로직에서 호출될 함수입니다.
    event_type: 'FIRE', 'RESCUE' 등
    angle: ReSpeaker에서 읽어온 DOAANGLE
    """
    print(f"\n[카메라 액션] {event_type} 감지! {angle}도 방향으로 수색 시작.")
    
    # 여기에 PTZ 제어 프로토콜(ONVIF 또는 HTTP CGI) 코드를 추가합니다.
    # 예: 하이크비전(Hikvision) 카메라는 보통 ISAPI를 사용합니다.
    # send_ptz_command(angle) 

if __name__ == "__main__":
    start_camera_monitoring()