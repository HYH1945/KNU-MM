import requests

# 1. 서버 주소 
SERVER_URL = "https://invocative-guttate-tammy.ngrok-free.dev/publish"

# 서버 전송 함수 예씨시
def send_angle(angle, source_name="예시) YOLO 등"):
    payload = {
        "source": source_name,
        "type": "CAMERA_MOVE", #CAMERA_MOVE 타입으로 보내면 실시간 대시보드의 카메라 각도를 해당 각도로 바꿈
        "data": {
            "angle": float(angle)
            # 라벨 추가 가능
            # "object_label": "Fire",
            # "severity": "High"
        }
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"전송 성공: {angle}도 ({source_name})")
        else:
            print(f"전송 실패: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 서버 연결 에러: {e}")

# --- 사용 예시 ---
# 180도 방향에서 객체 detection시 
send_angle(180.0, "YOLO_TEAM")

