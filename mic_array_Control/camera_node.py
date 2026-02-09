import redis, json, requests, time
from requests.auth import HTTPDigestAuth

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CAM_IP = "192.168.0.60"
AUTH = HTTPDigestAuth("admin", "saloris4321")



def get_ptz_status():
    """카메라의 현재 PTZ 상태를 가져와 Redis에 발행"""
    url = f"http://{CAM_IP}/ISAPI/PTZCtrl/channels/1/status"
    try:
        response = requests.get(url, auth=AUTH, timeout=1)
        if response.status_code == 200:
            # XML에서 각도 추출 (단순 문자열 파싱 혹은 xml.etree 사용)
            # <azimuth>1234</azimuth> 형태임 (123.4도)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            # 네임스페이스가 있을 수 있으므로 주의
            ns = {'ns': 'http://www.hikvision.com/ver20/XMLSchema'}
            azimuth = int(root.find('.//ns:azimuth', ns).text) / 10.0
            
            status_payload = {
                "source": "CAMERA_NODE",
                "type": "CAMERA_STATUS",
                "current_pan": azimuth,
                "timestamp": time.time()
            }
            r.publish('security_events', json.dumps(status_payload))
    except Exception as e:
        print(f"상태 가져오기 실패: {e}")

def control_ptz(pan):
    url = f"http://{CAM_IP}/ISAPI/PTZCtrl/channels/1/absolute"
    azimuth = int(pan * 10)
    xml_data = f"""<PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
        <AbsoluteHigh>
            <elevation>-150</elevation>
            <azimuth>{azimuth}</azimuth>
            <absoluteZoom>10</absoluteZoom>
        </AbsoluteHigh>
    </PTZData>"""
    try:
        requests.put(url, data=xml_data, auth=AUTH, timeout=1)
        # 카메라 이동 명령 후 약간의 딜레이 뒤에 상태 업데이트
        time.sleep(0.5) 
        get_ptz_status()
    except Exception as e:
        print(f"제어 실패: {e}")

print(">>> [Cam Node] Waiting for Redis Events...")

pubsub = r.pubsub()
pubsub.subscribe('security_events')

for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        if data['type'] == 'AUDIO_DETECTION':
            target_angle = ((int(data['angle']) + 15) // 30) * 30 % 360
            print(f"[*] Moving Camera to {target_angle} deg")
            control_ptz(target_angle)