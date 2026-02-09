import redis, json, time, usb.core, collections
import numpy as np
from tuning import Tuning

r = redis.Redis(host='localhost', port=6379, db=0)
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic_tuning = Tuning(dev)
angle_history = collections.deque(maxlen=10)

print(">>> [Mic Node] Starting Audio Producer...")

while True:
    try:
        if mic_tuning.read('SPEECHDETECTED') == 1:
            raw_angle = mic_tuning.read('DOAANGLE')
            angle_history.append(raw_angle)
            
            if len(angle_history) >= 5:
                rad_angles = np.deg2rad(list(angle_history))
                confidence = np.sqrt(np.mean(np.sin(rad_angles))**2 + np.mean(np.cos(rad_angles))**2)

                if confidence > 0.6:
                    smooth_angle = np.rad2deg(np.arctan2(np.mean(np.sin(rad_angles)), np.mean(np.cos(rad_angles)))) % 360
                    
                    # 데이터 규격화 (팀원들과 공유할 포맷)
                    payload = {
                        "source": "MIC_ARRAY",
                        "type": "AUDIO_DETECTION",
                        "angle": round(smooth_angle, 1),
                        "confidence": round(float(confidence), 2),
                        "timestamp": time.time()
                    }
                    r.publish('security_events', json.dumps(payload))
        time.sleep(0.05)
    except Exception as e:
        print(f"Mic Error: {e}")