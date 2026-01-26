import time
from tuning import Tuning
import usb.core
import usb.util

# ReSpeaker v3.0의 Vendor ID와 Product ID (일반적으로 0x2886, 0x0018)
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if dev:
    mic_tuning = Tuning(dev)
    print("음향 탐지 시작")
    
    try:
        while True:
            angle = mic_tuning.direction #소리 발생 각도 (0~359)
           
            is_voice = mic_tuning.is_voice() # 소리 감지 여부 (1: 감지, 0: 대기)
            
            if is_voice:
                print(f"소리 발생 위치: {angle}도")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass