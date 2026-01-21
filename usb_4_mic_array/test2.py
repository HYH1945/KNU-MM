import time
import usb.core
import usb.util
from tuning import Tuning

# 1. 장치 찾기
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if not dev:
    print("장치를 찾을 수 없습니다.")
    exit()

mic_tuning = Tuning(dev)

####################### TEST #############################

# print("--- 정밀 관제 시스템 모니터링 시작 ---")

# try:
#     while True:
#         # 각도 데이터
#         angle = mic_tuning.direction
        
#         # 음성 활동 여부 (0 또는 1)
#         vad = mic_tuning.is_voice()
        
#         # 음성 검출 여부 (0 또는 1) - PARAMETERS에 정의된 SPEECHDETECTED 사용
#         is_speech = mic_tuning.read('SPEECHDETECTED')
        
#         # (선택) 감도 조절이 필요하다면 VAD Threshold를 읽어볼 수도 있습니다.
#         #vad_threshold = mic_tuning.read('GAMMAVAD_SR')
#         #print(vad_threshold)
#         # [논리 구조] 
#         # 1. 소리가 감지(VAD)되고 
#         # 2. 그것이 사람 목소리로 판단(SPEECHDETECTED)될 때만 이벤트 발생
#         if vad and is_speech:
#             print(f"[EVENT] 유의미한 음성 탐지")
#             print(f" > 위치: {angle}도")
#             print(f" > 상태: VAD({vad}), Speech({is_speech})")
            
#             # 여기서 PTZ 카메라 제어 함수 호출
#             # move_camera(pan=angle)
            
#         else:
#             # 대기 상태 출력 (로그가 너무 많으면 삭제 가능)
#             print(f"모니터링 중... (현재 각도: {angle})", end='\r')

#         time.sleep(0.1) # CPU 부하 감소를 위해 0.1초 대기

# except KeyboardInterrupt:
#     print("\n모니터링 종료")
# finally:
#     mic_tuning.close()

###################AGC GAIN TEST #############################

AGC_LIMIT = 15.0  # 최대 증폭, 낮을수록 민감해짐
mic_tuning.write('AGCMAXGAIN', AGC_LIMIT)
mic_tuning.write('GAMMAVAD_SR', 5.0) # 또다른 민감도 파라미터, 낮을수록 민감해짐

print(f"관제 시스템 가동 중... (기준 Gain: {AGC_LIMIT})")

try:
    while True:
        current_gain = mic_tuning.read('AGCGAIN')
        
        # 1. 소리가 감지되었는가? (VAD)
        if mic_tuning.is_voice():
            angle = mic_tuning.direction
            is_speech = mic_tuning.read('SPEECHDETECTED')
            
            # A. 사람의 음성인 경우 (화재/재난/도움요청)
            if is_speech:
                print(f"[음성 탐지] {angle}도 방향 대화/음성 감지 (Gain: {current_gain:.2f})")
                # TODO: LLM/STT 로직 연결
                
            # B. 사람이 아닌 큰 소리인 경우 (침입/파손/낙상)
            # 기준 게인보다 확실히 떨어졌을 때(소리가 클 때)만 반응
            elif current_gain < (AGC_LIMIT - 3.0): 
                print(f"[경보] 이상 소음 감지! 각도: {angle}도 (Gain: {current_gain:.2f})")
                # TODO: PTZ 카메라 해당 각도로 회전
            
            else:
                # 소리는 났지만 무의미한 작은 소음인 경우
                print(f"단순 소음 무시 중... (Gain: {current_gain:.2f})", end='\r')
        
        time.sleep(0.2) 

except KeyboardInterrupt:
    print("\n시스템 종료")
    mic_tuning.close()