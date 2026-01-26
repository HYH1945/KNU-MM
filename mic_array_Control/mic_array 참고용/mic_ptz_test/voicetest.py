import speech_recognition as sr
from tuning import Tuning
import usb.core
import time

# 1. ReSpeaker 설정
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic_tuning = Tuning(dev)
recognizer = sr.Recognizer()


# 사람 음성이 감지되었을 때 텍스트를 감지하는 test (임시로 구글의 webspeech 사용)


def listen_and_convert():
    with sr.Microphone() as source:
        print(">>> [음성 분석 중] 말씀하세요...")
        # 주변 소음에 적응 (0.5초)
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        # 실제 음성 녹음 (최대 5초)
        audio = recognizer.listen(source, phrase_time_limit=5)
    
    try:
        # 구글 webspeech로 인식
        text = recognizer.recognize_google(audio, language='ko-KR')
        return text
    except:
        return None

print("관제 시스템 대기 중...")

try:
    while True:
        # ReSpeaker가 사람 목소리를 감지했을 때만 STT 가동
        if mic_tuning.read('SPEECHDETECTED') == 1:
            print("\n[이벤트] 사람 목소리 감지!")
            result = listen_and_convert()
            
            if result:
                print(f"[인식 결과]: {result}")
                
                # LLM 연동 포인트: "불이야" 또는 "도와주세요"가 포함되었는지 확인
                if "불" in result or "화재" in result:
                    print("!!! [비상] 화재 상황 판단, 카메라 정밀 수색 모드 전환 !!!")
                elif "도와" in result or "살려" in result:
                    print("!!! [비상] 구조 요청 감지, 관리자 호출 !!!")
            else:
                print("[알림] 음성을 인식하지 못했습니다.")
            
            time.sleep(1) # 연속 감지 방지
        
        time.sleep(0.1)

except KeyboardInterrupt:
    mic_tuning.close()