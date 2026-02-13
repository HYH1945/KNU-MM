#!/usr/bin/env python3
"""
간단한 웹캠 OpenCV 뷰어
10fps 제한 적용
"""

import cv2
import time

def main():
    # 웹캠 열기 (기본값: 0번 카메라)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다")
        return
    
    print("📹 웹캠 시작 (Ctrl+C로 종료)")
    print("   FPS 제한: 10fps\n")
    
    try:
        frame_time = 1.0 / 10  # 10fps = 0.1초
        last_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ 프레임 읽기 실패")
                break
            
            # 현재 시간
            current_time = time.time()
            elapsed = current_time - last_time
            
            # 10fps 제한 (프레임 간격이 0.1초 이상일 때만 표시)
            if elapsed >= frame_time:
                # 프레임 정보 표시
                fps = 1.0 / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 프레임 표시
                cv2.imshow("Webcam (10fps)", frame)
                last_time = current_time
                
                # 키 입력 확인 (1ms 대기)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 시간이 충분하지 않으면 짧게 대기
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⏹️  종료")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 종료됨")

if __name__ == "__main__":
    main()
