import time

import cv2


def _handle_key(key_code, ptz_controller) -> bool:
    if key_code == ord("q"):
        return False
    if key_code == ord("w"):
        ptz_controller.start_continuous_move(pan_velocity=0.0, tilt_velocity=0.5)
    elif key_code == ord("s"):
        ptz_controller.start_continuous_move(pan_velocity=0.0, tilt_velocity=-0.5)
    elif key_code == ord("a"):
        ptz_controller.start_continuous_move(pan_velocity=-0.5, tilt_velocity=0.0)
    elif key_code == ord("d"):
        ptz_controller.start_continuous_move(pan_velocity=0.5, tilt_velocity=0.0)
    elif key_code == ord(" "):
        ptz_controller.stop_move()
    return True


def run_monitoring(rtsp_url, detector, ptz_controller, window_name, skip_frames):
    if not rtsp_url:
        print("Error: RTSP_URL is not set.")
        return

    skip_frames = max(1, int(skip_frames))

    video_capture = cv2.VideoCapture(rtsp_url)
    if not video_capture.isOpened():
        print("Error: Cannot open video stream.")
        video_capture.release()
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Controls: [W/A/S/D] Move, [Space] Stop, [Q] Quit")
    print(f"YOLO detection: Every {skip_frames} frames (for better performance)")

    frame_count = 0
    last_results = None
    fps = 0.0
    frames_since_last = 0
    last_fps_time = time.perf_counter()

    try:
        while True:
            is_frame_read, frame = video_capture.read()

            if not is_frame_read:
                print("Video stream disconnected.")
                break

            frame_count += 1
            frames_since_last += 1
            fps_updated = False
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = frames_since_last / elapsed
                frames_since_last = 0
                last_fps_time = now
                fps_updated = True

            if frame_count % skip_frames == 0:
                last_results = detector.infer(frame)

            if last_results is not None:
                annotated_frame = last_results[0].plot()
            else:
                annotated_frame = frame

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if fps_updated:
                print(f"FPS: {fps:.1f}")

            cv2.imshow(window_name, annotated_frame)

            key_code = cv2.waitKey(1) & 0xFF
            if not _handle_key(key_code, ptz_controller):
                break
    finally:
        ptz_controller.stop_move()
        video_capture.release()
        cv2.destroyAllWindows()
