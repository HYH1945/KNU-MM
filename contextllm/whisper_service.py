# whisper_service.py
import whisper
import sys
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

def transcribe(file_path):
    # 모델 로드 (속도를 위해 'base'나 'turbo' 추천)
    # 처음 실행 시 모델 다운로드 시간이 걸립니다.
    model = whisper.load_model("base") 
    result = model.transcribe(file_path, language="ko")
    print(result["text"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe(sys.argv[1])