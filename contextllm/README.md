# 🎤 로컬 Whisper + LLM 음성 인식 시스템

**macOS에서 로컬 AI를 사용한 실시간 음성 인식 및 상황 분석 시스템**

[![Python 3.10.19](https://img.shields.io/badge/Python-3.10.19-3776ab?logo=python)](https://www.python.org/downloads/release/python-31019/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🔐 **완전 로컬 실행** - 모든 데이터가 당신의 컴퓨터에서만 처리됨  
> 🚀 **프라이빗 + 빠름** - 외부 API 불필요  
> 🇰🇷 **한국어 완벽 지원** - 한국어 문맥 분석 최적화  

---

## 📋 목차

1. [프로젝트 소개](#프로젝트-소개)
2. [필수 설치 요구사항](#필수-설치-요구사항)
3. [OS별 설치 가이드](#-os별-설치-가이드)
4. [설치 방법](#설치-방법)
5. [사용 방법](#사용-방법)
6. [프로젝트 구조](#프로젝트-구조)
7. [다른 프로젝트에 이식하기](#다른-프로젝트에-이식하기)
8. [자주 묻는 질문](#자주-묻는-질문)

---

## 프로젝트 소개

이 프로젝트는 **로컬 AI 모델**을 사용하여 음성을 실시간으로 인식하고 분석하는 시스템입니다.

### 주요 기능

| 기능 | 설명 | 기술 |
|------|------|------|
| 🎵 **음성 인식** | 음성 파일을 텍스트로 변환 | OpenAI Whisper |
| 🤖 **컨텍스트 분석** | 텍스트를 분석하여 상황 파악 | Ollama + LLM |
| 💾 **자동 저장** | 모든 결과를 JSON으로 저장 | Python |
| 🔊 **실시간 모니터링** | 백그라운드에서 지속적으로 모니터링 | Threading |
| 📡 **REST API** | 다른 프로그램과 통합 가능 | Flask |

### 처리 흐름

```
🎤 음성 녹음
    ↓
📝 Whisper → 텍스트 변환
    ↓
🧠 Ollama LLM → 컨텍스트 분석
    ↓
💾 JSON 파일에 저장
    ↓
📊 결과 확인
```

---

## 필수 설치 요구사항

### 1. 시스템 요구사항

- **OS**: macOS (M1/M2/Intel)
- **Python**: 3.10.19 (다른 프로젝트와의 호환성을 위해 고정)
- **RAM**: 최소 8GB (권장 16GB)
- **디스크**: 10GB 여유 공간
- **인터넷**: 초기 모델 다운로드 시에만 필요

### 2. 외부 도구 설치

#### 🐘 Ollama (필수)

Ollama는 LLM을 로컬에서 실행하는 프레임워크입니다.

```bash
# macOS에 설치
brew install ollama

# 또는 공식 사이트에서 다운로드
# https://ollama.ai
```

**Ollama 확인:**
```bash
ollama --version
# ollama version is 0.x.x
```

#### 🔊 Sox (음성 녹음용)

```bash
# macOS에 설치
brew install sox
```

**Sox 확인:**
```bash
sox --version
# sox: SoX v14.4.2
```

### 3. Python 환경 설정

이 프로젝트는 **Python 3.10.19**를 사용합니다.

```bash
# 현재 Python 버전 확인
python3 --version
# Python 3.10.19

# Python 3.10이 없는 경우 설치
brew install python@3.10
```

---

## 🖥️ OS별 설치 가이드

### macOS (권장 - 본 가이드 기본)

위의 기본 설치 방법을 따르면 됩니다.

```bash
# 패키지 설치
brew install ollama sox python@3.10

# Homebrew 경로 확인
/opt/homebrew/bin/python3.10 --version
```

### 🪟 Windows

Windows에서 설치하려면 다음을 수정하세요:

#### 1. Ollama 설치
- https://ollama.ai 에서 Windows 버전 다운로드
- 설치 후 PowerShell 또는 CMD에서:
  ```cmd
  ollama --version
  ollama serve
  ```

#### 2. Sox 설치
Windows에서는 SoX 대신 **PyAudio** 또는 **sounddevice** 사용:

```bash
# 옵션 1: PyAudio (권장)
pip install pyaudio

# 옵션 2: sounddevice
pip install sounddevice
```

#### 3. Python 설정
```bash
# Python 3.10 설치 후
python --version
# Python 3.10.19

# 가상 환경 생성 (macOS와 동일)
python -m venv .venv

# 가상 환경 활성화 (Windows)
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

#### 4. 음성 녹음 수정
[voice_analyzer.py](voice_analyzer.py)에서 `record_audio()` 메서드를 수정:

```python
# macOS (기존)
subprocess.run(['sox', '-d', output_file, ...])

# Windows (수정)
import sounddevice as sd
import scipy.io.wavfile as wavfile

def record_audio(self, duration=10, output_file=None):
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    
    sample_rate = 16000
    print(f"🎤 녹음 중... ({duration}초)")
    
    recording = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1)
    sd.wait()
    
    wavfile.write(output_file, sample_rate, recording)
    print(f"✅ 녹음 완료: {output_file}")
    return output_file
```

#### 5. 설정 파일 수정 (voice_analyzer.py)
```python
# macOS
VENV_PYTHON = "./.venv/bin/python3"

# Windows (수정)
import sys
VENV_PYTHON = ".\\venv\\Scripts\\python.exe"  # 또는 sys.executable
```

#### 6. 실행
```bash
# PowerShell 또는 CMD
python voice_analyzer.py
```

**주의:** Windows에서는 FFmpeg도 필요할 수 있습니다:
```bash
# Chocolatey 사용
choco install ffmpeg

# 또는 수동 다운로드
# https://ffmpeg.org/download.html
```

### 🐧 Linux (Ubuntu/Debian)

Linux에서 설치하려면 다음을 수정하세요:

#### 1. 필수 패키지 설치
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv sox ffmpeg libsndfile1

# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 2. Ollama 실행
```bash
# 시스템 서비스로 실행
sudo systemctl start ollama
sudo systemctl status ollama

# 또는 수동 실행
ollama serve
```

#### 3. Python 가상 환경 생성
```bash
# Python 3.10으로 venv 생성
python3.10 -m venv .venv

# 활성화
source .venv/bin/activate

# 확인
python --version
# Python 3.10.19
```

#### 4. 패키지 설치
```bash
pip install -r requirements.txt

# 음성 처리 추가 패키지
pip install soundfile libsndfile
```

#### 5. 음성 녹음 수정 (선택사항)
Linux에서도 Sox가 기본적으로 작동하므로, 추가 수정이 필요 없습니다.

#### 6. 실행
```bash
source .venv/bin/activate
python3 voice_analyzer.py
```

**알림:** 권한 문제 발생 시:
```bash
# 마이크 권한 확인
groups $USER | grep -q audio || sudo usermod -aG audio $USER

# 재부팅 필요할 수 있음
```

### 🍎 Apple Silicon vs Intel Mac

#### Intel Mac (x86_64)
```bash
# 기본 Homebrew 사용 가능
brew install python@3.10 ollama sox

python3.10 -m venv .venv
source .venv/bin/activate
```

#### Apple Silicon Mac (M1/M2/M3, arm64)
```bash
# Apple Silicon 최적화 버전 설치
brew install python@3.10 ollama sox

# Rosetta 2 호환성 확인
arch
# arm64

# 가상 환경 생성 (자동 arm64 사용)
python3.10 -m venv .venv
source .venv/bin/activate
```

**문제 발생 시:**
```bash
# Rosetta 2 환경에서 설치 (필요한 경우)
arch -x86_64 /bin/bash
arch -x86_64 python3.10 -m venv .venv
```

---

## 설치 방법

### 📥 Step 1: 저장소 클론

```bash
# 원하는 디렉토리로 이동
cd ~/Desktop

# 저장소 클론
git clone https://github.com/YOUR_USERNAME/contextllm.git
cd contextllm
```

### 🐍 Step 2: Python 가상 환경 생성

```bash
# Python 3.10으로 가상 환경 생성
/opt/homebrew/bin/python3.10 -m venv .venv

# 가상 환경 활성화
source .venv/bin/activate

# 확인
python --version
# Python 3.10.19
```

### 📦 Step 3: 패키지 설치

```bash
# 가상 환경 활성화 상태에서
pip install -r requirements.txt

# 설치 확인 (약 2-3분 소요)
pip list | grep -E "whisper|torch|numpy"
```

### 🤖 Step 4: Ollama 모델 준비

```bash
# 터미널 1: Ollama 서버 시작 (백그라운드에서 실행)
ollama serve

# 터미널 2: 모델 다운로드 (첫 실행 시에만)
ollama pull mistral        # 약 4GB
# 또는
ollama pull neural-chat    # 더 빠른 응답

# 모델 확인
ollama list
```

**모델 선택 가이드:**

| 모델 | 크기 | 속도 | 품질 | 메모리 |
|------|------|------|------|--------|
| mistral | 4GB | 중간 | 높음 | 8GB |
| neural-chat | 3.9GB | 빠름 | 중간 | 8GB |
| openchat | 3.8GB | 빠름 | 중간 | 8GB |

---

## 사용 방법

### 🎙️ 방법 1: 기본 음성 분석 (추천)

```bash
# 가상 환경 활성화
source .venv/bin/activate

# 음성 인식 + LLM 분석
python3 voice_analyzer.py

# 출력 예:
# 🎤 녹음 중... (10초)
# ✅ 녹음 완료
# 📝 텍스트 변환 중...
# 🧠 LLM 분석 중...
# 💾 결과 저장 완료
```

**인터랙티브 사용:**

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 🎤 방식 1: 고정 시간 녹음 (15초)
result = analyzer.transcribe_and_analyze(duration=15)

print("🎤 음성:", result['transcribed_text'])
print("📊 분석 결과:")
print(f"  - 상황: {result['analysis']['situation']}")
print(f"  - 감정: {result['analysis']['emotional_state']}")
print(f"  - 위급도: {result['analysis']['urgency']}")
```

#### 🎤 특별 기능: 무한 녹음 (Enter로 종료)

기술 한계: **진정한 실시간 음성 처리는 불가능**합니다. 왜냐하면:
- Whisper는 완성된 음성 파일이 필요
- 실시간 처리하려면 별도의 스트리밍 음성인식 API 필요 (Google Speech-to-Text, Azure Speech Services 등)

대신 **Enter 키로 언제든 녹음을 중단할 수 있는 무한 녹음 모드**를 제공합니다:

```python
# 💡 Enter 키까지 무한 녹음 (10초 이상 필요 시)
result = analyzer.transcribe_and_analyze(duration=None)
# 🎤 무한 녹음 시작... (Enter 키를 누르면 종료)
# [사용자가 말함...]
# [Enter 키 누름]
# ⏹️  녹음 중지 중...
# ✅ 녹음 완료
```

#### 🚀 거의 실시간 처리: 10초 간격 반복 실행 (신기능!)

음성이 입력될 때마다 자동으로 처리하려면, **10초 간격으로 반복 실행**하세요:

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# 순차 처리: 10초 간격으로 무한 반복 (거의 실시간)
analyzer.run_continuously(interval=10)

# 병렬 처리: 진정한 실시간 (NEW!)
analyzer.run_parallel_realtime(interval=10)
```

**실행 결과 (병렬 처리 - 진정한 실시간):**
```
============================================================
⚡ 병렬 처리 실시간 모니터링 시작 (진정한 실시간!)
============================================================

[녹음 1차] 시간: 11:55:00
  🎤 10초 녹음 중...

[분석 1차] 시간: 11:55:05
  📝 Whisper 변환 중...
  ✅ 음성: "안녕하세요 날씨가 정말 좋네요"
  🚨 위급도: 낮음
  😊 감정: 긍정

[녹음 2차] 시간: 11:55:10 (동시에 진행!)
  🎤 10초 녹음 중...

[분석 2차] 시간: 11:55:15
  📝 Whisper 변환 중...
  ✅ 음성: "도와주세요 지금 긴급상황입니다"
  🚨 위급도: 긴급
  😊 감정: 부정
```

**테스트용 제한 반복:**

```python
# 순차 처리: 5회만 반복
analyzer.run_continuously(interval=10, max_iterations=5)

# 병렬 처리: 5회만 반복 (권장)
analyzer.run_parallel_realtime(interval=10, max_iterations=5)

# 또는 명령줄에서
python voice_analyzer.py
# 선택: 4 (병렬 처리 모니터링 - 진정한 실시간)
# 반복 횟수: 1 (무한 반복)
```

**6가지 사용 모드:**

| 모드 | 코드 | 특징 | 지연시간 |
|------|------|------|---------|
| 고정 시간 | `duration=10` | 정확히 10초 녹음 | 즉시 |
| 무한 녹음 | `duration=None` | Enter 키 눌 때까지 | 즉시 |
| 순차 처리 | `run_continuously(interval=10)` | 거의 실시간 | ~15-20초 |
| **병렬 처리 ⭐** | **`run_parallel_realtime(interval=10)`** | **진정한 실시간 (병렬)** | **~10초** |
| 테스트 (순차) | `run_continuously(max_iterations=5)` | 순차 제한 반복 | ~15-20초 |
| 테스트 (병렬) | `run_parallel_realtime(max_iterations=5)` | 병렬 제한 반복 | ~10초 |

**순차 vs 병렬 처리 비교:**

```
🔴 순차 처리 (기존)
════════════════════════════════════
녹음(10s) → 변환(2-5s) → 분석(2-3s) → 녹음(10s) → ...
총: 15-20초 간격

🟢 병렬 처리 (신규) ⭐
════════════════════════════════════
스레드1: 녹음1 → 녹음2 → 녹음3 → ...
스레드2:      변환1+분석1 → 변환2+분석2 → ...

결과: 약 10초 간격 (진정한 거의 실시간!)
```

**거의 실시간 처리의 특징:**

✅ **병렬 처리의 장점:**
- ⚡ 약 10초 간격 (빠름!)
- 🎤 녹음과 분석 동시 진행
- 진정한 거의 실시간!
- 위급도별 실시간 알림 가능
- CPU 효율적 (지연시간 단축)

❌ **제약:**
- 정확한 실시간은 아님 (10초 지연)
- CPU/메모리 사용 (2개 스레드)

- Ollama 서버 항상 실행 필요

### 🚀 방법 2: REST API 서버로 실행

다른 프로그램에서 HTTP 요청으로 접근 가능합니다.

```bash
# API 서버 시작
python3 api_server.py

# 서버는 http://localhost:5000 에서 실행됨
```

**Python에서 API 사용:**

```python
import requests

response = requests.post('http://localhost:5000/api/transcribe', 
    json={'audio_file': '/path/to/audio.wav'})
result = response.json()
print(result['text'])
```

**JavaScript/Node.js에서 API 사용:**

```javascript
const response = await fetch('http://localhost:5000/api/transcribe', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({audio_file: '/path/to/audio.wav'})
});
const result = await response.json();
console.log(result.text);
```

### 🔄 방법 3: 백그라운드 모니터링

```bash
# 지속적으로 음성을 모니터링 (Ctrl+C로 종료)
python3 voice_monitor.py
```

### 📊 결과 확인

모든 결과는 `transcriptions/` 디렉토리에 저장됩니다.

```
transcriptions/
├── 2026-01-22/
│   ├── transcriptions.json    # 해당 날짜 모든 결과 (배열)
│   ├── latest.json            # 최신 결과 (덮어쓰기)
│   ├── transcriptions.txt     # 텍스트 형식 로그
│   └── 2026-01-22T10-30-45Z.json  # 개별 결과
```

**결과 구조:**

```json
{
  "timestamp": "2026-01-22T10:30:45Z",
  "transcribed_text": "안녕하세요. 날씨가 정말 좋네요.",
  "analysis": {
    "context": "일상 대화",
    "situation": "사용자가 날씨에 대해 긍정적으로 표현함",
    "emotional_state": "긍정",
    "urgency": "낮음"
  }
}
```

---

## 프로젝트 구조

```
contextllm/
├── README.md                      # 이 파일
├── requirements.txt               # Python 패키지 목록 (Python 3.10.19용)
│
├── 🎤 핵심 모듈
├── voice_analyzer.py              # 음성 인식 + LLM 분석 (메인)
├── voice_example.py               # 사용 예제
├── voice_monitor.py               # 백그라운드 모니터링
├── whisper_service.py             # Whisper 래퍼
│
├── 📡 API 서버
├── api_server.py                  # REST API 서버 (Flask)
│
├── 📝 문서
├── SETUP_GUIDE.md                 # 상세 설정 가이드
├── GITHUB_SETUP.md                # GitHub 업로드 가이드
├── OLLAMA_GUIDE.md                # Ollama 상세 가이드
├── REAL_TEST_GUIDE.md             # 실제 테스트 가이드
│
├── 🧪 테스트 스크립트
├── test_ollama.py                 # Ollama 연결 테스트
├── test_korean_analysis.py        # 한국어 분석 테스트
├── test_real_analysis.py          # 실제 음성 분석 테스트
│
├── 📁 디렉토리
├── .venv/                         # Python 가상 환경 (자동 생성)
├── recordings/                    # 녹음 파일 저장소
├── transcriptions/                # 분석 결과 저장소
└── src/                           # VS Code 확장 소스 (TypeScript)
    └── extension.ts               # VS Code 플러그인
```

### 주요 파일 설명

#### 🎯 voice_analyzer.py (메인)

```python
class VoiceAnalyzer:
    """음성 인식 + LLM 분석 메인 클래스"""
    
    def record_audio(duration=10)
        # Sox를 사용하여 음성 녹음
    
    def transcribe(audio_file)
        # Whisper로 음성 → 텍스트 변환
    
    def analyze(text, system_prompt=None)
        # Ollama LLM으로 텍스트 분석
    
    def transcribe_and_analyze(duration=10, system_prompt=None)
        # 녹음 + 변환 + 분석 전체 프로세스
```

#### 📡 api_server.py (REST API)

```python
# POST /api/transcribe - 음성 파일 변환
response = requests.post('http://localhost:5000/api/transcribe',
    json={'audio_file': 'path/to/audio.wav'})

# POST /api/analyze - 텍스트 분석
response = requests.post('http://localhost:5000/api/analyze',
    json={'text': '분석할 텍스트'})

# GET /api/status - 서버 상태 확인
response = requests.get('http://localhost:5000/api/status')
```

---

## 다른 프로젝트에 이식하기

### 🔗 방법 1: 모듈로 임포트

이 프로젝트를 다른 Python 프로젝트에서 모듈로 사용할 수 있습니다.

#### 1단계: 파일 복사

```bash
# 당신의 프로젝트 디렉토리로 이동
cd ~/Desktop/my_project

# contextllm 파일들을 복사
cp ~/Desktop/contextllm/voice_analyzer.py .
cp ~/Desktop/contextllm/whisper_service.py .
cp ~/Desktop/contextllm/requirements.txt ./requirements_whisper.txt
```

#### 2단계: 패키지 설치

```bash
# 기존 requirements.txt에 병합
cat requirements_whisper.txt >> requirements.txt

# 설치
pip install -r requirements.txt
```

#### 3단계: 코드에서 사용

```python
from voice_analyzer import VoiceAnalyzer

class MyApp:
    def __init__(self):
        self.analyzer = VoiceAnalyzer()
    
    def process_voice(self):
        result = self.analyzer.transcribe_and_analyze(duration=10)
        return result['analysis']

# 사용
app = MyApp()
analysis = app.process_voice()
```

### 🌐 방법 2: REST API로 연결

별도의 서버로 실행하고 HTTP로 통신합니다.

#### 1단계: 이 프로젝트를 API 서버로 실행

```bash
cd ~/Desktop/contextllm
source .venv/bin/activate
python3 api_server.py
# API 서버 실행 중... (http://localhost:5000)
```

#### 2단계: 다른 프로젝트에서 API 호출

**Python:**

```python
import requests

class VoiceClient:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
    
    def transcribe(self, audio_file):
        response = requests.post(
            f'{self.api_url}/api/transcribe',
            json={'audio_file': audio_file}
        )
        return response.json()
    
    def analyze(self, text):
        response = requests.post(
            f'{self.api_url}/api/analyze',
            json={'text': text}
        )
        return response.json()

# 사용
client = VoiceClient()
result = client.transcribe('/path/to/audio.wav')
```

**Node.js/JavaScript:**

```javascript
class VoiceClient {
    constructor(apiUrl = 'http://localhost:5000') {
        this.apiUrl = apiUrl;
    }
    
    async transcribe(audioFile) {
        const response = await fetch(`${this.apiUrl}/api/transcribe`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({audio_file: audioFile})
        });
        return await response.json();
    }
    
    async analyze(text) {
        const response = await fetch(`${this.apiUrl}/api/analyze`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        });
        return await response.json();
    }
}

// 사용
const client = new VoiceClient();
const result = await client.transcribe('/path/to/audio.wav');
```

### 📦 방법 3: Docker 컨테이너로 배포

```dockerfile
FROM python:3.10.19-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "api_server.py"]
```

**빌드 및 실행:**

```bash
# 빌드
docker build -t voice-analyzer .

# 실행
docker run -p 5000:5000 voice-analyzer
```

### 🔧 환경 변수 설정

다른 프로젝트와의 호환성을 위해 환경 변수로 설정할 수 있습니다.

```bash
# .env 파일 또는 시스템 환경 변수
export OLLAMA_MODEL=mistral
export WHISPER_MODEL=base
export API_PORT=5000
export RECORDING_DURATION=10
```

**Python 코드에서 사용:**

```python
import os

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
API_PORT = int(os.getenv('API_PORT', 5000))
```

---

## 자주 묻는 질문 (FAQ)

### Q1: 음성 녹음이 되지 않습니다.

**A:** Sox가 설치되지 않았을 가능성이 높습니다.

```bash
# Sox 설치
brew install sox

# 확인
sox --version

# 마이크 테스트
sox -d test.wav
# 3초간 녹음 후 Ctrl+C로 종료
play test.wav  # 재생 테스트
```

### Q2: Ollama 연결 오류가 발생합니다.

**A:** Ollama 서버가 실행 중이어야 합니다.

```bash
# 새 터미널에서 실행
ollama serve

# 다른 터미널에서 테스트
curl http://localhost:11434/api/tags
```

### Q3: "No module named 'whisper'" 오류

**A:** 가상 환경이 활성화되지 않았을 가능성입니다.

```bash
# 가상 환경 활성화 확인
which python
# /Users/.../.venv/bin/python 인지 확인

# 재설치
source .venv/bin/activate
pip install -r requirements.txt
```

### Q4: Python 3.10.19이 없습니다.

**A:** Homebrew로 설치하세요.

```bash
brew install python@3.10

# 확인
/opt/homebrew/bin/python3.10 --version
# Python 3.10.19
```

### Q5: 한국어가 제대로 인식되지 않습니다.

**A:** Whisper의 언어 설정을 명시하세요.

```python
from voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()

# whisper_service.py에서 언어 설정
# --language ko 옵션 추가
result = analyzer.transcribe_and_analyze(
    duration=10,
    language='ko'  # 한국어 지정
)
```

### Q6: 메모리 부족 오류가 발생합니다.

**A:** 더 가벼운 모델을 사용하거나 RAM을 증설하세요.

```bash
# 더 작은 모델 사용
ollama pull neural-chat  # Mistral보다 가벼움

# 또는 venv 설정에서 모델 메모리 제한
export OLLAMA_NUM_THREAD=4  # CPU 스레드 제한
export OLLAMA_NUM_GPU=0      # GPU 미사용
```

### Q7: API 서버의 응답이 느립니다.

**A:** 다음을 시도해보세요:

1. **더 빠른 모델 사용:**
   ```bash
   ollama pull neural-chat  # Mistral보다 빠름
   ```

2. **Whisper 모델 최적화:**
   ```python
   # tiny 또는 small 모델 사용
   export WHISPER_MODEL=small
   ```

3. **멀티 스레딩 활성화:**
   ```bash
   export OLLAMA_NUM_THREAD=8
   ```

### Q8: 다른 프로젝트와 Python 버전이 충돌합니다.

**A:** pyenv를 사용하여 여러 Python 버전을 관리할 수 있습니다.

```bash
# pyenv 설치
brew install pyenv

# 여러 Python 버전 설치
pyenv install 3.10.19
pyenv install 3.11.0

# 프로젝트별 버전 설정
cd ~/Desktop/contextllm
pyenv local 3.10.19

cd ~/Desktop/other_project
pyenv local 3.11.0
```

### Q9: Windows에서 음성 녹음 오류가 발생합니다.

**A:** PyAudio 또는 sounddevice를 설치하고 voice_analyzer.py를 수정하세요.

```bash
# sounddevice 설치 (권장)
pip install sounddevice scipy

# 또는 PyAudio 설치 (복잡할 수 있음)
pip install pyaudio
```

그 후 [voice_analyzer.py](voice_analyzer.py)의 `record_audio()` 메서드를 Windows 버전으로 수정하세요. (위의 "🪟 Windows" 섹션 참고)

### Q10: Linux에서 마이크 권한 오류가 발생합니다.

**A:** 사용자를 audio 그룹에 추가하세요.

```bash
# 현재 사용자를 audio 그룹에 추가
sudo usermod -aG audio $USER

# 또는 임시로 권한 주기
sudo usermod -aG audio $(whoami)

# 그룹 변경 반영 (재부팅 또는 새 터미널)
newgrp audio

# 확인
groups
# audio가 포함되어 있는지 확인
```

### Q11: Docker에서 실행 시 음성이 인식되지 않습니다.

**A:** Docker 컨테이너에 호스트의 마이크 접근권한을 주어야 합니다.

```bash
# macOS
docker run -it --device /dev/snd voice-analyzer

# Linux
docker run -it --device /dev/snd:/dev/snd -v /run/user/1000/pulse:/run/user/1000/pulse voice-analyzer

# Windows (WSL2)
docker run -it --device /dev/snd voice-analyzer
```

### Q12: OS별 호환성 문제가 있습니다.

**A:** 다음을 확인하세요:

| 기능 | macOS | Windows | Linux |
|------|-------|---------|-------|
| 음성 녹음 | ✅ Sox | ⚠️ PyAudio/sounddevice | ✅ Sox |
| Ollama | ✅ | ✅ | ✅ |
| Whisper | ✅ | ✅ | ✅ |
| REST API | ✅ | ✅ | ✅ |
| Python 3.10.19 | ✅ | ✅ | ✅ |

**Windows 사용자:**
- PyAudio 설치 시 Visual Studio Build Tools 필요
- 또는 sounddevice 사용 권장
- requirements.txt에 `sounddevice` 추가 필요

**Linux 사용자:**
- PulseAudio/ALSA 설정 필요할 수 있음
- 마이크 권한(audio 그룹) 필요

**모든 OS:**
- Python 3.10.19 필수 (호환성)
- Ollama 서버 반드시 실행 중
- 인터넷 연결 필요 (초기 모델 다운로드 시)


---

## 🚀 다음 단계

1. **사용자 정의 프롬프트 작성** - `SETUP_GUIDE.md`의 시나리오별 커스텀 프롬프트 참고
2. **VS Code 확장 개발** - `src/extension.ts`에서 플러그인 커스터마이징
3. **데이터베이스 연동** - 결과를 MongoDB/PostgreSQL에 저장
4. **클라우드 배포** - Azure/AWS에 배포
5. **팀 협업** - GitHub에 푸시하고 협업 시작

---

## 📞 지원

문제가 발생하면:

1. **로그 확인**: `transcriptions/` 디렉토리의 결과 파일 확인
2. **테스트 실행**: `test_*.py` 파일 실행
3. **GitHub Issues**: 버그 리포트 제출
4. **문서 참고**: `SETUP_GUIDE.md`, `OLLAMA_GUIDE.md`, `REAL_TEST_GUIDE.md`

---

## 📄 라이선스

이 프로젝트는 **MIT 라이선스**를 따릅니다.

---

## 🙏 감사의 말

- [OpenAI Whisper](https://github.com/openai/whisper) - 음성 인식
- [Ollama](https://ollama.ai) - LLM 로컬 실행
- [PyTorch](https://pytorch.org) - 머신 러닝 프레임워크

---

**마지막 업데이트**: 2026년 1월 22일  
**Python 버전**: 3.10.19  
**상태**: ✅ 프로덕션 준비 완료
