# 🎤 무료 음성 인식 완전 가이드 (SpeechRecognition)

**API 키 필요 없음! 무료로 시작하세요**

> ✅ **완전 무료** - API 키 필요 없음  
> 🚀 **매우 간단** - 3단계 설치  
> 🎯 **실시간 처리** - 문장 단위 즉시 분석  
> 🌍 **한국어 지원** - Google 음성 인식 (무료)  

---

## ⚡ 3분 시작

### 1️⃣ 패키지 설치

```bash
pip install SpeechRecognition pyaudio openai flask python-dotenv
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### 2️⃣ OpenAI API 키 설정 (분석용)

```bash
# config/.env 파일에 추가
OPENAI_API_KEY=sk-your-key
```

### 3️⃣ 서버 시작

```bash
python tests/test_free_realtime_simple.py
```

### 4️⃣ 마이크로 말하기!

```
"서버가 다운됐어요! 긴급입니다!"
```

---

## 🎯 무료 vs 유료 비교

| 기능 | 무료 (이것!) | Google Cloud | 가격 |
|-----|-----------|------------|------|
| **음성 인식** | SpeechRecognition | Google STT | 무료 vs $0.025/분 |
| **API 키** | 필요 없음 ✅ | 필요함 | 무료 vs $900/100시간 |
| **한국어** | ✅ 지원 | ✅ 우수 | ✅ vs ✅ |
| **레이턴시** | 1-2초 | 200-500ms | OK vs 우수 |
| **설정 난이도** | 매우 쉬움 | 복잡 | ✅ vs ❌ |

---

## 📡 REST API (선택사항)

무료 버전의 REST API 서버:

```bash
python src/api/free_realtime_api_server.py
```

### 엔드포인트

```bash
# 모니터링 시작
curl -X POST http://localhost:5000/api/monitor/start

# 결과 확인
curl http://localhost:5000/api/results/latest

# 모니터링 중지
curl -X POST http://localhost:5000/api/monitor/stop
```

---

## 🧪 테스트 결과 확인

실행 후 로그 확인:

```bash
cat data/logs/conversation_history_*.json | python -m json.tool
```

---

## 🐛 트러블슈팅

**PyAudio 설치 오류:**
```bash
brew install portaudio
pip install pyaudio
```

**마이크 권한 오류 (macOS):**
- 시스템 설정 → 개인정보 보호 및 보안 → 마이크 → 터미널 허용

**Google 인식 오류:**
- 인터넷 연결 확인

---

## 📚 다음 단계

- [QUICKSTART.md](QUICKSTART.md) - 5분 빠른 시작
- [PRIORITY_MANAGEMENT.md](PRIORITY_MANAGEMENT.md) - 우선순위 시스템
- [GOOGLE_REALTIME_GUIDE.md](GOOGLE_REALTIME_GUIDE.md) - 유료 버전
