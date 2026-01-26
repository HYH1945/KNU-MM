# ⚡ QUICKSTART - 5분 시작

## 1️⃣ 설치 (2분)

```bash
# 필수 패키지
pip install -r requirements.txt

# macOS 추가 (PyAudio)
brew install portaudio
```

## 2️⃣ 설정 (1분)

```bash
# config/.env 파일 생성
echo "OPENAI_API_KEY=sk-your-key" > config/.env
```

## 3️⃣ 실행 (2분)

```bash
# 무료 버전 (권장)
python tests/test_free_realtime_simple.py

# 또는 유료 버전
python tests/test_google_realtime_simple.py
```

## 4️⃣ 말하기

마이크로 말씀하세요:
```
"서버가 다운됐어요! 긴급입니다!"
```

---

## 📊 기대되는 결과

```
📝 인식된 텍스트:
   '서버가 다운됐어요! 긴급입니다!'

🎯 상황 분석:
   시스템이 다운된 심각한 상황입니다. 즉시 대응이 필요합니다.

📌 상황 유형: 시스템장애
🔧 권장 조치: 즉시 시스템 관리자에게 알리고 복구 절차 실행

🚨 긴급 상황 - 즉시 대응 필요
📋 참고 예시: 서버 다운, 보안 침해, 시스템 오류

💾 기록 저장 (총 1건): conversation_history_20260126_114235.json
```

---

## 💡 팁

- **Ctrl+C** - 언제든 종료
- **'quit' 말하기** - 프로그램 끝냄
- **30초 침묵** - 자동으로 음성 끝남
- **로그 확인**: `data/logs/conversation_history_*.json`

---

## 🚀 다음 단계

- [FREE_REALTIME_GUIDE.md](FREE_REALTIME_GUIDE.md) - 자세한 가이드
- [PRIORITY_MANAGEMENT.md](PRIORITY_MANAGEMENT.md) - 우선순위 시스템
