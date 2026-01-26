# 🎤 Google Realtime STT 가이드 (유료 버전)

**낮은 레이턴시, 더 높은 정확도**

> ⚡ **초저 레이턴시** - 200-500ms  
> 🎯 **높은 정확도** - Google의 최신 음성 인식  
> 💰 **비용** - ~$0.025/분 (~$155/월)  

---

## ⚡ 설정

### 1️⃣ Google Cloud 자격증명

```bash
# Google Cloud Service Account 키 다운로드
# https://cloud.google.com/docs/authentication/getting-started

export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### 2️⃣ 패키지 설치

```bash
pip install google-cloud-speech
```

### 3️⃣ 서버 시작

```bash
python tests/test_google_realtime_simple.py
```

---

## 비용 계산

- **가격**: $0.025/분
- **월간 사용**: 100시간 = ~$150
- **프리 티어**: 처음 60분 무료

---

## 🎯 언제 사용할까?

- 프로덕션 환경
- 높은 정확도 필요
- 낮은 레이턴시 중요
- 비용 여유 있음

---

## 📚 다음 단계

- [FREE_REALTIME_GUIDE.md](FREE_REALTIME_GUIDE.md) - 무료 버전 (권장)
- [QUICKSTART.md](QUICKSTART.md) - 빠른 시작
