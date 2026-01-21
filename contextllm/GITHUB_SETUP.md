# 🚀 GitHub 업로드 가이드

## 📋 준비사항

✅ GitHub 계정 (없으면 https://github.com에서 가입)
✅ Git 설치 (macOS는 기본 포함)
✅ `.gitignore` 파일 생성됨 ✓

---

## 📍 현재 폴더 상태

```bash
# 현재 위치 확인
pwd
# /Users/jangjun-yong/Desktop/jongf1

# Git 상태 확인
git status
# fatal: not a git repository (아직 초기화 안 됨)
```

---

## 🎯 단계별 진행

### 1️⃣ 로컬 Git 초기화

```bash
cd /Users/jangjun-yong/Desktop/jongf1

# Git 초기화
git init

# 사용자 정보 설정
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 2️⃣ 첫 번째 커밋

```bash
# 모든 파일 추가 (제외: .gitignore 내용)
git add .

# 커밋
git commit -m "🎤 Initial commit: Local Whisper + LLM voice analysis system"
```

**예상 출력:**
```
✓ create mode 100644 .gitignore
✓ create mode 100644 README.md
✓ create mode 100644 requirements.txt
✓ create mode 100644 voice_analyzer.py
✓ create mode 100644 voice_example.py
...
```

### 3️⃣ GitHub에 저장소 생성

**https://github.com/new 방문:**

1. Repository name: `jongf1` (또는 다른 이름)
2. Description: "Local Whisper + LLM voice analysis system"
3. Public / Private 선택
4. **Create repository** 클릭

**생성 후 나오는 명령어 참고:**
```
…or push an existing repository from the command line
```

### 4️⃣ 원격 저장소 연결

```bash
# YOUR_USERNAME을 실제 GitHub 계정명으로 변경!
git remote add origin https://github.com/YOUR_USERNAME/jongf1.git

# 기본 브랜치 이름 설정 (main)
git branch -M main

# 원격 저장소에 푸시
git push -u origin main
```

**예상 메시지:**
```
remote: Create a pull request for 'main' on GitHub by visiting:
remote:      https://github.com/YOUR_USERNAME/jongf1/pull/new/main
```

---

## ✅ 검증

### GitHub에서 확인

1. https://github.com/YOUR_USERNAME/jongf1 방문
2. 파일 목록 확인:
   - ✅ README.md
   - ✅ voice_analyzer.py
   - ✅ requirements.txt
   - ❌ .venv/ (제외됨)
   - ❌ node_modules/ (제외됨)
   - ❌ recordings/ (제외됨)

### 터미널에서 확인

```bash
# Git 상태 확인
git status
# On branch main
# nothing to commit, working tree clean

# 원격 저장소 확인
git remote -v
# origin  https://github.com/YOUR_USERNAME/jongf1.git (fetch)
# origin  https://github.com/YOUR_USERNAME/jongf1.git (push)
```

---

## 🔄 이후 작업 흐름

### 수정 후 업로드

```bash
# 변경사항 확인
git status

# 파일 추가
git add .

# 커밋
git commit -m "🔧 Fix: improved error handling"

# 푸시
git push
```

### 일반적인 커밋 메시지

```bash
# 새 기능
git commit -m "✨ feat: add streaming support"

# 버그 수정
git commit -m "🐛 fix: connection timeout issue"

# 문서 개선
git commit -m "📝 docs: update installation guide"

# 성능 개선
git commit -m "⚡ perf: optimize LLM response time"

# 테스트 추가
git commit -m "✅ test: add Korean analysis tests"
```

---

## 🛠️ 유용한 Git 명령어

```bash
# 커밋 이력 보기
git log --oneline

# 특정 파일 변경사항 보기
git diff voice_analyzer.py

# 마지막 커밋 수정
git commit --amend

# 특정 파일 원래 상태로 복구
git checkout -- voice_analyzer.py

# 마지막 커밋 취소
git reset HEAD~1
```

---

## 📊 .gitignore 확인

```bash
# 제외되는 파일 목록 확인
git status --ignored

# 예상 결과:
# Ignored: .venv/
# Ignored: node_modules/
# Ignored: recordings/
# Ignored: .DS_Store
```

---

## 🚨 실수 방지

❌ **하지 말 것:**
- `.venv` 폴더 추가
- `node_modules` 폴더 추가
- `.env` 파일 (API 키 포함)
- 개인 정보 포함

✅ **해야 할 것:**
- 의존성은 `requirements.txt`에 기록
- 설정은 가이드에 문서화
- 테스트 후 푸시

---

## 💡 추가 팁

### GitHub를 다른 기기에서 사용

```bash
# 다른 컴퓨터에서 클론
git clone https://github.com/YOUR_USERNAME/jongf1.git
cd jongf1

# 의존성 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 바로 사용 가능!
python3 voice_analyzer.py
```

### 팀 협업

```bash
# 최신 코드 가져오기
git pull

# 새 브랜치 생성 (기능 개발)
git checkout -b feature/new-feature

# 작업 후 푸시
git push -u origin feature/new-feature

# GitHub에서 Pull Request 생성
```

### SSH 설정 (선택사항, 비밀번호 불필요)

```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "your.email@example.com"

# GitHub에 공개키 추가
# https://github.com/settings/keys

# SSH 사용
git remote set-url origin git@github.com:YOUR_USERNAME/jongf1.git
```

---

## 🎓 유용한 리소스

- [GitHub 가이드](https://guides.github.com/)
- [Git 치트시트](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

**GitHub 업로드 완료! 🎉**
