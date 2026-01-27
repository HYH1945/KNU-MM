# 테스트셋 폴더

이 폴더에 테스트용 이미지나 비디오 파일을 넣어주세요.

## 지원 형식

### 이미지
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.gif`
- `.webp`

### 비디오
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`
- `.flv`
- `.wmv`

## 사용 예시

```python
from core.integrated_multimodal_system import IntegratedMultimodalSystem

system = IntegratedMultimodalSystem()

# 테스트셋 폴더 지정
system.use_testset("testsets/")

# 파일 목록 확인
files = system.get_testset_files()
print(files)  # ['violence1.mp4', 'accident.jpg', ...]

# 특정 파일 선택
system.select_testset_file("violence1.mp4")
# 또는 인덱스로
system.select_testset_file(0)

# 영상만 분석 (음성 입력 없이)
result = system.analyze_video_only("폭행 상황인지 분석해 주세요")

# 모든 파일 순차 분석
results = system.analyze_testset_all()
```

## 추천 테스트 시나리오

1. **폭행/싸움 영상**
2. **화재/연기 이미지**
3. **사고 현장**
4. **정상 상황 (비교용)**
5. **혼잡한 환경**
6. **야간 환경**

## 주의사항

- 민감한 콘텐츠는 OpenAI 안전 정책에 의해 분석이 거부될 수 있습니다.
- 파일 이름에 한글이 포함되어도 작동합니다.
- 파일은 알파벳/숫자 순으로 정렬됩니다.
