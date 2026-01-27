# 🔗 모듈 통합 가이드

Context LLM을 다른 모듈(YOLO 등)과 통합할 때 참고해볼 자료 <<

---

## 🏗️ 아키텍처 변화

### 현재 (단위 모듈)
```
main.py → IntegratedMultimodalSystem (독립 실행)
```

### 통합 시 (목표)
```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│              (메인 컨트롤러/스케줄러)                  │
└─────────────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ YOLO Module │ │ContextLLM   │ │ Other Module│
│ (객체탐지)   │ │(음성+영상)   │ │ (...)       │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

## 🔧 통합 방법 3가지

### 1️⃣ 공통 인터페이스 (추천)

각 모듈이 같은 인터페이스를 구현:

```python
# base_module.py (공통)
from abc import ABC, abstractmethod

class BaseAnalysisModule(ABC):
    """모든 분석 모듈이 구현해야 할 인터페이스"""
    
    @abstractmethod
    def analyze(self, input_data: dict) -> dict:
        """분석 실행"""
        pass
    
    @abstractmethod
    def get_priority(self) -> str:
        """우선순위 반환 (CRITICAL/HIGH/MEDIUM/LOW)"""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """모듈 준비 상태"""
        pass
```

```python
# context_llm_module.py (현재 시스템을 래핑)
from base_module import BaseAnalysisModule
from src.core.integrated_multimodal_system import IntegratedMultimodalSystem

class ContextLLMModule(BaseAnalysisModule):
    def __init__(self, config):
        self.system = IntegratedMultimodalSystem(**config)
        self.last_result = {}
    
    def analyze(self, input_data: dict) -> dict:
        # input_data: {"frame": np.array, "audio": bytes, ...}
        self.last_result = self.system.analyze_once()
        return self.last_result
    
    def get_priority(self) -> str:
        return self.last_result.get("priority", "LOW")
    
    def is_ready(self) -> bool:
        return self.system.is_initialized()
```

```python
# yolo_module.py
class YOLOModule(BaseAnalysisModule):
    def __init__(self, model_path):
        self.model = load_yolo(model_path)
    
    def analyze(self, input_data: dict) -> dict:
        frame = input_data["frame"]
        detections = self.model.detect(frame)
        return {
            "objects": detections,
            "person_detected": any(d["class"] == "person" for d in detections),
            "priority": self._calc_priority(detections)
        }
    
    def get_priority(self) -> str:
        return self.last_result.get("priority", "LOW")
    
    def is_ready(self) -> bool:
        return self.model is not None
```

---

### 2️⃣ 이벤트 기반 (Pub/Sub) << 쉬운건 이거 아닐까? >>

모듈 간 느슨한 결합:

```python
# event_bus.py
from typing import Callable, Dict, List

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """이벤트 구독"""
        self.subscribers.setdefault(event_type, []).append(callback)
    
    def publish(self, event_type: str, data: dict):
        """이벤트 발행"""
        for callback in self.subscribers.get(event_type, []):
            callback(data)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """구독 해제"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
```

```python
# 사용 예시
bus = EventBus()

# ContextLLM이 "emergency" 이벤트 발행
bus.subscribe("emergency", alert_system.handle)
bus.subscribe("emergency", logging_system.log)

# YOLO가 "person_detected" 이벤트 발행
bus.subscribe("person_detected", context_llm.on_person_detected)

# 모듈 내부에서 이벤트 발행
class ContextLLMModule:
    def analyze(self, input_data):
        result = self.system.analyze_once()
        
        if result.get("is_emergency"):
            self.event_bus.publish("emergency", result)
        
        return result
```

---

### 3️⃣ 오케스트레이터 패턴

중앙에서 모듈 실행 순서/조건 관리:

```python
# orchestrator.py
from typing import Dict, List, Callable, Any

class ModuleOrchestrator:
    def __init__(self):
        self.modules: Dict[str, BaseAnalysisModule] = {}
        self.pipelines: Dict[str, List[dict]] = {}
    
    def register(self, name: str, module: BaseAnalysisModule):
        """모듈 등록"""
        self.modules[name] = module
        print(f"✅ 모듈 등록: {name}")
    
    def define_pipeline(self, name: str, steps: List[dict]):
        """
        파이프라인 정의
        
        steps 예시:
        [
            {"module": "yolo"},
            {"module": "context_llm", "condition": lambda r: r["yolo"]["person_detected"]},
            {"module": "alert", "condition": lambda r: r.get("context_llm", {}).get("is_emergency")},
        ]
        """
        self.pipelines[name] = steps
        print(f"✅ 파이프라인 정의: {name} ({len(steps)} steps)")
    
    def run_pipeline(self, pipeline_name: str, input_data: dict) -> Dict[str, Any]:
        """파이프라인 실행"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"파이프라인 없음: {pipeline_name}")
        
        results = {}
        
        for step in self.pipelines[pipeline_name]:
            module_name = step["module"]
            module = self.modules.get(module_name)
            
            if not module:
                print(f"⚠️ 모듈 없음: {module_name}")
                continue
            
            # 조건 체크 (이전 결과에 따라 실행 여부 결정)
            condition = step.get("condition")
            if condition and not condition(results):
                print(f"⏭️ 조건 불충족, 스킵: {module_name}")
                continue
            
            # 모듈 실행
            print(f"▶️ 실행: {module_name}")
            results[module_name] = module.analyze(input_data)
        
        return results
    
    def get_highest_priority(self, results: Dict[str, Any]) -> str:
        """모든 결과 중 가장 높은 우선순위 반환"""
        priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        
        for priority in priority_order:
            for module_result in results.values():
                if module_result.get("priority") == priority:
                    return priority
        
        return "LOW"
```

```python
# 사용 예시
orch = ModuleOrchestrator()

# 모듈 등록
orch.register("yolo", YOLOModule("yolov8n.pt"))
orch.register("context_llm", ContextLLMModule(config))
orch.register("alert", AlertModule())

# 파이프라인 정의
# 보안 파이프라인: YOLO → (사람 감지시) → ContextLLM → (긴급시) → Alert
orch.define_pipeline("security", [
    {"module": "yolo"},
    {"module": "context_llm", "condition": lambda r: r["yolo"]["person_detected"]},
    {"module": "alert", "condition": lambda r: r.get("context_llm", {}).get("is_emergency")},
])

# 단순 파이프라인: 모든 모듈 순차 실행
orch.define_pipeline("full_analysis", [
    {"module": "yolo"},
    {"module": "context_llm"},
])

# 실행
results = orch.run_pipeline("security", {"frame": frame, "audio": audio})
print(f"최고 우선순위: {orch.get_highest_priority(results)}")
```

---

## 📋 매핑 테이블

| 현재 (Context LLM) | 통합 시 |
|-------------------|---------|
| `IntegratedMultimodalSystem` | `ContextLLMModule`로 래핑 |
| `analyze_once()` | `analyze(input_data)` 인터페이스 |
| `result["priority"]` | 공통 우선순위 체계 사용 |
| `config/config.yaml` | 통합 config에서 섹션으로 분리 |
| `main.py` | 오케스트레이터가 호출 |

---

## 📁 추천 디렉토리 구조

```
main_system/
├── orchestrator.py          # 메인 컨트롤러
├── event_bus.py             # 이벤트 시스템
├── base_module.py           # 공통 인터페이스
│
├── config/
│   └── main_config.yaml     # 통합 설정
│       ├── yolo:
│       ├── context_llm:
│       └── alert:
│
├── modules/
│   ├── context_llm/         # LLM 모듈
│   │   ├── src/
│   │   │   └── core/
│   │   ├── config/
│   │   └── main.py
│   │
│   ├── yolo_detection/      # YOLO 모듈
│   │   ├── model/
│   │   └── detector.py
│   │
│   └── alert_system/        # 알림 모듈
│       └── alerter.py
│
├── wrappers/                # 각 모듈의 래퍼
│   ├── context_llm_wrapper.py
│   ├── yolo_wrapper.py
│   └── alert_wrapper.py
│
└── main.py                  # 통합 진입점
```

---

## 🚀 통합 예시 (최종)

```python
# main.py (통합 시스템)
from orchestrator import ModuleOrchestrator
from event_bus import EventBus
from wrappers.context_llm_wrapper import ContextLLMModule
from wrappers.yolo_wrapper import YOLOModule
from wrappers.alert_wrapper import AlertModule

def main():
    # 이벤트 버스 초기화
    bus = EventBus()
    
    # 오케스트레이터 초기화
    orch = ModuleOrchestrator()
    
    # 모듈 등록
    orch.register("yolo", YOLOModule("models/yolov8n.pt"))
    orch.register("context_llm", ContextLLMModule())
    orch.register("alert", AlertModule(bus))
    
    # 파이프라인 정의
    orch.define_pipeline("realtime_security", [
        {"module": "yolo"},
        {"module": "context_llm", "condition": lambda r: r["yolo"]["person_detected"]},
        {"module": "alert", "condition": lambda r: r.get("context_llm", {}).get("is_emergency")},
    ])
    
    # 메인 루프
    camera = cv2.VideoCapture(0)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # 파이프라인 실행
        results = orch.run_pipeline("realtime_security", {
            "frame": frame,
            "timestamp": time.time()
        })
        
        # 결과 처리
        priority = orch.get_highest_priority(results)
        if priority == "CRITICAL":
            print("🚨 긴급 상황 감지!")

if __name__ == "__main__":
    main()
```

---

## ✅ 체크리스트

통합 전 확인사항:

- [ ] 각 모듈이 `BaseAnalysisModule` 인터페이스 구현
- [ ] 공통 우선순위 체계 정의 (CRITICAL/HIGH/MEDIUM/LOW)
- [ ] 모듈 간 데이터 형식 통일 (`input_data` dict 구조)
- [ ] 에러 핸들링 추가 (한 모듈 실패 시 다른 모듈 영향 없도록)
- [ ] 설정 파일 통합 (`main_config.yaml`)
- [ ] 로깅 시스템 통합
- [ ] 테스트 코드 작성

---

## 📝 참고

현재 Context LLM 모듈은 이미 잘 분리되어 있어서, **래퍼 클래스만 만들면** 바로 가능.

```python
# wrappers/context_llm_wrapper.py (최소 구현)
import sys
sys.path.insert(0, "modules/context_llm/src")

from core.integrated_multimodal_system import IntegratedMultimodalSystem

class ContextLLMModule:
    def __init__(self):
        self.system = IntegratedMultimodalSystem()
    
    def analyze(self, input_data: dict) -> dict:
        # 기존 시스템 그대로 사용
        return self.system.analyze_once()
```
