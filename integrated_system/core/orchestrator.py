"""
오케스트레이터 - 모든 모듈의 등록, 파이프라인 정의, 실행 관리

메인 루프에서 파이프라인을 실행하며, 각 단계(step)의 조건을 평가하여
조건부 모듈 실행을 지원합니다.

사용법:
    orch = Orchestrator(event_bus)
    orch.register(yolo_module)
    orch.register(llm_module)
    
    orch.define_pipeline("security", [
        {"module": "yolo"},
        {"module": "context_llm", "condition": lambda r: r.get("yolo", {}).get("person_detected")},
    ])
    
    results = orch.run_pipeline("security", shared_data)
"""

import time
import logging
from typing import Dict, List, Any, Callable, Optional
from .base_module import BaseModule
from .event_bus import EventBus, Event

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    모듈 오케스트레이터
    
    역할:
    1. 모듈 등록/관리
    2. 파이프라인 정의 (모듈 실행 순서 + 조건)
    3. 파이프라인 실행 (조건부 모듈 실행)
    4. 결과 수집 및 우선순위 판정
    """

    # 우선순위 순서 (높은 것 → 낮은 것)
    PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._modules: Dict[str, BaseModule] = {}
        self._pipelines: Dict[str, List[dict]] = {}
        self._current_mode: str = "IDLE"

    # ─── 모듈 관리 ───

    def register(self, module: BaseModule) -> bool:
        """
        모듈 등록 및 초기화
        
        Returns:
            초기화 성공 여부
        """
        name = module.name
        logger.info(f"[Orchestrator] 모듈 등록 시작: {name}")
        
        try:
            success = module.initialize()
            if success:
                self._modules[name] = module
                module._initialized = True
                logger.info(f"[Orchestrator] ✅ 모듈 등록 완료: {name}")
            else:
                logger.warning(f"[Orchestrator] ⚠️ 모듈 초기화 실패 (비활성): {name}")
                module._initialized = False
                self._modules[name] = module  # 등록은 하되 비활성화 상태
            return success
        except Exception as e:
            logger.error(f"[Orchestrator] ❌ 모듈 등록 오류 [{name}]: {e}")
            return False

    def get_module(self, name: str) -> Optional[BaseModule]:
        """모듈 조회"""
        return self._modules.get(name)

    def list_modules(self) -> Dict[str, Dict]:
        """등록된 모듈 상태 목록"""
        return {name: mod.get_status() for name, mod in self._modules.items()}

    # ─── 파이프라인 관리 ───

    def define_pipeline(self, name: str, steps: List[dict]) -> None:
        """
        파이프라인 정의
        
        Args:
            name: 파이프라인 이름
            steps: 실행 단계 리스트
                [
                    {"module": "yolo"},
                    {"module": "context_llm", "condition": lambda results: ...},
                    {"module": "server_reporter", "condition": lambda results: ...},
                ]
        """
        self._pipelines[name] = steps
        module_names = [s["module"] for s in steps]
        logger.info(f"[Orchestrator] 파이프라인 정의: {name} → {' → '.join(module_names)}")

    def run_pipeline(self, pipeline_name: str, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            pipeline_name: 실행할 파이프라인 이름
            shared_data: 공유 데이터 (frame, timestamp 등)
        
        Returns:
            각 모듈의 결과를 { module_name: result_dict } 형태로 반환
        """
        if pipeline_name not in self._pipelines:
            logger.error(f"파이프라인 없음: {pipeline_name}")
            return {}

        results: Dict[str, Any] = {}
        shared_data.setdefault("results", {})

        for step in self._pipelines[pipeline_name]:
            module_name = step["module"]
            module = self._modules.get(module_name)

            if not module or not module.is_ready:
                continue

            # 조건 체크 (이전 결과에 따라 실행 여부)
            condition = step.get("condition")
            if condition and not condition(results):
                logger.debug(f"⏭️ 조건 불충족, 스킵: {module_name}")
                continue

            # 모듈 실행
            shared_data["results"] = results  # 이전 모듈 결과 전달
            result = module.safe_process(shared_data)
            results[module_name] = result

        return results

    def get_highest_priority(self, results: Dict[str, Any]) -> str:
        """모든 결과 중 가장 높은 우선순위 반환"""
        for priority in self.PRIORITY_ORDER:
            for module_result in results.values():
                if isinstance(module_result, dict) and module_result.get("priority") == priority:
                    return priority
        return "LOW"

    # ─── 전체 모듈 관리 ───

    def shutdown_all(self) -> None:
        """모든 모듈 안전 종료"""
        logger.info("[Orchestrator] 전체 종료 시작...")
        for name, module in self._modules.items():
            try:
                module.shutdown()
                logger.info(f"  ✅ {name} 종료 완료")
            except Exception as e:
                logger.error(f"  ❌ {name} 종료 오류: {e}")
        
        self.event_bus.shutdown()
        logger.info("[Orchestrator] 전체 종료 완료")
