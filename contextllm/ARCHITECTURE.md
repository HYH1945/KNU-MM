# ContextLLM Architecture

## Goals
- Keep core analysis logic stable.
- Minimize CLI coupling.
- Provide a programmatic API for external orchestrators.
- Keep config as the single source of truth.

## Layering

### 1) Entry Layer
- `main.py`
- Responsibility: parse minimal CLI (`--config`, `--mode`, `--show-config`) and delegate.

### 2) App Layer
- `src/app/settings.py`
- `src/app/runner.py`
- `src/app/service.py`
- Responsibility:
  - parse/validate config into typed settings
  - execute mode flows
  - expose integration-friendly service API

### 3) Core Layer
- `src/core/*`
- Responsibility: audio/video capture, multimodal analysis, voice features, dashboard bridge.
- Core remains reusable and independent from CLI details.

## Integration Pattern

Use `ContextLLMService` instead of shelling out to CLI:

1. Load one config snapshot.
2. Reuse one initialized system/analyzer.
3. Call `analyze_frame()` or `analyze_image()` from external system.

This avoids:
- process spawn overhead
- config drift between systems
- duplicated init cost per call

## Config Governance

- Runtime config synchronization is handled by `core.config_manager.set_runtime_config(...)`.
- Prompt and thresholds are versioned in `config/config.yaml`.
- Keep prompt schema stable (`priority`, `urgency`, `is_emergency`, `action`, etc.) so downstream consumers are not broken.
