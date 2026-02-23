# Mic Array + ContextLLM Fusion

`mic_context_fusion/` is a focused bridge runner that wires only:
- `MicArrayModule` (DOA detection + PTZ move trigger)
- `STTModule` (speech to text + non-speech audio candidate event)
- `ContextLLMService` (text + camera frame multimodal analysis, config-first)

## Workflow

1. `mic.doa_detected` occurs from mic array.
2. PTZ moves to detected DOA sector (absolute move).
3. `stt.text_recognized` (speech) or `stt.non_speech_audio` (speech-unrecognized audio) occurs.
4. Non-speech path uses YAMNet (`SoundEventDetector`) to classify emergency-like events.
5. After PTZ settle delay, current camera frame is captured.
6. `ContextLLMService.analyze_frame(text, frame)` runs with DOA + non-speech context.
7. If analyzed, `fusion.analysis_complete` is published.
8. Emergency results are re-emitted as `llm.emergency` for compatibility.

This keeps the event sequence:
`sound detected -> camera moves -> moved camera frame captured -> multimodal urgency analysis`
for both speech and non-speech risk sounds.

## Run

From project root:

```bash
python mic_context_fusion/main.py
```

With custom config:

```bash
python mic_context_fusion/main.py \
  --config mic_context_fusion/config.yaml
```

With direct API key input:

```bash
python mic_context_fusion/main.py --openai-api-key sk-...
```

## Main Files

- `mic_context_fusion/main.py`
- `mic_context_fusion/config.yaml`

## Notes

- This folder imports existing modules from `integrated_system` and `contextllm/src/app`.
- LLM analysis is executed via `contextllm/src/app/service.py` (not legacy CLI argument wiring).
- `OPENAI_API_KEY` priority in this runner:
  - `--openai-api-key` (highest)
  - `mic_context_fusion/config.yaml` -> `openai.api_key`
  - existing env (`.env` or shell env)
- Non-speech tuning is in `mic_context_fusion/config.yaml` -> `non_speech`.
  - detector defaults come from `contextllm/config/config.yaml` -> `sound_event`
  - fusion config can override threshold keywords only when needed
- You can still set key in:
  - `.env`, or
  - `contextllm/config/.env`
