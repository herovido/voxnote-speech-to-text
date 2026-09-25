---
name: voxnote-workflow
description: Develop VoxNote local transcription, diarization, meeting analysis, upload jobs, and transcript UI. Use when changes need privacy-preserving local behavior and separation of mocked tests from real-model validation.
---

# VoxNote workflow

1. Read [root instructions](../../../AGENTS.md). Trace [API](../../../backend/main.py) → [jobs](../../../backend/jobs.py) → [transcription](../../../backend/transcription.py) before editing job states or response schemas.
2. For language changes, check auto-detection, explicit language settings, per-language model overrides, and CJK filtering against the original transcript. Preserve the distinction between detected and user-selected language.
3. For speaker changes, inspect [diarization](../../../backend/diarization.py); preserve timestamp alignment, appearance-order numbering, and the fallback when diarization fails.
4. For analysis, inspect [local analysis](../../../backend/local_analysis.py). Preserve loopback-only endpoints, fallback rules, and grounding of tasks/deadlines in the transcript. Avoid assigning an unrelated meeting date to every task.
5. Run the relevant cases in [tests](../../../tests) with fakes first. For API tests, isolate runtime uploads. Only use [smoke script](../../../scripts/smoke_local_ai.py) with a permitted sample and ready local models when inference quality is in scope.
6. For UI changes, follow [design reference](../../../design-system/speech-to-text/MASTER.md), render transcript text safely, and verify task/speaker state across job switches. Clearly identify placeholders and checks that did not exercise actual audio inference.

