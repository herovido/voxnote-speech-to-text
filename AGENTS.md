# Working in VoxNote

Read [README](README.md) and the [workflow skill](.agents/skills/voxnote-workflow/SKILL.md). This repository combines static UI with FastAPI, local transcription, diarization, and local analysis.

## Boundaries
- `backend/main.py` handles upload/job APIs and file retention; `jobs.py` holds in-memory jobs; `transcription.py` chooses transcription/language; `diarization.py` assigns speakers; `local_analysis.py` produces summaries and tasks.
- Preserve local transcript processing and the loopback-only analysis URL validation. Initial model download and inference are separate operations.
- Keep upload limits, safe filenames, active-job retention, and failure fallbacks. Do not send recordings or transcript text to a cloud provider as a fallback.
- For UI work, read [design source](design-system/speech-to-text/MASTER.md) and keep untrusted transcript text escaped. Some controls are placeholders; do not claim authentication or audio playback without tracing implementation.

## Verification
From root: `python -m compileall -q backend scripts tests`, `node --check app.js`.
With development dependencies installed: `python -m pytest -q`. Existing tests use demo/fake transcription and local analysis fakes; they do not establish real-model quality. Importing the API creates/sweeps runtime upload state, so use an isolated checkout for tests.
`python -m scripts.smoke_local_ai <audio-path> --local-files-only` is a real-model check; use an authorized fixture and already-installed models. Do not download heavy weights for a documentation change.
Report syntax, mocked behavior, and real inference checks separately.

