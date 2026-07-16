from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .local_analysis import (
    MeetingAnalyzer,
    OllamaMeetingAnalyzer,
    ResilientLocalAnalyzer,
    RuleBasedMeetingAnalyzer,
)


ProgressCallback = Callable[[int, str], None]


class Transcriber(Protocol):
    """Contract for a speech-to-text provider."""

    mode: str

    def transcribe(self, file_path: Path, on_progress: ProgressCallback) -> dict:
        """Return a normalized meeting result."""

    def describe(self) -> dict:
        """Return non-sensitive provider configuration for health checks."""


@dataclass(slots=True)
class DemoTranscriber:
    """Deterministic local provider kept for automated tests and UI demos."""

    mode: str = "demo"

    def describe(self) -> dict:
        return {"provider": self.mode, "local_ai": True, "asr_model": "deterministic-demo"}

    def transcribe(self, file_path: Path, on_progress: ProgressCallback) -> dict:
        on_progress(42, "Đang nhận dạng giọng nói…")
        on_progress(68, "Đang chuẩn hóa bản ghi…")
        on_progress(86, "Đang tạo tóm tắt local…")

        title = file_path.stem.replace("_", " ").replace("-", " ").strip().title() or "Cuộc họp mới"
        return {
            "title": title,
            "language": "vi",
            "duration_seconds": 2060,
            "speaker_count": 1,
            "speakers": [
                {"id": "speaker-1", "name": "Người nói 1", "initials": "N1", "color": "speaker-one"},
            ],
            "segments": [
                {
                    "id": "segment-1",
                    "speaker_id": "speaker-1",
                    "start_seconds": 0,
                    "end_seconds": 12,
                    "text": "Tuần này chúng ta cần chốt trải nghiệm tải file và kiểm thử nhận dạng tiếng Việt.",
                    "confidence": 0.96,
                },
                {
                    "id": "segment-2",
                    "speaker_id": "speaker-1",
                    "start_seconds": 13,
                    "end_seconds": 30,
                    "text": "Nhóm thống nhất toàn bộ mô hình AI phải chạy trên máy local.",
                    "confidence": 0.94,
                },
            ],
            "summary": "Nhóm ưu tiên trải nghiệm tải file, nhận dạng tiếng Việt và chạy toàn bộ AI trên máy local.",
            "decisions": ["Toàn bộ mô hình AI phải chạy trên máy local."],
            "action_items": [
                {
                    "id": "task-1",
                    "text": "Kiểm thử nhận dạng tiếng Việt",
                    "assignee": "Chưa giao",
                    "due": "Chưa đặt",
                    "priority": "high",
                }
            ],
            "analysis_mode": "local-demo",
        }


@dataclass(slots=True)
class LocalWhisperTranscriber:
    """Run multilingual Whisper inference locally through faster-whisper."""

    model_name: str
    device: str
    compute_type: str
    language: str | None
    download_root: Path
    local_files_only: bool
    analyzer: MeetingAnalyzer
    mode: str = "local"
    _model: Any = field(default=None, init=False, repr=False)
    _model_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _inference_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def describe(self) -> dict:
        return {
            "provider": self.mode,
            "local_ai": True,
            "asr_model": self.model_name,
            "asr_device": self.device,
            "analysis_provider": self.analyzer.name,
        }

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            self.download_root.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root),
                local_files_only=self.local_files_only,
            )
            return self._model

    def transcribe(self, file_path: Path, on_progress: ProgressCallback) -> dict:
        on_progress(24, f"Đang nạp mô hình Whisper {self.model_name} trên {self.device.upper()}…")
        model = self._load_model()
        on_progress(32, "Đang nhận dạng giọng nói hoàn toàn trên máy…")

        with self._inference_lock:
            raw_segments, info = model.transcribe(
                str(file_path),
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=True,
                word_timestamps=False,
                initial_prompt=(
                    "Đây là bản ghi cuộc họp bằng tiếng Việt. "
                    "Giữ nguyên tên riêng, thuật ngữ kỹ thuật và dấu câu."
                    if self.language == "vi"
                    else None
                ),
            )
            duration = max(float(getattr(info, "duration", 0) or 0), 0.0)
            segments: list[dict] = []
            for index, segment in enumerate(raw_segments, start=1):
                text = str(getattr(segment, "text", "")).strip()
                if not text:
                    continue
                start = max(float(getattr(segment, "start", 0) or 0), 0.0)
                end = max(float(getattr(segment, "end", start) or start), start)
                avg_logprob = float(getattr(segment, "avg_logprob", -1.0) or -1.0)
                segments.append(
                    {
                        "id": f"segment-{index}",
                        "speaker_id": "speaker-1",
                        "start_seconds": round(start, 2),
                        "end_seconds": round(end, 2),
                        "text": text,
                        "confidence": round(max(0.0, min(1.0, math.exp(avg_logprob))), 3),
                    }
                )
                if duration > 0:
                    progress = 32 + int(min(end / duration, 1.0) * 45)
                    on_progress(progress, "Đang nhận dạng giọng nói hoàn toàn trên máy…")

        if segments and duration <= 0:
            duration = float(segments[-1]["end_seconds"])
        transcript = "\n".join(
            f"[{int(segment['start_seconds']) // 60:02d}:{int(segment['start_seconds']) % 60:02d}] "
            f"Người nói 1: {segment['text']}"
            for segment in segments
        )
        on_progress(82, "Đang tóm tắt bằng Local AI…")
        analysis = self.analyzer.analyze(transcript)

        title = file_path.stem.replace("_", " ").replace("-", " ").strip().title() or "Cuộc họp mới"
        return {
            "title": title,
            "language": str(getattr(info, "language", self.language or "unknown")),
            "duration_seconds": round(duration),
            "speaker_count": 1 if segments else 0,
            "speakers": (
                [{"id": "speaker-1", "name": "Người nói 1", "initials": "N1", "color": "speaker-one"}]
                if segments
                else []
            ),
            "segments": segments,
            **analysis,
        }


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _create_local_analyzer() -> MeetingAnalyzer:
    provider = os.getenv("LOCAL_LLM_PROVIDER", "ollama").strip().lower()
    primary: MeetingAnalyzer | None
    if provider == "ollama":
        primary = OllamaMeetingAnalyzer(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
            context_tokens=int(os.getenv("OLLAMA_CONTEXT_TOKENS", "32768")),
        )
    elif provider in {"rules", "none"}:
        primary = None
    else:
        raise ValueError("LOCAL_LLM_PROVIDER chỉ hỗ trợ 'ollama' hoặc 'rules'.")
    return ResilientLocalAnalyzer(primary=primary, fallback=RuleBasedMeetingAnalyzer())


def create_transcriber(provider: str, project_root: Path | None = None) -> Transcriber:
    normalized = provider.strip().lower()
    if normalized == "demo":
        return DemoTranscriber()
    if normalized == "local":
        root = project_root or Path(__file__).resolve().parents[1]
        language = os.getenv("LOCAL_ASR_LANGUAGE", "vi").strip() or None
        return LocalWhisperTranscriber(
            model_name=os.getenv("LOCAL_ASR_MODEL", "large-v3-turbo"),
            device=os.getenv("LOCAL_ASR_DEVICE", "cpu"),
            compute_type=os.getenv("LOCAL_ASR_COMPUTE_TYPE", "int8"),
            language=language,
            download_root=root / os.getenv("LOCAL_MODEL_DIR", "runtime/models"),
            local_files_only=_env_bool("LOCAL_MODELS_ONLY"),
            analyzer=_create_local_analyzer(),
        )
    raise ValueError("TRANSCRIPTION_PROVIDER chỉ hỗ trợ 'local' hoặc 'demo'.")
