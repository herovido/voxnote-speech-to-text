from __future__ import annotations

import math
import os
import sys
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

VIETNAMESE_INITIAL_PROMPT = (
    "Đây là bản ghi cuộc họp bằng tiếng Việt. "
    "Giữ nguyên tên riêng, thuật ngữ kỹ thuật và dấu câu."
)


def _windows_cuda_dll_dirs() -> list[str]:
    """Thư mục DLL cuBLAS/cuDNN do pip wheel nvidia-* cài kèm trong venv."""
    root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    return [str(root / name / "bin") for name in ("cublas", "cudnn") if (root / name / "bin").is_dir()]


def _ensure_cuda_dlls_on_path() -> None:
    """CTranslate2 trên Windows tìm cudnn/cublas DLL qua PATH lúc inference.

    Nếu venv có wheel nvidia-cublas-cu12 / nvidia-cudnn-cu12 thì tự thêm vào PATH
    để `uvicorn backend.main:app` chạy GPU được ngay, không cần chỉnh PATH thủ công.
    """
    if os.name != "nt":
        return
    current = os.environ.get("PATH", "")
    missing = [d for d in _windows_cuda_dll_dirs() if d not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing) + os.pathsep + current


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
    """Run multilingual Whisper inference locally through faster-whisper.

    Khi `language` để trống, ngôn ngữ được nhận diện tự động; nếu ngôn ngữ đó có
    model chuyên biệt trong `model_overrides` (vd Khmer — large-v3 rất yếu với km)
    thì toàn bộ file được transcribe bằng model đó thay vì model chính.
    """

    model_name: str
    device: str
    compute_type: str
    language: str | None
    download_root: Path
    local_files_only: bool
    analyzer: MeetingAnalyzer
    model_overrides: dict[str, str] = field(default_factory=dict)
    mode: str = "local"
    _models: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _model_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _inference_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def describe(self) -> dict:
        return {
            "provider": self.mode,
            "local_ai": True,
            "asr_model": self.model_name,
            "asr_device": self.device,
            "asr_model_overrides": dict(self.model_overrides),
            "analysis_provider": self.analyzer.name,
        }

    def _load_model(self, model_name: str) -> Any:
        with self._model_lock:
            model = self._models.get(model_name)
            if model is not None:
                return model
            if self.device.startswith("cuda"):
                _ensure_cuda_dlls_on_path()
            from faster_whisper import WhisperModel

            self.download_root.mkdir(parents=True, exist_ok=True)
            model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root),
                local_files_only=self.local_files_only,
            )
            self._models[model_name] = model
            return model

    @staticmethod
    def _decode_audio(file_path: Path) -> Any:
        from faster_whisper.audio import decode_audio

        # detect_language yêu cầu MẢNG audio đã giải mã 16kHz, không nhận đường dẫn.
        return decode_audio(str(file_path), sampling_rate=16000)

    def _pick_model(self, file_path: Path, on_progress: ProgressCallback) -> tuple[Any, str | None, Any]:
        """Chọn (model, ngôn ngữ ghim, audio đã giải mã nếu có)."""
        language = self.language
        model_name = self.model_name

        if language:
            model_name = self.model_overrides.get(language, model_name)
            return self._load_model(model_name), language, None

        on_progress(20, "Đang giải mã audio…")
        audio = self._decode_audio(file_path)
        on_progress(22, "Đang nhận diện ngôn ngữ trên máy…")
        primary = self._load_model(model_name)
        detected, _probability, _all = primary.detect_language(audio, vad_filter=True)
        detected = str(detected or "").lower()
        override = self.model_overrides.get(detected)
        if override and override != model_name:
            on_progress(
                26,
                f"Ngôn ngữ '{detected}' có mô hình chuyên biệt — chuyển sang {override}…",
            )
            return self._load_model(override), detected, audio
        # Ghim ngôn ngữ đã nhận diện để Whisper không đổi ngôn ngữ giữa file.
        return primary, (detected or None), audio

    def transcribe(self, file_path: Path, on_progress: ProgressCallback) -> dict:
        with self._inference_lock:
            model, language, audio = self._pick_model(file_path, on_progress)
            on_progress(32, "Đang nhận dạng giọng nói hoàn toàn trên máy…")
            raw_segments, info = model.transcribe(
                # Tái dùng audio đã giải mã ở bước nhận diện; numpy array phải so
                # sánh bằng `is not None` (truthiness của array là ambiguous).
                audio if audio is not None else str(file_path),
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=True,
                word_timestamps=False,
                initial_prompt=VIETNAMESE_INITIAL_PROMPT if language == "vi" else None,
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
            "language": str(getattr(info, "language", language or "unknown")),
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


def parse_model_overrides(raw: str) -> dict[str, str]:
    """'km=PhanithLIM/whisper-tiny-khmer-ct2,ja=...' -> {'km': '...', 'ja': '...'}"""
    overrides: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        language, separator, model = pair.partition("=")
        if not separator or not language.strip() or not model.strip():
            raise ValueError(
                "LOCAL_ASR_MODEL_OVERRIDES phải có dạng 'lang=model[,lang2=model2]', "
                f"gặp phần không hợp lệ: {pair!r}"
            )
        overrides[language.strip().lower()] = model.strip()
    return overrides


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
        language = os.getenv("LOCAL_ASR_LANGUAGE", "").strip() or None
        return LocalWhisperTranscriber(
            model_name=os.getenv("LOCAL_ASR_MODEL", "large-v3"),
            device=os.getenv("LOCAL_ASR_DEVICE", "cpu"),
            compute_type=os.getenv("LOCAL_ASR_COMPUTE_TYPE", "int8"),
            language=language,
            download_root=root / os.getenv("LOCAL_MODEL_DIR", "runtime/models"),
            local_files_only=_env_bool("LOCAL_MODELS_ONLY"),
            analyzer=_create_local_analyzer(),
            model_overrides=parse_model_overrides(os.getenv("LOCAL_ASR_MODEL_OVERRIDES", "")),
        )
    raise ValueError("TRANSCRIPTION_PROVIDER chỉ hỗ trợ 'local' hoặc 'demo'.")
