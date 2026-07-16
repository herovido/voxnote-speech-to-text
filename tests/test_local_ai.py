from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.local_analysis import (
    OllamaMeetingAnalyzer,
    ResilientLocalAnalyzer,
    RuleBasedMeetingAnalyzer,
)
from backend.transcription import LocalWhisperTranscriber


class FakeWhisperModel:
    def transcribe(self, _path: str, **_kwargs):
        segments = iter(
            [
                SimpleNamespace(
                    text="Nhóm thống nhất chạy AI trên máy local.",
                    start=0.0,
                    end=4.5,
                    avg_logprob=-0.1,
                ),
                SimpleNamespace(
                    text="Lan cần hoàn thành kiểm thử ngay.",
                    start=4.5,
                    end=8.0,
                    avg_logprob=-0.2,
                ),
            ]
        )
        return segments, SimpleNamespace(duration=8.0, language="vi")


def test_local_whisper_normalizes_real_model_shape(tmp_path: Path) -> None:
    transcriber = LocalWhisperTranscriber(
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language="vi",
        download_root=tmp_path / "models",
        local_files_only=True,
        analyzer=RuleBasedMeetingAnalyzer(),
    )
    transcriber._model = FakeWhisperModel()
    progress: list[int] = []

    result = transcriber.transcribe(tmp_path / "hop-tuan.wav", lambda value, _stage: progress.append(value))

    assert result["language"] == "vi"
    assert result["speaker_count"] == 1
    assert result["duration_seconds"] == 8
    assert len(result["segments"]) == 2
    assert result["decisions"][0].endswith("Nhóm thống nhất chạy AI trên máy local.")
    assert result["action_items"][0]["priority"] == "high"
    assert max(progress) >= 82


def test_ollama_rejects_non_local_url() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OllamaMeetingAnalyzer(base_url="https://example.com")


def test_ollama_enriches_missing_due_from_timestamped_evidence() -> None:
    analyzer = OllamaMeetingAnalyzer()
    result = {
        "summary": "Local AI",
        "decisions": [],
        "action_items": [
            {"id": "task-1", "text": "Alice review transcript", "assignee": "Alice", "due": "Chưa rõ"}
        ],
        "analysis_mode": analyzer.name,
    }

    enriched = analyzer._enrich(
        result,
        "[00:07] Người nói 1: Alice needs to review the transcript before Friday.",
    )

    assert enriched["action_items"][0]["due"] == "Friday"


def test_resilient_analyzer_falls_back_without_network() -> None:
    class BrokenLocalAnalyzer:
        name = "broken-local"

        def analyze(self, _transcript: str) -> dict:
            raise ValueError("local model unavailable")

    analyzer = ResilientLocalAnalyzer(primary=BrokenLocalAnalyzer())
    result = analyzer.analyze("Chúng ta cần kiểm thử ngay.")

    assert result["analysis_mode"] == "local-rules"
    assert result["action_items"]
