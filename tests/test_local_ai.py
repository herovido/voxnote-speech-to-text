from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.local_analysis import (
    OllamaMeetingAnalyzer,
    ResilientLocalAnalyzer,
    RuleBasedMeetingAnalyzer,
    _extract_due,
)
from backend.transcription import LocalWhisperTranscriber, parse_model_overrides


class FakeWhisperModel:
    """Mô phỏng WhisperModel: transcribe trả generator lazy + detect_language."""

    def __init__(self, language: str = "vi", texts: list[str] | None = None) -> None:
        self.language = language
        self.texts = texts or [
            "Nhóm thống nhất chạy AI trên máy local.",
            "Lan cần hoàn thành kiểm thử ngay.",
        ]
        self.detect_calls = 0

    def detect_language(self, _path: str, **_kwargs):
        self.detect_calls += 1
        return self.language, 0.99, []

    def transcribe(self, _path: str, **kwargs):
        step = 4.0
        segments = iter(
            SimpleNamespace(
                text=text,
                start=index * step,
                end=(index + 1) * step,
                avg_logprob=-0.1 if index == 0 else -0.2,
            )
            for index, text in enumerate(self.texts)
        )
        info = SimpleNamespace(
            duration=len(self.texts) * step,
            language=kwargs.get("language") or self.language,
        )
        return segments, info


def make_transcriber(tmp_path: Path, **kwargs) -> LocalWhisperTranscriber:
    defaults = {
        "model_name": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "language": "vi",
        "download_root": tmp_path / "models",
        "local_files_only": True,
        "analyzer": RuleBasedMeetingAnalyzer(),
    }
    defaults.update(kwargs)
    return LocalWhisperTranscriber(**defaults)


def test_local_whisper_normalizes_real_model_shape(tmp_path: Path) -> None:
    transcriber = make_transcriber(tmp_path)
    transcriber._models["tiny"] = FakeWhisperModel()
    progress: list[int] = []

    result = transcriber.transcribe(tmp_path / "hop-tuan.wav", lambda value, _stage: progress.append(value))

    assert result["language"] == "vi"
    assert result["speaker_count"] == 1
    assert result["duration_seconds"] == 8
    assert len(result["segments"]) == 2
    assert result["decisions"][0].endswith("Nhóm thống nhất chạy AI trên máy local.")
    assert result["action_items"][0]["priority"] == "high"
    assert max(progress) >= 82


def test_auto_detect_routes_to_language_override_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = FakeWhisperModel(language="km", texts=["văn bản từ model chính"])
    khmer = FakeWhisperModel(language="km", texts=["អត្ថបទពីម៉ូដែលខ្មែរ"])
    monkeypatch.setattr(
        LocalWhisperTranscriber, "_decode_audio", staticmethod(lambda _path: b"fake-audio")
    )
    transcriber = make_transcriber(
        tmp_path,
        model_name="primary-fake",
        language=None,
        model_overrides={"km": "khmer-fake"},
    )
    transcriber._models.update({"primary-fake": primary, "khmer-fake": khmer})

    result = transcriber.transcribe(tmp_path / "kech-brochum.mp3", lambda *_args: None)

    assert primary.detect_calls == 1
    assert result["language"] == "km"
    assert result["segments"][0]["text"] == "អត្ថបទពីម៉ូដែលខ្មែរ"


def test_pinned_language_uses_override_without_detection(tmp_path: Path) -> None:
    primary = FakeWhisperModel()
    khmer = FakeWhisperModel(language="km", texts=["អត្ថបទ"])
    transcriber = make_transcriber(
        tmp_path,
        model_name="primary-fake",
        language="km",
        model_overrides={"km": "khmer-fake"},
    )
    transcriber._models.update({"primary-fake": primary, "khmer-fake": khmer})

    result = transcriber.transcribe(tmp_path / "kech-brochum.mp3", lambda *_args: None)

    assert primary.detect_calls == 0
    assert result["segments"][0]["text"] == "អត្ថបទ"


def test_parse_model_overrides() -> None:
    assert parse_model_overrides("") == {}
    assert parse_model_overrides("km=PhanithLIM/whisper-tiny-khmer-ct2, JA = org/model ") == {
        "km": "PhanithLIM/whisper-tiny-khmer-ct2",
        "ja": "org/model",
    }
    with pytest.raises(ValueError, match="lang=model"):
        parse_model_overrides("km")


def test_extract_due_matches_chu_nhat_and_numeric_weekdays() -> None:
    assert _extract_due("hạn chót là chủ nhật") == "chủ nhật"
    assert _extract_due("hoàn thành trước thứ 2") == "thứ 2"
    assert _extract_due("deadline là thứ sáu") == "thứ sáu"


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


def test_enrich_never_fabricates_due_from_unrelated_lines() -> None:
    analyzer = OllamaMeetingAnalyzer()
    result = {
        "summary": "Local AI",
        "decisions": [],
        "action_items": [
            {"id": "task-1", "text": "Hoàn thiện brandbook", "assignee": "Chưa rõ", "due": "Chưa rõ"}
        ],
        "analysis_mode": analyzer.name,
    }

    enriched = analyzer._enrich(
        result,
        "[00:01] Người nói 1: Sếp đi công tác vào thứ sáu.\n"
        "[00:09] Người nói 2: Cần làm lại bộ nhận diện thương hiệu.",
    )

    # Ngày của câu không liên quan KHÔNG được gán vào công việc.
    assert enriched["action_items"][0]["due"] == "Chưa rõ"


def test_normalize_drops_abnormally_long_short_fields() -> None:
    analyzer = OllamaMeetingAnalyzer()
    payload = {
        "summary": "Tóm tắt hợp lệ.",
        "decisions": [],
        "action_items": [
            {
                "text": "Việc A",
                "assignee": "Lan",
                "due": "Thứ Sáu (không rõ thời gian cụ thể)." + "rác token trượt " * 10,
                "priority": "high",
            }
        ],
    }

    result = analyzer._normalize(payload)

    assert result["action_items"][0]["assignee"] == "Lan"
    assert result["action_items"][0]["due"] == "Chưa rõ"


def test_normalize_scrubs_cjk_garbage_when_meeting_has_none() -> None:
    analyzer = OllamaMeetingAnalyzer()
    payload = {
        "summary": "Tóm tắt hợp lệ.",
        "decisions": ["Dùng mô hình local"],
        "action_items": [
            {
                "text": "Chuẩn bị demo trước thứ sá六大模型无法直接翻译或生成诗歌",
                "assignee": "Nam",
                "due": "Trước thứ sá六, 08:00:00 2023（周五）",
                "priority": "high",
            }
        ],
    }

    result = analyzer._normalize(payload, "Tôi sẽ chuẩn bị bản demo trước thứ sáu.")

    item = result["action_items"][0]
    assert item["text"] == "Chuẩn bị demo trước thứ sá"  # text dài: cắt đuôi giữ phần hợp lệ
    assert item["due"] == "Chưa rõ"  # field ngắn dính CJK lạ: bỏ hẳn, để _enrich suy lại
    assert item["assignee"] == "Nam"


def test_normalize_keeps_cjk_for_chinese_meetings() -> None:
    analyzer = OllamaMeetingAnalyzer()
    payload = {
        "summary": "会议总结。",
        "decisions": [],
        "action_items": [
            {"text": "准备演示", "assignee": "小明", "due": "周五", "priority": "high"}
        ],
    }

    result = analyzer._normalize(payload, "小明说：我周五之前准备演示。")

    assert result["summary"] == "会议总结。"
    assert result["action_items"][0]["due"] == "周五"


def test_rule_based_handles_english_and_strips_timestamps() -> None:
    transcript = (
        "[00:00] Người nói 1: We decided to launch the beta on Friday.\n"
        "[00:05] Người nói 1: John needs to finish the upload flow."
    )

    result = RuleBasedMeetingAnalyzer().analyze(transcript)

    assert "[00:" not in result["summary"]
    assert result["decisions"]
    assert result["action_items"]


def test_resilient_analyzer_falls_back_without_network() -> None:
    class BrokenLocalAnalyzer:
        name = "broken-local"

        def analyze(self, _transcript: str) -> dict:
            raise ValueError("local model unavailable")

    analyzer = ResilientLocalAnalyzer(primary=BrokenLocalAnalyzer())
    result = analyzer.analyze("Chúng ta cần kiểm thử ngay.")

    assert result["analysis_mode"] == "local-rules"
    assert result["action_items"]
