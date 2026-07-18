from pathlib import Path

from backend.diarization import SpeakerTurn, assign_speakers, single_speaker_layout
from backend.transcription import LocalWhisperTranscriber
from backend.local_analysis import RuleBasedMeetingAnalyzer

from tests.test_local_ai import FakeWhisperModel


def seg(index: int, start: float, end: float, text: str = "…") -> dict:
    return {
        "id": f"segment-{index}",
        "speaker_id": "speaker-1",
        "start_seconds": start,
        "end_seconds": end,
        "text": text,
        "confidence": 0.9,
    }


def test_assign_speakers_by_overlap_and_appearance_order() -> None:
    segments = [seg(1, 0.0, 5.0), seg(2, 8.0, 15.0), seg(3, 18.0, 23.0)]
    # Cluster thô đánh số ngược (giọng đầu tiên là cluster 1) — hiển thị phải theo
    # thứ tự xuất hiện: cluster 1 -> "Người nói 1", cluster 0 -> "Người nói 2".
    turns = [
        SpeakerTurn(0.0, 6.5, speaker=1),
        SpeakerTurn(8.0, 16.0, speaker=0),
        SpeakerTurn(17.5, 23.0, speaker=1),
    ]

    result, speakers = assign_speakers(segments, turns)

    assert [s["speaker_id"] for s in result] == ["speaker-1", "speaker-2", "speaker-1"]
    assert [s["name"] for s in speakers] == ["Người nói 1", "Người nói 2"]
    assert speakers[0]["color"] == "speaker-one"
    assert speakers[1]["color"] == "speaker-two"


def test_assign_speakers_without_turns_falls_back_to_single() -> None:
    segments = [seg(1, 0.0, 5.0)]

    result, speakers = assign_speakers(segments, [])

    assert result[0]["speaker_id"] == "speaker-1"
    assert len(speakers) == 1


def test_segment_without_overlap_goes_to_nearest_turn() -> None:
    # Segment nằm trong khoảng lặng giữa hai lượt — phải về lượt GẦN nhất (lượt 2).
    segments = [seg(1, 9.6, 9.9)]
    turns = [SpeakerTurn(0.0, 5.0, speaker=0), SpeakerTurn(10.0, 15.0, speaker=1)]

    result, _speakers = assign_speakers(segments, turns)

    assert result[0]["speaker_id"] == "speaker-1"  # lượt 2 xuất hiện đầu tiên với segment này


def test_single_speaker_layout_empty_segments() -> None:
    result, speakers = single_speaker_layout([])
    assert result == [] and speakers == []


class FakeDiarizer:
    name = "fake"

    def diarize(self, _audio) -> list[SpeakerTurn]:
        return [SpeakerTurn(0.0, 4.0, speaker=0), SpeakerTurn(4.0, 8.0, speaker=1)]


def test_transcriber_reports_multiple_speakers(tmp_path: Path, monkeypatch) -> None:
    import pytest as _pytest  # noqa: F401

    monkeypatch.setattr(
        LocalWhisperTranscriber, "_decode_audio", staticmethod(lambda _path: b"fake-audio")
    )
    transcriber = LocalWhisperTranscriber(
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language="vi",
        download_root=tmp_path / "models",
        local_files_only=True,
        analyzer=RuleBasedMeetingAnalyzer(),
        diarizer=FakeDiarizer(),
    )
    transcriber._models["tiny"] = FakeWhisperModel()

    result = transcriber.transcribe(tmp_path / "hop-2-nguoi.wav", lambda *_args: None)

    assert result["speaker_count"] == 2
    assert {s["speaker_id"] for s in result["segments"]} == {"speaker-1", "speaker-2"}
    assert [s["name"] for s in result["speakers"]] == ["Người nói 1", "Người nói 2"]
