from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Transcriber(Protocol):
    """Contract for a speech-to-text provider."""

    def transcribe(self, file_path: Path, on_progress) -> dict:
        """Return a normalized meeting result."""


@dataclass(slots=True)
class DemoTranscriber:
    """Deterministic provider used until a real API key is configured."""

    def transcribe(self, file_path: Path, on_progress) -> dict:
        on_progress(42, "Đang nhận dạng giọng nói…")
        on_progress(68, "Đang phân biệt người nói…")
        on_progress(86, "Đang tạo tóm tắt…")

        title = file_path.stem.replace("_", " ").replace("-", " ").strip().title()
        if not title:
            title = "Cuộc họp mới"

        return {
            "title": title,
            "language": "vi",
            "duration_seconds": 2060,
            "speaker_count": 3,
            "speakers": [
                {"id": "speaker-1", "name": "Minh Long", "initials": "ML", "color": "speaker-one"},
                {"id": "speaker-2", "name": "Hà Anh", "initials": "HA", "color": "speaker-two"},
                {"id": "speaker-3", "name": "Quang Nam", "initials": "QN", "color": "speaker-three"},
            ],
            "segments": [
                {
                    "id": "segment-1",
                    "speaker_id": "speaker-1",
                    "start_seconds": 762,
                    "end_seconds": 774,
                    "text": "Tuần này chúng ta cần chốt trải nghiệm tải file. Mục tiêu là người dùng hiểu ngay hệ thống đang làm gì và còn phải chờ bao lâu.",
                    "confidence": 0.96,
                },
                {
                    "id": "segment-2",
                    "speaker_id": "speaker-2",
                    "start_seconds": 775,
                    "end_seconds": 793,
                    "text": "Em đã chuẩn bị ba trạng thái: đang tải lên, đang nhận dạng giọng nói và đang tạo bản tóm tắt. Mỗi trạng thái đều có tiến độ cụ thể.",
                    "confidence": 0.94,
                },
                {
                    "id": "segment-3",
                    "speaker_id": "speaker-3",
                    "start_seconds": 798,
                    "end_seconds": 817,
                    "text": "Phần phân biệt người nói nên cho phép đổi tên một lần và áp dụng cho toàn bộ cuộc họp. Đây là tính năng quan trọng nhất của bản đầu.",
                    "confidence": 0.93,
                },
            ],
            "summary": "Nhóm thống nhất ưu tiên thử nghiệm luồng tải file và cải thiện độ chính xác cho tiếng Việt trước khi mở rộng tính năng.",
            "decisions": [
                "Ưu tiên hoàn thiện trải nghiệm tải file trong MVP.",
                "Tên người nói được đổi một lần và áp dụng cho toàn bộ cuộc họp.",
            ],
            "action_items": [
                {"id": "task-1", "text": "Hoàn thiện luồng tải file", "assignee": "Hà Anh", "due": "Hôm nay", "priority": "high"},
                {"id": "task-2", "text": "Kiểm thử nhận dạng tiếng Việt", "assignee": "Quang Nam", "due": "Thứ sáu", "priority": "medium"},
                {"id": "task-3", "text": "Chốt tiêu chí nghiệm thu MVP", "assignee": "Minh Long", "due": "Tuần này", "priority": "medium"},
            ],
        }


def create_transcriber(provider: str) -> Transcriber:
    normalized = provider.strip().lower()
    if normalized == "demo":
        return DemoTranscriber()
    raise ValueError(
        f"TRANSCRIPTION_PROVIDER={provider!r} chưa được hỗ trợ. "
        "Dùng 'demo' cho đến khi adapter dịch vụ thật được cấu hình."
    )
