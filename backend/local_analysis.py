from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("voxnote.analysis")


class MeetingAnalyzer(Protocol):
    """Create meeting insights without sending transcript data off-device."""

    name: str

    def analyze(self, transcript: str) -> dict:
        """Return a normalized summary, decisions, and action items."""


def _sentences(transcript: str) -> list[str]:
    compact = re.sub(r"\s+", " ", transcript).strip()
    if not compact:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+", compact) if part.strip()]


_LINE_PREFIX = re.compile(r"^\[\d{1,3}:\d{2}\]\s*(?:Người nói \d+\s*:|Speaker \d+\s*:)?\s*")


def _strip_speaker_prefixes(transcript: str) -> str:
    """Bỏ tiền tố '[mm:ss] Người nói 1:' để summary/decision không dính rác timestamp."""
    return "\n".join(_LINE_PREFIX.sub("", line) for line in transcript.splitlines())


def _unique(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.casefold().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(item.strip())
        if len(output) >= limit:
            break
    return output


def _extract_due(text: str) -> str | None:
    patterns = (
        # "chủ nhật" phải đứng riêng (không ai nói "thứ chủ nhật"); "thứ 2..7" dạng số rất phổ biến.
        r"\b(?:trước|vào|đến|hạn(?: chót)?(?: là)?|deadline(?: là)?)\s+"
        r"((?:thứ\s+(?:hai|ba|tư|năm|sáu|bảy|[2-7]))|chủ\s+nhật|hôm nay|ngày mai|tuần này|tuần sau|"
        r"ngày\s+\d{1,2}(?:[./-]\d{1,2}(?:[./-]\d{2,4})?)?)",
        r"\b(?:before|by|on)\s+"
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow|next week|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s+\d{4})?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


@dataclass(slots=True)
class RuleBasedMeetingAnalyzer:
    """Offline fallback used when the local Ollama service is unavailable."""

    name: str = "local-rules"

    def analyze(self, transcript: str) -> dict:
        sentences = _sentences(_strip_speaker_prefixes(transcript))
        if not sentences:
            return {
                "summary": "Không phát hiện nội dung lời nói rõ ràng trong file.",
                "decisions": [],
                "action_items": [],
                "analysis_mode": self.name,
            }

        summary = " ".join(sentences[:3])
        # Transcript có thể là bất kỳ ngôn ngữ nào Whisper nhận được — kèm từ khóa
        # tiếng Anh để fallback không trả rỗng với cuộc họp không phải tiếng Việt.
        decision_terms = (
            "quyết định", "thống nhất", "chốt", "đồng ý", "sẽ triển khai",
            "decided", "agreed", "approved", "final decision",
        )
        action_terms = (
            "cần", "phải", "hãy", "giao cho", "deadline", "hoàn thành",
            "need to", "needs to", "must", "should", "will", "assigned",
            "finish", "complete", "follow up",
        )
        decisions = _unique(
            [sentence for sentence in sentences if any(term in sentence.casefold() for term in decision_terms)],
            limit=5,
        )
        action_sentences = _unique(
            [sentence for sentence in sentences if any(term in sentence.casefold() for term in action_terms)],
            limit=8,
        )
        action_items = [
            {
                "id": f"task-{index}",
                "text": sentence,
                "assignee": "Chưa giao",
                "due": "Chưa đặt",
                "priority": "high" if any(term in sentence.casefold() for term in ("gấp", "khẩn", "ngay")) else "medium",
            }
            for index, sentence in enumerate(action_sentences, start=1)
        ]
        return {
            "summary": summary,
            "decisions": decisions,
            "action_items": action_items,
            "analysis_mode": self.name,
        }


MEETING_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "assignee": {"type": "string"},
                    "due": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["text", "assignee", "due", "priority"],
            },
        },
    },
    "required": ["summary", "decisions", "action_items"],
}


def _short_field(value: object, fallback: str) -> str:
    """Người nhận / hạn chót là cụm NGẮN. Chuỗi dài bất thường hoặc nhiều dòng là
    dấu hiệu model local trượt sang rác token (đã quan sát với qwen2.5:7b) —
    trả fallback để _enrich suy lại từ chính bản ghi thay vì hiển thị rác."""
    text = str(value or "").strip()
    if not text or len(text) > 60 or "\n" in text:
        return fallback
    return text


def _local_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("OLLAMA_BASE_URL chỉ được phép trỏ tới dịch vụ loopback trên máy local.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_BASE_URL không được chứa thông tin đăng nhập, query hoặc fragment.")
    return value.rstrip("/")


@dataclass(slots=True)
class OllamaMeetingAnalyzer:
    """Summarize transcripts with an Ollama model running on this machine."""

    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 180.0
    chunk_chars: int = 12_000
    max_chunks: int = 32
    context_tokens: int = 32_768
    name: str = "local-ollama"
    _endpoint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._endpoint = f"{_local_base_url(self.base_url)}/api/generate"
        self.name = f"local-ollama:{self.model}"

    def analyze(self, transcript: str) -> dict:
        chunks = self._chunks(transcript)
        if not chunks:
            return RuleBasedMeetingAnalyzer().analyze(transcript)

        partials = [
            self._generate(
                "Bạn là trợ lý cuộc họp chạy hoàn toàn trên máy local. "
                "Hãy phân tích đoạn bản ghi dưới đây bằng tiếng Việt. Không suy diễn dữ kiện không có trong bản ghi. "
                "Giữ lại tên người, thời hạn, con số và mốc thời gian làm bằng chứng. "
                "Tóm tắt phải bao quát mọi chủ đề chính trong 3 đến 6 câu. "
                "Mỗi công việc phải giữ nguyên người thực hiện và thời hạn nếu chúng xuất hiện trong cùng câu. "
                "Nếu không có quyết định hoặc công việc thì trả về mảng rỗng. "
                "assignee và due dùng 'Chưa rõ' khi bản ghi không nêu.\n\n"
                f"ĐOẠN {index}/{len(chunks)}:\n{chunk}",
                evidence_text=chunk,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        if len(partials) == 1:
            return partials[0]

        evidence = json.dumps(partials, ensure_ascii=False)
        return self._generate(
            "Hãy tổng hợp các kết quả phân tích từng đoạn của cùng một cuộc họp. "
            "Loại nội dung trùng lặp, giữ nguyên tên người/thời hạn/con số, không thêm dữ kiện mới, "
            "và tạo một tóm tắt xuyên suốt toàn bộ cuộc họp.\n\n"
            f"KẾT QUẢ TỪNG ĐOẠN:\n{evidence}",
            evidence_text=transcript,
        )

    def _generate(self, prompt: str, evidence_text: str) -> dict:
        response = httpx.post(
            self._endpoint,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": MEETING_SCHEMA,
                "options": {"temperature": 0, "num_ctx": self.context_tokens},
                "keep_alive": "10m",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        generated = body.get("response")
        if not isinstance(generated, str):
            raise ValueError("Ollama không trả về trường response hợp lệ.")
        return self._enrich(self._normalize(json.loads(generated)), evidence_text)

    def _chunks(self, transcript: str) -> list[str]:
        lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for line in lines:
            if current and current_length + len(line) + 1 > self.chunk_chars:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
                if len(chunks) >= self.max_chunks:
                    break
            current.append(line)
            current_length += len(line) + 1

        if current and len(chunks) < self.max_chunks:
            chunks.append("\n".join(current))
        return chunks

    def _normalize(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Kết quả Ollama không phải JSON object.")

        summary = str(payload.get("summary", "")).strip()[:2_000]
        if not summary:
            raise ValueError("Ollama không tạo được nội dung tóm tắt.")

        raw_decisions = payload.get("decisions", [])
        decisions = _unique(
            [str(item).strip()[:500] for item in raw_decisions if str(item).strip()]
            if isinstance(raw_decisions, list)
            else [],
            limit=8,
        )

        action_items: list[dict] = []
        raw_actions = payload.get("action_items", [])
        if isinstance(raw_actions, list):
            for index, item in enumerate(raw_actions[:12], start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()[:500]
                if not text:
                    continue
                priority = str(item.get("priority", "medium")).lower()
                if priority not in {"high", "medium", "low"}:
                    priority = "medium"
                action_items.append(
                    {
                        "id": f"task-{index}",
                        "text": text,
                        "assignee": _short_field(item.get("assignee"), "Chưa rõ"),
                        "due": _short_field(item.get("due"), "Chưa rõ"),
                        "priority": priority,
                    }
                )

        return {
            "summary": summary,
            "decisions": decisions,
            "action_items": action_items,
            "analysis_mode": self.name,
        }

    def _enrich(self, result: dict, evidence_text: str) -> dict:
        lines = [line.strip() for line in evidence_text.splitlines() if line.strip()]
        for action in result["action_items"]:
            due = str(action.get("due", "")).strip().casefold()
            if due not in {"", "chưa rõ", "chưa đặt", "unknown", "none", "n/a"}:
                continue

            assignee = str(action.get("assignee", "")).strip()
            candidates = [line for line in lines if assignee and assignee.casefold() in line.casefold()]
            if not candidates:
                action_words = {
                    word.casefold()
                    for word in re.findall(r"\w+", str(action.get("text", "")), flags=re.UNICODE)
                    if len(word) >= 4
                }
                candidates = [
                    line
                    for line in lines
                    if len(action_words.intersection(word.casefold() for word in re.findall(r"\w+", line))) >= 2
                ]

            # CHỈ suy hạn từ những dòng thực sự liên quan (khớp assignee hoặc từ khóa
            # của công việc). Quét toàn bộ transcript sẽ gán bừa ngày của câu không
            # liên quan vào công việc — tức là bịa deadline.
            if not candidates:
                continue
            extracted = _extract_due(" ".join(candidates))
            if extracted:
                action["due"] = extracted
        return result


@dataclass(slots=True)
class ResilientLocalAnalyzer:
    """Prefer a local LLM and remain fully offline with a rules fallback."""

    primary: MeetingAnalyzer | None
    fallback: MeetingAnalyzer = field(default_factory=RuleBasedMeetingAnalyzer)
    name: str = "local-only"

    def __post_init__(self) -> None:
        self.name = f"{self.primary.name}+{self.fallback.name}" if self.primary else self.fallback.name

    def analyze(self, transcript: str) -> dict:
        if self.primary is not None:
            try:
                return self.primary.analyze(transcript)
            except (httpx.HTTPError, json.JSONDecodeError, ValueError, OSError) as exc:
                # Fallback là hành vi đúng, nhưng lý do fail phải thấy được trong log
                # server, không được nuốt im lặng.
                logger.warning("Analyzer %s lỗi (%s: %s) — chuyển sang %s",
                               self.primary.name, type(exc).__name__, exc, self.fallback.name)
        return self.fallback.analyze(transcript)
