from __future__ import annotations

import logging
import threading
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .transcription import Transcriber

logger = logging.getLogger("voxnote.jobs")


class JobStore:
    """Small thread-safe in-memory job store for the MVP."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, job_id: str, original_name: str, stored_path: Path, mode: str) -> dict[str, Any]:
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 5,
            "stage": "Đã nhận file, đang xếp hàng…",
            "original_name": original_name,
            "created_at": datetime.now(UTC).isoformat(),
            "mode": mode,
            "result": None,
            "error": None,
            "_stored_path": str(stored_path),
        }
        with self._lock:
            self._jobs[job_id] = job
        return self.public(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self.public(job) if job else None

    def raw(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    def delete(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job["status"] in {"queued", "processing"})

    def prune_finished_before(self, cutoff: datetime) -> list[str]:
        """Xóa job đã xong/lỗi cũ hơn cutoff; trả về đường dẫn file để dọn."""
        removed: list[str] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job["status"] not in {"completed", "failed"}:
                    continue
                stamp = job.get("completed_at") or job["created_at"]
                try:
                    finished_at = datetime.fromisoformat(stamp)
                except (TypeError, ValueError):
                    continue
                if finished_at < cutoff:
                    removed.append(job["_stored_path"])
                    del self._jobs[job_id]
        return removed

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(job)
        result.pop("_stored_path", None)
        return result


def _title_from_name(name: str) -> str:
    stem = Path(name).stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() or "Cuộc họp mới"


def process_job(job_id: str, store: JobStore, transcriber: Transcriber) -> None:
    job = store.raw(job_id)
    if not job:
        return

    def report(progress: int, stage: str) -> None:
        store.update(job_id, status="processing", progress=progress, stage=stage)

    try:
        report(18, "Đang kiểm tra và chuẩn hóa file…")
        result = transcriber.transcribe(Path(job["_stored_path"]), report)
        # File trên đĩa mang tên job-id; tiêu đề phải lấy từ TÊN GỐC người dùng tải lên.
        result["title"] = _title_from_name(job["original_name"])
        report(96, "Đang hoàn thiện bản ghi…")
        store.update(
            job_id,
            status="completed",
            progress=100,
            stage="Bản ghi đã sẵn sàng",
            result=result,
            completed_at=datetime.now(UTC).isoformat(),
        )
    except Exception:  # The API must expose a recoverable state, not crash the worker.
        # Chi tiết lỗi (đường dẫn, stack trace nội bộ) chỉ ghi log server,
        # không trả nguyên văn cho client.
        logger.exception("Job %s xử lý thất bại", job_id)
        store.update(
            job_id,
            status="failed",
            stage="Không thể xử lý file",
            error="Xử lý thất bại trên máy chủ. Hãy thử lại hoặc xem log server để biết chi tiết.",
            completed_at=datetime.now(UTC).isoformat(),
        )

