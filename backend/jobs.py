from __future__ import annotations

import threading
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .transcription import Transcriber


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

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(job)
        result.pop("_stored_path", None)
        return result


def process_job(job_id: str, store: JobStore, transcriber: Transcriber) -> None:
    job = store.raw(job_id)
    if not job:
        return

    def report(progress: int, stage: str) -> None:
        store.update(job_id, status="processing", progress=progress, stage=stage)

    try:
        report(18, "Đang kiểm tra và chuẩn hóa file…")
        result = transcriber.transcribe(Path(job["_stored_path"]), report)
        report(96, "Đang hoàn thiện bản ghi…")
        store.update(
            job_id,
            status="completed",
            progress=100,
            stage="Bản ghi đã sẵn sàng",
            result=result,
            completed_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:  # The API must expose a recoverable state, not crash the worker.
        store.update(
            job_id,
            status="failed",
            stage="Không thể xử lý file",
            error=str(exc),
            completed_at=datetime.now(UTC).isoformat(),
        )

