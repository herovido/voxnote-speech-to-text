from __future__ import annotations

import os
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from .jobs import JobStore, process_job
from .transcription import create_transcriber


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
RUNTIME_DIR = PROJECT_ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".ogg"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
JOB_RETENTION_SECONDS = float(os.getenv("JOB_RETENTION_HOURS", "24")) * 3600
MAX_ACTIVE_JOBS = int(os.getenv("MAX_ACTIVE_JOBS", "10"))

app = FastAPI(title="VoxNote API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

store = JobStore()
transcriber = create_transcriber(os.getenv("TRANSCRIPTION_PROVIDER", "local"), PROJECT_ROOT)


def unlink_quietly(path: Path) -> None:
    """Windows không cho xóa file đang được job nền đọc — bỏ qua, sweep sau sẽ dọn."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def sweep_expired_data() -> None:
    """Thi hành JOB_RETENTION_HOURS: dọn job đã xong + file upload quá hạn.

    Quét thêm file mồ côi trong UPLOAD_DIR vì job store nằm trong RAM —
    restart server là mất job nhưng file thì còn.
    """
    if JOB_RETENTION_SECONDS <= 0:
        return
    cutoff = datetime.now(UTC) - timedelta(seconds=JOB_RETENTION_SECONDS)
    for stored_path in store.prune_finished_before(cutoff):
        unlink_quietly(Path(stored_path))
    cutoff_ts = time.time() - JOB_RETENTION_SECONDS
    for path in UPLOAD_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff_ts:
                path.unlink()
        except OSError:
            continue


sweep_expired_data()


def safe_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"Định dạng chưa hỗ trợ. Cho phép: {allowed}")
    return extension


def clean_display_name(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^\w.()\- ]", "_", name, flags=re.UNICODE)[:180] or "meeting-audio"


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", **transcriber.describe()}


@app.post("/api/jobs", status_code=202)
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    sweep_expired_data()
    if store.active_count() >= MAX_ACTIVE_JOBS:
        raise HTTPException(
            status_code=429,
            detail="Hệ thống đang xử lý quá nhiều file. Vui lòng thử lại sau ít phút.",
        )
    original_name = clean_display_name(file.filename or "meeting-audio")
    extension = safe_extension(original_name)
    job_id = uuid.uuid4().hex
    stored_path = UPLOAD_DIR / f"{job_id}{extension}"
    written = 0

    try:
        with stored_path.open("wb") as target:
            while chunk := await file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                target.write(chunk)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File tải lên đang trống.")

    job = store.create(job_id, original_name, stored_path, mode=transcriber.mode)
    background_tasks.add_task(process_job, job_id, store, transcriber)
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ.")
    return job


@app.delete("/api/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    job = store.delete(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ.")
    unlink_quietly(Path(job["_stored_path"]))


@app.get("/styles.css", include_in_schema=False)
def styles() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app.js", media_type="text/javascript")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html", media_type="text/html")
