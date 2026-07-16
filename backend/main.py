from __future__ import annotations

import os
import re
import uuid
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

app = FastAPI(title="VoxNote API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

store = JobStore()
transcriber = create_transcriber(os.getenv("TRANSCRIPTION_PROVIDER", "demo"))


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
def health() -> dict:
    return {"status": "ok", "provider": os.getenv("TRANSCRIPTION_PROVIDER", "demo")}


@app.post("/api/jobs", status_code=202)
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
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

    job = store.create(job_id, original_name, stored_path)
    background_tasks.add_task(process_job, job_id, store, transcriber)
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ.")
    return job


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    job = store.delete(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ.")
    Path(job["_stored_path"]).unlink(missing_ok=True)


@app.get("/styles.css", include_in_schema=False)
def styles() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app.js", media_type="text/javascript")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html", media_type="text/html")
