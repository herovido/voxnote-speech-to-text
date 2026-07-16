import asyncio

import httpx

from backend.main import app


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health() -> None:
    response = request("GET", "/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_get_completed_job() -> None:
    response = request(
        "POST",
        "/api/jobs",
        files={"file": ("hop-tuan.mp3", b"demo audio", "audio/mpeg")},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    job_response = request("GET", f"/api/jobs/{job_id}")
    assert job_response.status_code == 200
    payload = job_response.json()
    assert payload["status"] == "completed"
    assert payload["result"]["speaker_count"] == 3
    assert len(payload["result"]["segments"]) == 3


def test_rejects_unsupported_extension() -> None:
    response = request(
        "POST",
        "/api/jobs",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415
