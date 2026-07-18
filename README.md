# VoxNote — Local Speech to Text

VoxNote chuyển audio/video thành bản ghi và phân tích nội dung bằng các mô hình chạy trên chính máy của bạn. Runtime không gọi OpenAI API hay dịch vụ AI bên ngoài.

## Local AI pipeline

1. `faster-whisper` nhận diện ngôn ngữ của file (khi `LOCAL_ASR_LANGUAGE` để trống), ghim ngôn ngữ đó cho cả file.
2. Nếu ngôn ngữ có model chuyên biệt trong `LOCAL_ASR_MODEL_OVERRIDES` (mặc định: Khmer), toàn bộ file được transcribe bằng model đó; ngược lại dùng model chính (`large-v3`).
3. Transcript dài được chia thành các đoạn có timestamp.
4. Ollama local (`qwen2.5:7b`) tóm tắt, trích xuất quyết định và công việc; kết quả từng đoạn được tổng hợp thêm một lượt để giữ context toàn cuộc họp.
5. Nếu Ollama chưa chạy, hệ thống tự chuyển sang bộ phân tích rule-based offline (hỗ trợ từ khóa tiếng Việt + tiếng Anh); transcript vẫn hoạt động.

`OLLAMA_BASE_URL` bị giới hạn trong code ở `127.0.0.1`, `localhost` hoặc `::1`. Cấu hình URL bên ngoài sẽ bị từ chối khi khởi động.

## Chất lượng theo ngôn ngữ (đo thực tế trên RTX 3080, 2026-07)

| Ngôn ngữ | Model dùng | Kết quả |
|---|---|---|
| Tiếng Việt | `large-v3` | Gần như chính xác tuyệt đối, confidence ~0.9 |
| Tiếng Anh | `large-v3` | Gần như chính xác tuyệt đối, confidence ~0.9 |
| Tiếng Khmer | `large-v3` | **Rác hoàn toàn** (confidence ~0.4) — điểm yếu cố hữu của Whisper với ngôn ngữ ít dữ liệu |
| Tiếng Khmer | `PhanithLIM/whisper-tiny-khmer-ct2` (qua overrides) | Đúng gần từng chữ, confidence 0.99 |

Đây là lý do tồn tại của cơ chế `LOCAL_ASR_MODEL_OVERRIDES` — không model đơn lẻ nào tốt cho mọi ngôn ngữ.

## Trạng thái tính năng

- Nhận dạng audio/video thật bằng Whisper local, tự nhận diện ngôn ngữ (99 ngôn ngữ).
- Routing model theo ngôn ngữ nhận diện được (Khmer mặc định).
- Tự phát hiện vùng có tiếng nói và bỏ khoảng lặng.
- Transcript có mốc thời gian và độ tin cậy.
- Tóm tắt phân cấp cho cuộc họp dài bằng Ollama local (chống rác token: temperature 0 + chặn field bất thường).
- Trích xuất quyết định và công việc; hạn chót chỉ được suy từ dòng liên quan, không bịa.
- Tìm kiếm transcript (highlight không dùng innerHTML — an toàn XSS), đổi tên người nói, đánh dấu công việc theo từng cuộc họp, xuất TXT.
- Job đã xong + file upload tự dọn sau `JOB_RETENTION_HOURS` (mặc định 24h); chặn quá tải bằng `MAX_ACTIVE_JOBS`.
- Job vẫn lưu trong RAM (restart là mất); file mồ côi sau restart được sweep dọn.
- Bản hiện tại chưa diarization: tất cả nội dung được gom vào `Người nói 1`. Tách nhiều người nói là bước tiếp theo.
- Chưa kiểm tra nội dung codec trước khi decode — chỉ chặn theo đuôi file và dung lượng (đủ cho dùng local, chưa đủ nếu mở ra Internet).

## Yêu cầu

- Python 3.11 trở lên (đã test với 3.14).
- Ollama đang chạy trên máy + model `qwen2.5:7b` (`ollama pull qwen2.5:7b`).
- Lần chạy Whisper đầu tiên cần Internet để tải model vào `runtime/models/`; sau đó có thể đặt `LOCAL_MODELS_ONLY=true`.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
ollama pull qwen2.5:7b
uvicorn backend.main:app --reload --port 8000
```

Mở `http://127.0.0.1:8000`. API docs ở `http://127.0.0.1:8000/docs`.

### Chạy GPU NVIDIA

```powershell
pip install -r requirements-gpu.txt
```

rồi trong `.env` đặt `LOCAL_ASR_DEVICE=cuda` + `LOCAL_ASR_COMPUTE_TYPE=float16`. Hai wheel `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` cung cấp DLL cần thiết; trên Windows backend **tự thêm chúng vào PATH** lúc nạp model — không cần cài CUDA Toolkit hay chỉnh PATH thủ công. Yêu cầu driver NVIDIA hỗ trợ CUDA 12.

Máy chỉ có CPU vẫn chạy được với cấu hình mặc định (`cpu` + `int8`), chậm hơn đáng kể với model `large-v3` — cân nhắc hạ xuống `medium`.

## Cấu hình model

Xem `.env.example` (đã có chú thích từng mục). Các điểm đáng chú ý:

- `LOCAL_ASR_LANGUAGE` **để trống** = tự nhận diện rồi ghim ngôn ngữ cho cả file. Đặt cứng `vi` chỉ khi mọi file chắc chắn là tiếng Việt (khi đó prompt tối ưu tiếng Việt được bật).
- `LOCAL_ASR_MODEL_OVERRIDES=km=PhanithLIM/whisper-tiny-khmer-ct2` — thêm ngôn ngữ khác dạng `lang=model,lang2=model2`; model phải ở định dạng CTranslate2.
- `large-v3-turbo` nhanh hơn ~4× nhưng bị cắt decoder 32→4 lớp, suy giảm rõ với ngôn ngữ ít dữ liệu — chỉ dùng khi tốc độ quan trọng hơn độ chính xác.
- `OLLAMA_MODEL`: `qwen2.5:7b` cho chất lượng tiếng Việt tốt nhất trong các model đã thử nghiệm (gemma4:12b làm hỏng chữ tiếng Việt và bỏ sót quyết định).

## Kiểm thử

```powershell
pytest -q
python -m scripts.smoke_local_ai duong-dan/audio.mp3 --model large-v3 --language "" --device cuda --compute-type float16
```

Test API dùng provider demo cố định để không tải model. `tests/test_local_ai.py` kiểm tra chuẩn hóa output Whisper, routing model theo ngôn ngữ, chống bịa hạn chót, guard rác token, local-only URL và fallback khi Ollama không sẵn sàng.

## Cấu trúc

- `backend/transcription.py`: adapter demo + faster-whisper local, routing model theo ngôn ngữ, tự nạp DLL CUDA trên Windows.
- `backend/local_analysis.py`: Ollama local, hierarchical summary, guard chống rác/bịa, offline fallback đa ngôn ngữ.
- `backend/jobs.py`: job store trong RAM, xử lý nền, prune theo retention.
- `backend/main.py`: FastAPI upload/job API, retention sweep, giới hạn hàng đợi, frontend tĩnh.
- `index.html`, `styles.css`, `app.js`: giao diện web.
- `tests/`: kiểm thử API và pipeline local AI.

Nguồn tham khảo: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3), [whisper-tiny-khmer-ct2](https://huggingface.co/PhanithLIM/whisper-tiny-khmer-ct2), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
