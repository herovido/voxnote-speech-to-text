# VoxNote — Local Speech to Text

VoxNote chuyển audio/video thành bản ghi và phân tích nội dung bằng các mô hình chạy trên chính máy của bạn. Runtime không gọi OpenAI API hay dịch vụ AI bên ngoài.

## Local AI pipeline

1. `faster-whisper` chạy `large-v3-turbo` để nhận dạng tiếng Việt và tạo transcript có mốc thời gian.
2. Transcript dài được chia thành các đoạn có timestamp.
3. Ollama local chạy `qwen2.5:7b` để tóm tắt, trích xuất quyết định và công việc.
4. Kết quả từng đoạn được tổng hợp thêm một lượt để giữ context của toàn cuộc họp.
5. Nếu Ollama chưa chạy, hệ thống tự chuyển sang bộ phân tích rule-based offline; transcript vẫn hoạt động.

`OLLAMA_BASE_URL` bị giới hạn trong code ở `127.0.0.1`, `localhost` hoặc `::1`. Cấu hình URL bên ngoài sẽ bị từ chối khi khởi động.

## Trạng thái tính năng

- Nhận dạng audio/video thật bằng Whisper local.
- Tự phát hiện vùng có tiếng nói và bỏ khoảng lặng.
- Transcript có mốc thời gian và độ tin cậy.
- Tóm tắt phân cấp cho cuộc họp dài bằng Ollama local.
- Trích xuất quyết định và công việc sang giao diện.
- Tìm kiếm transcript, đổi tên người nói, đánh dấu công việc và xuất TXT.
- Job vẫn lưu trong RAM và file upload vẫn lưu trong `runtime/uploads/`.
- Bản hiện tại chưa diarization: tất cả nội dung được gom vào `Người nói 1`. Tách nhiều người nói là bước tiếp theo.

## Yêu cầu

- Python 3.9 trở lên.
- Ollama đang chạy trên máy.
- Model Ollama `qwen2.5:7b`.
- Lần chạy Whisper đầu tiên cần Internet để tải model vào `runtime/models/`; sau đó có thể bật chế độ chỉ dùng file local.

Máy chỉ có CPU vẫn chạy được với `int8`. GPU NVIDIA cần CUDA 12, cuBLAS và cuDNN 9 tương thích với CTranslate2.

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

## Cấu hình model

Các giá trị mặc định trong `.env.example`:

```dotenv
TRANSCRIPTION_PROVIDER=local
LOCAL_ASR_MODEL=large-v3-turbo
LOCAL_ASR_DEVICE=cpu
LOCAL_ASR_COMPUTE_TYPE=int8
LOCAL_ASR_LANGUAGE=vi
LOCAL_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_CONTEXT_TOKENS=32768
```

Sau khi model Whisper đã tải xong, đặt `LOCAL_MODELS_ONLY=true` để ngăn mọi lần tải model tiếp theo.

### Lựa chọn model

- `large-v3-turbo`: mặc định cân bằng tốt giữa độ chính xác và tốc độ, hỗ trợ đa ngôn ngữ.
- `large-v3`: chính xác hơn một chút nhưng nặng và chậm hơn.
- `medium` hoặc `small`: dùng khi máy yếu hoặc cần xử lý nhanh hơn.
- `vinai/PhoWhisper-large`: lựa chọn chuyên tiếng Việt tốt, nhưng cần adapter Transformers/PyTorch riêng và chưa được bật trong pipeline hiện tại.

## Kiểm thử

```powershell
pytest -q
python -m scripts.smoke_local_ai runtime/samples/your-audio.wav --local-files-only
```

Test API dùng provider demo cố định để không tải model. `tests/test_local_ai.py` kiểm tra việc chuẩn hóa output Whisper, local-only URL và fallback khi Ollama không sẵn sàng.

## Cấu trúc

- `backend/transcription.py`: adapter demo và faster-whisper local.
- `backend/local_analysis.py`: Ollama local, hierarchical summary và offline fallback.
- `backend/jobs.py`: job store trong RAM và xử lý nền.
- `backend/main.py`: FastAPI upload/job API và frontend tĩnh.
- `index.html`, `styles.css`, `app.js`: giao diện web.
- `tests/`: kiểm thử API và pipeline local AI.

Nguồn tham khảo: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
