# VoxNote — Speech to Text

Prototype giao diện cho nền tảng chuyển audio/video thành văn bản, phân biệt người nói và tóm tắt cuộc họp.

## Chạy frontend tĩnh

Không cần cài dependency. Mở `index.html` trực tiếp trong trình duyệt hoặc dùng một static server bất kỳ.

## Chạy đầy đủ frontend + backend

Yêu cầu Python 3.11 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Mở `http://127.0.0.1:8000`. API docs có tại `http://127.0.0.1:8000/docs`.

Backend mặc định dùng `TRANSCRIPTION_PROVIDER=demo`, vì vậy chạy được ngay mà không cần API key. File tải lên được lưu trong `runtime/uploads/` và không được đưa vào Git.

## Tính năng prototype

- Tải file bằng cách chọn hoặc kéo thả.
- Kiểm tra định dạng, dung lượng và mô phỏng từng giai đoạn xử lý.
- Bản ghi được chia theo từng người nói.
- Tìm kiếm nội dung hoặc tên người nói trong transcript.
- Đổi tên người nói và ghi nhớ tên trên trình duyệt.
- Tóm tắt, quyết định và công việc cần theo dõi.
- Đánh dấu công việc hoàn thành và xuất transcript dạng TXT.
- Responsive cho desktop, tablet và mobile.
- Hỗ trợ điều hướng bàn phím và reduced motion.

## Cấu trúc

- `index.html`: giao diện và nội dung.
- `styles.css`: design system và responsive layout.
- `app.js`: tương tác tải file và trạng thái xử lý.
- `backend/`: API upload, job xử lý nền và adapter transcription.
- `tests/`: kiểm thử API.
- `design-system/`: quy chuẩn thiết kế được tạo bởi UI/UX Pro Max.

MVP hiện đã có luồng end-to-end ở chế độ demo. Bước tiếp theo là thêm adapter transcription thật, lưu trạng thái job vào PostgreSQL và đưa file lên object storage.
