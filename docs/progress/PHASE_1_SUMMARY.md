# Báo cáo Tiến trình: ĐÃ HOÀN THÀNH PHASE 1 (Foundation)

> File này ghi nhận toàn bộ các hạng mục kỹ thuật cốt lõi và bug-fixing đã thực thi thành công trong Phase 1 (Bao gồm cả bước chuyển tiếp Phase 1.5). Hệ thống nền tảng đã sẵn sàng để thu thập dữ liệu và chuyển giao sang Phase 2.

---

## PHÂN HỆ I: PHASE 1.0 (Nền móng hệ thống Cốt lõi)

### 1. Môi trường Container & Triển khai Jetson
- **Jetson AI Server Deployment:** Khởi tạo thành công kiến trúc Docker cho Jetson Orin Nano. Xử lý triệt để xung đột Dependency (NumPy 2.x so với PyTorch/L4T).
- **TensorRT Optimization:** Khắc phục lỗi Crash môi trường liên quan đến việc khởi tạo TensorRT Engine bên trong Docker container, bảo đảm mô hình YOLOv11s load và chạy bằng FP16 tăng tốc phần cứng. 

### 2. Giao thức Truyền thông (ZMQ & Protobuf)
- **Protobuf Compilation:** Sửa dứt điểm lỗi `ModuleNotFoundError: No module named 'pkg_resources'` chặn đường sinh mã của `grpcio-tools`, giúp tự động generate thành công các Python stubs `messages_pb2.py`.
- **ZMQ Subscriber Shutdown:** Bắt và fix triệt để lỗi `zmq.error.ZMQError: not a socket`. Implement luồng tắt thủ công an toàn (safe shutdown sequence), ngắt poller trước khi dập thiết bị socket, giúp CPU/thread trên Jetson không bị crash/rò rỉ khi kill process.

### 3. Perception Pipeline (Quy trình Nhận thức)
- **Tracking Ổn định:** Phục hồi và kích hoạt thành công thuật toán ByteTrack. Đảm bảo gán `track_id` ổn định qua từng frame.
- **Tính khoảng cách độ sâu thực (Depth-based Distance):** Thay thế logic tính vị trí bằng diện tích (Bounding Box area) sang đo trực tiếp bằng **Depth Camera**. Thay vì bị đánh lừa khi con người khom lưng hay bị ngã (bbox phình to), giờ đây khoảng cách luôn trả về chuẩn số mét (`depth_mm`).
- **Normalise Occupancy Grid:** Fix lỗi vỡ giá trị lưới chiếm dụng. Đảm bảo biểu diễn vật cản dưới góc nhìn ma trận cho State RL.
- **COCO Class Context Mapping:** Trực tiếp cấu hình thuật toán Mapping phân tích đối tượng từ YOLO 80-class tiêu chuẩn đổ về các nhóm chuyên biệt theo thực tế kho bãi/sân bay. Chỉ giữ lại xử lý `person`, `dynamic_hazard` (balo, vali, bóng bay), và `static_obstacle` (ghế băng, bàn, tv).

### 4. Luồng dữ liệu HDF5 (Experience Streamer)
- **Bug-fix HDF5 Segmentation Fault:** Xử lý tận gốc cơ chế `h5py` Garbage Collection làm sụp bộ nhớ. Vô hiệu hóa `gc` cục bộ trong block lưu file, khóa lock luồng an toàn giúp Jetson lưu dữ liệu 102 chiều vào `.h5` mượt mà không crash.
- **Safety Policy:** Đã thiết lập Heuristic fallback để làm mồi nhử lái robot lấy Data. Test chặn được robot (Hard stop) khi nhận thấy người < 0.5m.

---

## PHÂN HỆ II: PHASE 1.5 (Hạ tầng Huấn luyện Intent CNN)

### 1. ROI Dataset Extraction
- Implement logic trích xuất riêng biệt ảnh viền (crop box) cực kỳ nhỏ gọn bằng 128x256 của đối tượng.
- Lưu kèm ID thành định dạng file `roi_<track_id>_...jpg` để tiện quá trình gỡ lỗi theo chuỗi khung hình.

### 2. Auto-labeling & Metadata Sync
- Sinh thêm thư viện metadata phụ trợ `metadata.jsonl` đính vào ngay cạnh mỗi bức ảnh. 
- Chuẩn bị nền móng bắt sự tương quan giữa độ lệch tâm (Delta cx_cy) so với độ thay đổi chiều sâu Delta Depth để chuẩn bị tạo kịch bản Auto-label (Gán nhãn tự động chuẩn xác hơn dùng Bbox ảo).

### 3. Data Auto Sync (Daemon & Cap)
- Tạo một sidecar container độc lập chạy nền liên tục. Dùng thuật toán rsync qua phương thức bảo mật SSH để đẩy gọn gàng từng Batch file ROI ảnh lẫn file hưu trí HDF5 về Training Server mà không phụ thuộc lệnh chạy thủ công.
- Ngăn chặn triệt để tình trạng tràn bộ nhớ Flash trên bo mạch bằng cách giới hạn chính sách xóa cạn (Chỉ giữ tối đa 5,000 files local) trên môi trường Jetson.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Data Validation (Kiểm định tính toàn vẹn Dữ liệu HDF5)
- **Mục tiêu:** Đảm bảo hệ thống Async Sync và HDF5 Writer lưu đủ và chuẩn dữ liệu trước khi train model.
- **Hướng dẫn thực hiện chi tiết:** 
  - Mở thư mục chứa file sync từ Jetson (hoặc chạy thử trên tập HDF5 dummy).
  - Viết 1 script nhỏ `scripts/verify_hdf5.py`.
  - Quét qua file `session_*.h5` để in ra và kiểm tra: `image_jpeg` giải mã được không? Timestamp array có lệch không? Mảng `observation` có dính NaN/Zero toàn bộ hay không.

### 2. Triển khai Temporal State Stacking (Chuẩn bị Phase 2.0)
- **Mục tiêu:** Chuyển bộ State Observation từ góc nhìn "Điểm" (Snapshot) sang "Chuỗi thời gian" (Temporal) để phục vụ AI đoán ý định.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở file `src/navigation/context_builder.py`.
  - Bật config `temporal_stack_size = 3` (hoặc 5).
  - Sử dụng module `collections.deque` để lưu trữ lịch sử state, xoay đuôi liên tục.
  - Test kịch bản: Robot đứng yên nhưng người bước tới -> Mảng state history phải thể hiện khoảng cách `depth` đang nhỏ dần qua từng frame.

### 3. Xây dựng Intent Trajectory Buffer 
- **Mục tiêu:** Theo dõi lịch sử di chuyển mượt mà tách biệt cho từng `track_id` (Từng người riêng rẽ). Tránh nhầm lẫn người A với người B.
- **Hướng dẫn thực hiện chi tiết:**
  - Tạo cấu trúc dữ liệu mới `intent_trajectory.py`.
  - Liên kết với ByteTrack ID từ `FrameDetections`. Maintain Memory Dictionary lưu vết toạ độ (x, y, distance) của đối tượng trong N frames.
