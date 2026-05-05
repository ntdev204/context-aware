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
- **Loại bỏ Protobuf cho Control Command:** Tối ưu hóa việc truyền lệnh `ai/nav_cmd` bằng định dạng nhị phân nguyên thủy `struct` (`!iffiffB`, 25 bytes) thay vì protobuf, giải quyết triệt để lỗi mismatch kiểu dữ liệu và giảm tối đa băng thông.
- **Khắc phục Bug ZMQ Multipart bị cắt xén do `CONFLATE`:** Loại bỏ cờ `zmq.CONFLATE` trên Jetson Publisher và thay thế bằng `RCVHWM = 2` trên Raspberry Pi Subscriber. Xử lý thành công lỗi mất phần Topic ID khiến Pi từ chối nhận lệnh điều hướng. Rebuild image Docker `jetson-prod` để nạp code mới.

### 3. Perception Pipeline & Control Policy

- **Tracking Ổn định:** Phục hồi và kích hoạt thành công thuật toán ByteTrack. Đảm bảo gán `track_id` ổn định qua từng frame.
- **Bổ sung Fallback cho Depth Camera:** Sửa lỗi `depth_cov=0%` (khi đối tượng quá gần hoặc ánh sáng yếu) gây kích hoạt nhầm Hard Stop. Áp dụng công thức ước lượng khoảng cách dự phòng bằng chiều cao Bounding Box của YOLO (`dist ≈ (focal_length × 1.7) / bbox_h`).
- **Nâng cấp Heuristic P-Controller:** Điều chỉnh giới hạn `velocity_scale` sang dải `[-1, 1]` và cho phép xuất vận tốc âm. Robot hiện tại có thể lùi về phía sau để giữ cự ly `2.0m` mục tiêu.
- **Normalise Occupancy Grid:** Fix lỗi vỡ giá trị lưới chiếm dụng. Đảm bảo biểu diễn vật cản dưới góc nhìn ma trận cho State RL.
- **COCO Class Context Mapping:** Trực tiếp cấu hình thuật toán Mapping phân tích đối tượng từ YOLO 80-class tiêu chuẩn đổ về các nhóm chuyên biệt theo thực tế kho bãi/sân bay. Chỉ giữ lại xử lý `person`, `dynamic_hazard` (balo, vali, bóng bay), và `static_obstacle` (ghế băng, bàn, tv).

### 4. Luồng dữ liệu HDF5 (Experience Streamer)

- **Bug-fix HDF5 Segmentation Fault:** Xử lý tận gốc cơ chế `h5py` Garbage Collection làm sụp bộ nhớ. Vô hiệu hóa `gc` cục bộ trong block lưu file, khóa lock luồng an toàn giúp Jetson lưu dữ liệu 102 chiều vào `.h5` mượt mà không crash.
- **Safety Policy:** Đã thiết lập Heuristic fallback để làm mồi nhử lái robot lấy Data. Test chặn được robot (Hard stop) khi nhận thấy người < 2.0m.

---

## PHÂN HỆ II: PHASE 1.5 (Hạ tầng Huấn luyện Intent CNN)

### 1. ROI Dataset Extraction

- Implement logic trích xuất riêng biệt ảnh viền (crop box) cực kỳ nhỏ gọn bằng 128x256 của đối tượng.
- Lưu kèm ID thành định dạng file `roi_<track_id>_...jpg` để tiện quá trình gỡ lỗi theo chuỗi khung hình.

### 2. Auto-labeling, Quality Gate & Metadata Sync (Nâng cấp Ego-Motion)

- Phân tách dứt điểm logic dán nhãn bằng `autolabel.py` qua thuật toán cửa sổ trượt (Sliding Window N=5).
- **Robot Ego-Motion Compensation:** Áp dụng thuật toán bù trừ tịnh tiến và góc xoay (`vx`, `vy`, `vtheta`) từ robot trước khi lấy hiệu số `delta_depth`. Tránh false-positive nhãn `APPROACHING` do góc nhìn tương đối.
- Cập nhật luồng Core `ExperienceBuffer`: Gắn thêm tọa độ ngang viền đối tượng `cx, cy` và định danh chuỗi `track_id` vào dữ liệu HDF5 và JSON dự phòng, phục vụ đắc lực cho việc đoán ý định bằng Lateral Displacement. Nhãn residual cũ được chuyển sang `UNCERTAIN` để review, không còn dùng `FOLLOWING`.
- Tách nhãn `ERRATIC` phân bổ bằng phương sai (`variance`) thay vì Auto-label ngầm định nhằm kiểm soát Review.
- **Quality Gate:** Đưa logic kiểm tra ngưỡng cân bằng vào script `explore_roi.py`. Bật cảnh báo đa dạng dữ liệu `< 35%` nếu nhóm 3 class khó tổng cộng quá ít, bảo đảm an toàn dữ liệu Huấn luyện CNN.

### 3. Data Auto Sync (Daemon & Cap)

- Tạo một sidecar container độc lập chạy nền liên tục. Dùng thuật toán rsync qua phương thức bảo mật SSH để đẩy gọn gàng từng Batch file ROI ảnh lẫn file hưu trí HDF5 về Training Server mà không phụ thuộc lệnh chạy thủ công.
- Ngăn chặn triệt để tình trạng tràn bộ nhớ Flash trên bo mạch bằng cách giới hạn chính sách xóa cạn (Chỉ giữ tối đa 5,000 files local) trên môi trường Jetson.

### 4. Kiến trúc Điều khiển (Nav2 vs AI Override)

- **Phân luồng Quyền Điều khiển /cmd_vel:** Xác định vấn đề xung đột lệnh điều khiển giữa Local Planner (Nav2) và Jetson AI. Quyết định áp dụng kiến trúc Override: sử dụng `twist_mux` trên Raspberry Pi để ưu tiên lệnh bám người tốc độ cao (~16ms latency) từ Jetson, tạm thời bypass Nav2 Planner khi tương tác cự ly gần.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Cài đặt và cấu hình Twist Mux trên Raspberry Pi (Ưu tiên Cao)

- **Mục tiêu:** Xử lý triệt để khả năng xung đột lệnh lái xe để robot không bị giật khi vừa bật Nav2 vừa chạy Jetson.
- **Hướng dẫn thực hiện chi tiết:** 
  - Cài đặt trên Pi: `sudo apt install ros-humble-twist-mux`
  - Đổi topic trong file `bridge_node.py` sang `/cmd_vel_ai` và build lại.
  - Cấu hình file `twist_mux.yaml` ưu tiên: `teleop` > `cmd_vel_ai` > `cmd_vel_nav`.
  - Launch node `twist_mux` thay vì cho bridge trỏ thẳng vào `/cmd_vel`.

### 2. Data Validation (Kiểm định tính toàn vẹn Dữ liệu HDF5)

- **Mục tiêu:** Đảm bảo hệ thống Async Sync và HDF5 Writer lưu đủ và chuẩn dữ liệu trước khi train model.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở thư mục chứa file sync từ Jetson (hoặc chạy thử trên tập HDF5 dummy).
  - Viết 1 script nhỏ `scripts/verify_hdf5.py`.
  - Quét qua file `session_*.h5` để in ra và kiểm tra: `image_jpeg` giải mã được không? Timestamp array có lệch không? Mảng `observation` có dính NaN/Zero toàn bộ hay không. Mảng `person_cxs` có ghi đầy đủ không?

### 2. Manual Review Dữ liệu HDF5 (Các nhãn ERRATIC)

- **Mục tiêu:** Kiểm tra chéo bằng mắt người xem thuật toán `autolabel.py` gắn cờ `ERRATIC` do phương sai cao đã chính xác chưa.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở Label Studio hoặc tool explore tương đương.
  - Tải lên thư mục thuộc nhánh Class `ERRATIC`.
  - Filter xem video sequence để loại bỏ False Positive do noise (nhiễu điểm ảnh) từ camera Depth.

### 3. Triển khai Temporal State Stacking (Chuẩn bị Phase 2.0)

- **Mục tiêu:** Chuyển bộ State Observation từ góc nhìn "Điểm" (Snapshot) sang "Chuỗi thời gian" (Temporal) để phục vụ AI đoán ý định.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở file `src/navigation/context_builder.py`.
  - Bật config `temporal_stack_size = 3` (hoặc 5).
  - Sử dụng module `collections.deque` để lưu trữ lịch sử state, xoay đuôi liên tục.
  - Test kịch bản: Robot đứng yên nhưng người bước tới -> Mảng state history phải thể hiện khoảng cách `depth` đang nhỏ dần qua từng frame.

### 4. Xây dựng Intent Trajectory Buffer

- **Mục tiêu:** Theo dõi lịch sử di chuyển mượt mà tách biệt cho từng `track_id` (Từng người riêng rẽ). Tránh nhầm lẫn người A với người B.
- **Hướng dẫn thực hiện chi tiết:**
  - Tạo cấu trúc dữ liệu mới `intent_trajectory.py`.
  - Liên kết với ByteTrack ID từ `FrameDetections`. Maintain Memory Dictionary lưu vết toạ độ (x, y, distance) của đối tượng trong N frames.
