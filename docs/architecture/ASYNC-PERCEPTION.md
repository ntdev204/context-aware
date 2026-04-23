# Kiến trúc Xử lý Bất đồng bộ (Asynchronous Perception)

Tài liệu này phân tích cơ chế xử lý luồng Video (từ Camera) và thuật toán Trí tuệ Nhân tạo (YOLO/CNN) bên trong module `src/perception/camera.py`. Đây là giải pháp cốt lõi để giải quyết bài toán độ trễ thời gian thực (Real-time Latency) trong xe tự hành.

---

## 1. Vấn đề của mô hình truyền thống (Queue-based)

Trong các hệ thống thị giác máy tính tiêu chuẩn, luồng dữ liệu thường được xử lý theo dạng xếp hàng (FIFO Queue):
- Camera thu nhận hình ảnh và đẩy vào một hàng đợi (Queue).
- GPU (AI) rút từng ảnh ra khỏi hàng đợi để suy luận (Inference).

**Hậu quả trong Robotics:**
Tốc độ của phần cứng Camera (thường là 30 FPS) thường nhanh hơn tốc độ suy luận của AI (ví dụ: Jetson xử lý YOLO + CNN mất ~60ms/frame, tương đương 15-20 FPS). Nếu dùng Queue, ảnh sẽ bị dồn ứ lại. Đến giây thứ 5, AI có thể đang xử lý hình ảnh của giây thứ 2. 
=> **Độ trễ tích lũy (Lag):** Robot phản ứng quá chậm, dẫn đến đâm va vào chướng ngại vật dù AI đã nhận diện đúng, đơn giản vì thời điểm nhận diện đã là quá khứ.

---

## 2. Giải pháp Double-Buffering & Multi-threading

Để triệt tiêu hoàn toàn độ trễ tích lũy (Zero-latency), dự án áp dụng mô hình Đệm kép (Double-Buffering) kết hợp Đa luồng (Multi-threading).

### A. Phân tách Luồng (Producer - Consumer)
Hệ thống tạo ra 2 tiến trình chạy song song và độc lập hoàn toàn với nhau:

1. **Luồng Phụ (Producer - Camera Thread):** 
   Luôn chạy ngầm (Daemon Thread) ở tốc độ tối đa của cảm biến (30 FPS). Nó liên tục kết nối với API của camera (OpenNI2 cho Astra S hoặc V4L2 cho USB), lấy ảnh RGB và Depth, sau đó cất vào bộ nhớ đệm (Back-buffer).
   
2. **Luồng Chính (Consumer - AI Thread):** 
   Chạy vòng lặp `while True` trong file `main.py`. Nó lấy ảnh từ bộ đệm ra, chạy toàn bộ Pipeline: YOLO -> ByteTrack -> Intent CNN -> Tính toán Nav2 -> Xuất lệnh ZMQ. 

### B. Nguyên lý "Ghi Đè" (Drop Frame)
Thay vì xếp hàng đợi chờ AI xử lý, Luồng Camera áp dụng cơ chế **GHI ĐÈ (Overwrite)** liên tục lên biến bộ đệm (`self._frame`). 

* Khi AI hoàn thành 1 chu trình xử lý (tốn khoảng 60ms) và quay lại gọi hàm `grab()`, nó sẽ bốc được **khung hình mới nhất** vừa mới được Camera chép vào vài mili-giây trước. 
* Những khung hình sinh ra trong lúc AI đang bận tính toán sẽ bị **Drop (vứt bỏ)** hoàn toàn. 

**Đánh đổi (Trade-off):** Hệ thống chấp nhận mất/bỏ sót khung hình trung gian để đổi lấy đặc quyền tuyệt đối: **Không bao giờ bị trễ hình**. AI luôn nhìn thấy "thực tại" ngay thời điểm nó hỏi xin ảnh.

---

## 3. Cơ chế Khóa an toàn (Thread Lock)

Khi chạy đa luồng, có một rủi ro cực lớn gọi là **Xé hình (Tearing)** hoặc Lỗi phân vùng bộ nhớ (Segmentation Fault): Xảy ra khi Luồng Camera đang chép đè dữ liệu ảnh vào biến đúng ngay lúc Luồng AI cũng thò tay vào đọc biến đó.

Để ngăn chặn, class `Camera` sử dụng khóa Mutex `threading.Lock()`:

```python
with self._lock:
    self._frame = rgb
    self._depth_frame = depth
```

- Khi Camera đang chép dữ liệu mới, nó "Khóa" biến này lại.
- Nếu hàm `grab()` của AI định lấy ảnh ngay lúc đó, luồng AI bị ép buộc chuyển sang trạng thái "Sleep" chờ vài micro-giây cho đến khi Camera chép xong và mở khóa.
- Quá trình chép mảng numpy (con trỏ) diễn ra cực kỳ nhanh (vài micro-giây), do đó việc Locking này không gây cản trở tốc độ của toàn bộ hệ thống.

---

## 4. Cô lập Lỗi (Graceful Degradation)

Module Camera có khả năng chịu lỗi (Fault-tolerant) tốt nhờ việc xử lý độc lập các kênh dữ liệu:

Luồng đọc Depth (đo độ sâu) và RGB (ảnh màu) được bọc trong các khối `try/except` riêng biệt. 
- Nếu cảm biến tia hồng ngoại (IR) của Depth Camera bị lóa sáng, báo lỗi khung hình, nó sẽ chỉ gán `depth = None`. 
- Ảnh màu `rgb` vẫn được chụp, ghi vào bộ đệm và gửi cho YOLO.
- Thuật toán Điều hướng (`HeuristicPolicy`) khi nhận thấy `depth_cov = 0%` (không có depth) sẽ tự động chuyển sang cơ chế dự phòng: Dùng chiều cao Bounding Box của YOLO để nội suy ra khoảng cách.

Nhờ cơ chế cô lập này, sự cố phần cứng của một cảm biến phụ trợ không làm sụp đổ (Crash) toàn bộ luồng chạy của Trí tuệ Nhân tạo.
