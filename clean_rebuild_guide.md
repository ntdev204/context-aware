# Hướng dẫn dọn dẹp toàn diện (Nuke & Rebuild)

> File này chứa các lệnh để xóa sạch toàn bộ Image, Container, rác của Docker và các dataset/cache cũ. Chạy các lệnh này khi bạn muốn build lại toàn bộ hệ thống từ con số 0.

---

### Bước 1: Dừng và xóa toàn bộ Container
Lệnh này sẽ tắt mọi dịch vụ đang chạy và tháo gỡ các phân vùng ảo (volumes).

```bash
docker compose -f docker/docker-compose.yml --profile jetson-prod --profile jetson-dev --profile server down --remove-orphans --volumes
```

### Bước 2: Xóa tận gốc các Image cũ và Cache mạng của Docker
Lệnh này sẽ gỡ bỏ hoàn toàn bộ nhớ lưu trữ code cũ, bắt Docker phải tải và build lại mọi thứ ở lần chạy tiếp theo.

```bash
docker rmi context-aware:jetson-prod context-aware:jetson-dev context-aware:server -f 2>/dev/null
docker system prune -a --volumes -f
```

### Bước 3: Dọn dẹp dữ liệu rác trên Jetson
Lệnh này xóa sạch các tệp ảnh Dataset cũ và file `.engine` bị lỗi để tránh gây sập (Segmentation Fault) cho PyTorch.

```bash
sudo rm -rf data/roi_dataset/*
sudo rm -rf models/yolo/*.engine
sudo rm -rf models/torch_cache/*
```

### Bước 4: Build lại hoàn toàn hệ thống
Sử dụng cờ `--no-cache` để đảm bảo không dùng lại bất kỳ mảnh mã nguồn cũ nào. Việc này có thể mất một chút thời gian (khoảng 5-10 phút).

```bash
sudo docker compose -f docker/docker-compose.yml build --no-cache jetson-dev
sudo docker compose -f docker/docker-compose.yml build jetson-prod
```

---

### Bước cuối cùng: Khởi động lại
Sau khi Bước 4 chạy xong 100%, bạn có thể bật lại hệ thống bằng lệnh quen thuộc:

```bash
sudo make jetson-up
sudo make jetson-logs
```
