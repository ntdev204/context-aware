# Hướng Dẫn Cài Đặt Jetson Orin Nano Từ Đầu (Phase 1 & 1.5)

Tài liệu này dành cho việc setup lại **hoàn toàn từ đầu** khi mới cài lại hệ điều hành cho Jetson Orin Nano Super, để hệ thống chạy mượt mà ngay Phase 1 (Nền tảng Inference) và Phase 1.5 (Gửi ảnh tự động sang Laptop học).

---

## 1. Cài Đặt OS & Hệ Thống Ban Đầu

### 1.1 Firmware & Power Mode
1. Flash **JetPack 6.x (L4T R36.x)** thông qua NVIDIA SDK Manager hoặc balenaEtcher.
2. Thiết lập chế độ điện năng tối đa (25W) và bật quạt:
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 1.2 Tắt GUI (Khuyến Nghị - Tiết kiệm ~800MB RAM)
```bash
sudo systemctl set-default multi-user.target
sudo reboot
# Để bật lại GUI nếu cần: sudo systemctl set-default graphical.target
```

### 1.3 Tạo Swap File 16GB (SIÊU QUAN TRỌNG)
Bộ nhớ Orin Nano 8GB cực kỳ dễ bị tràn khi build TensorRT hoặc chạy Docker nặng.
```bash
sudo fallocate -l 16G /mnt/16GB.swap
sudo chmod 600 /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap

# Để không bị mất sau khi khởi động lại:
echo '/mnt/16GB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

---

## 2. Phần Mềm Nền Tảng

### 2.1 Cài Docker & NVIDIA Toolkit
JetPack 6 thường có sẵn Docker, tuy nhiên hãy kiểm tra:
```bash
docker --version
# Nếu chưa có: curl -fsSL https://get.docker.com | sh

# Cho phép user chạy docker không cần sudo
sudo usermod -aG docker $USER
newgrp docker

# Đảm bảo nvidia-container-toolkit đã cài
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2.2 Mạng & Tailscale
Do Laptop (Windows) và Jetson cần gửi file từ xa qua lại một cách ổn định, dùng Tailscale:
```bash
# Cài Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# (Mở link được cung cấp để đăng nhập cùng acc với Laptop)
```

---

## 3. Pull Nguồn Code & Chuẩn Bị Codebase

1. Tải codebase về Jetson:
```bash
git clone <URL-của-repo-github> ~/context-aware
cd ~/context-aware
```

2. Tạo các thư mục lưu trữ models và dataset:
```bash
mkdir -p models/yolo
mkdir -p models/cnn_intent
mkdir -p logs/
```

3. Tải model YOLOv11 sơ khai:
```bash
make jetson-download-models
```

---

## 4. Thực Thi Phase 1 (Nền Tảng AI / Inference)

Quá trình này sẽ sử dụng ảnh Docker `jetson-dev` để build ra phiên bản TensorRT chạy cực nhanh trên Jetson.

```bash
# Bước 4.1: Build hình ảnh docker cho môi trường Dev
make jetson-build-dev

# Bước 4.2: Export YOLOv11 từ PyTorch (.pt) sang TensorRT (.engine)
# Tùy biến mất khoảng 5-10 phút. Đảm bảo Swap đã chạy bằng lệnh 'free -h'!
make jetson-export
```
*Kết quả:* Bạn sẽ nhận được file `models/yolo/yolo11s.engine` đã được tối ưu.

---

## 5. Thực Thi Phase 1.5 (Đường Ống Đẩy Dữ Liệu Lên Laptop)

Ở Phase 1.5, Jetson tự động thu thập ảnh Tracking người (có chứa Depth) để nén (`tar.gz`) và đẩy qua Laptop train. Cần kết nối SSH không cần mật khẩu.

### Bước 5.1: Bật Chảo Hứng Dữ Liệu (Laptop Windows)
1. Cài đặt Tailscale trên Windows và nhớ IP Tailscale (VD: `100.93.83.87`).
2. Mở Git Bash / CMD tại thư mục codebase, tiến hành bật server hứng file:
```bash
make laptop-build
make laptop-up
```
> **Lưu ý:** `make laptop-up` sẽ chạy môi trường Docker. Docker đã cài đặt sẵn hệ thống kết nối SSH ngầm ở Cổng `2222` và gắn thẳng ổ ảo `roi_incoming` cho Jetson ném file vào, bạn không cần cài thêm bất cứ tính năng phụ nào trên Windows!

### Bước 5.2: Setup Chìa Khóa SSH Từ Máy Jetson (1 lần duy nhất)
Chúng ta sẽ đẩy file mà **không cần gõ mật khẩu** qua SSH qua container Docker. Mật khẩu ảo mặc định của container là `robotics`.
Mở Terminal trên **Jetson**:
```bash
# 1. Tạo chìa khóa (Bấm Enter liên tục, KHÔNG nhập pass)
ssh-keygen -t ed25519 -f ~/.ssh/id_jetson -N ""

# 2. Đẩy chìa khóa sang Laptop Docker qua Cổng 2222 (Mật khẩu là: robotics)
ssh-copy-id -i ~/.ssh/id_jetson.pub -p 2222 root@<IP_LAPTOP_TAILSCALE>

# 3. Chạy thử xem kết nối thành công chưa
ssh -p 2222 root@<IP_LAPTOP_TAILSCALE> "echo OK Connected to Laptop Docker Workspace"
```

### Bước 5.3: Cấu Hình File `production.yaml` Trên Jetson
Mở file `config/production.yaml` bằng lệnh `nano config/production.yaml`:

```yaml
experience:
  # PHẢI LÀ FALSE ĐỂ CHẠY CHUYÊN ROI MODE (PHASE 1.5)
  hdf5_enabled: false 
  
  # CẬP NHẬT LẠI TARGET ĐÚNG CÚ PHÁP DO ĐỔI PORT 2222 VÀ USER ROOT
  roi_laptop_target: "-e 'ssh -p 2222' root@100.93.83.87:/data/roi_incoming/"
```

---

## 6. LÊN SÂN KHẤU (End-to-End Run)

Lúc này, toàn bộ hệ thống đã thông luồng một cách hoàn hảo:

1. **Laptop Windows**:
   Bạn chạy lệnh `make laptop-logs` để giám sát xem có batch nào về hay đang đào tạo AI gì không.
   
2. **Jetson**: 
   Khởi động hệ thống AI qua Camera Astra S:
   ```bash
   make jetson-build-prod    # (Chỉ làm 1 lần)
   make jetson-up            # Bật hệ thống AI chạy ngầm
   make jetson-logs          # Xem logs xe chạy
   ```

Tất cả ảnh crop sẽ chuyển thẳng 100% vào bộ nhớ Docker `intent-dataset` trên Laptop mà Windows Explorer không thể dòm ngó, bảo đảm sạch sẽ hoàn toàn khỏi ổ chứa Code. Cứ 5000 frame gửi về thì Laptop giật card màn hình tự động Fine-Tune một phát. System hoàn toàn khép kín tự trị!
