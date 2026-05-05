# Hướng dẫn Thiết lập và Cấu hình Twist Mux (Dành cho AI Agent)

**Mục tiêu (Goal):** 
Xử lý xung đột luồng điều khiển `/cmd_vel` trên Raspberry Pi. Hiện tại, Jetson AI và Local Planner (Nav2) đang cùng đè lệnh lên `/cmd_vel`, dẫn đến hiện tượng robot bị giật cục. Chúng ta sẽ sử dụng node `twist_mux` của ROS2 để phân quyền ưu tiên: Điều khiển tay (Teleop) > AI (Jetson) > Tự động (Nav2).

**Ngữ cảnh hệ thống (Context):**
- Thao tác này được thực hiện trực tiếp trên phần cứng/workspace của **Raspberry Pi** (không phải Jetson).
- Pi chạy ROS2 Humble.

> **Gửi AI Agent đang thực thi file này:** Hãy tuân thủ tuyệt đối quy tắc `Atomic Step-by-Step` (sửa từng file, báo cáo cho người dùng, sau đó mới đi tiếp). Đừng gom tất cả lại làm một lượt.

---

## Bước 1: Cài đặt thư viện Twist Mux
Yêu cầu hệ thống phải có gói `twist_mux` của ROS2.
- **Thực thi:** Chạy lệnh sau trên terminal của Pi:
  ```bash
  sudo apt update
  sudo apt install -y ros-humble-twist-mux
  ```

## Bước 2: Tạo file cấu hình `twist_mux_locks.yaml` và `twist_mux_topics.yaml`
Bạn cần tạo (hoặc cập nhật) file cấu hình cho `twist_mux`. Giả sử cấu hình nằm trong package chứa file launch của bạn (ví dụ: `wheeltec_ros2_bridge/config/twist_mux.yaml`).

Tạo file `twist_mux.yaml` với nội dung ưu tiên như sau:
```yaml
twist_mux:
  ros__parameters:
    topics:
      # 1. Ưu tiên cao nhất: Điều khiển bằng tay (Tay cầm PS4, bàn phím)
      joystick:
        topic: cmd_vel_teleop
        timeout: 0.5
        priority: 100

      # 2. Ưu tiên thứ hai: Jetson AI (Bám người, ra quyết định trực tiếp)
      ai_tracker:
        topic: cmd_vel_ai
        timeout: 0.5
        priority: 50

      # 3. Ưu tiên thấp nhất: Nav2 (Local Planner - né vật cản, đi map)
      navigation:
        topic: cmd_vel_nav
        timeout: 0.5
        priority: 10
```

## Bước 3: Đổi đầu ra của Bridge Node (AI Server)
Tìm đến file Python xử lý nhận ZMQ từ Jetson (chẳng hạn `bridge_node.py` trong `wheeltec_ros2_bridge`).
- **Thực thi:** Tìm dòng khai báo publisher:
  ```python
  self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
  ```
- **Sửa thành:**
  ```python
  self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_ai', 10)
  ```
Lưu ý không sửa logic nhận ZMQ.

## Bước 4: Remap đầu ra của Nav2
Hiện tại Nav2 (Local Planner) đang xuất lệnh trực tiếp ra `/cmd_vel`. 
- **Thực thi:** Mở file cấu hình hoặc file Launch của Nav2 (thường nằm ở `wheeltec_nav2` hoặc cấu hình Controller Server), tìm phần remappings của `controller_server` và thêm:
  ```python
  remappings=[
      ('/cmd_vel', '/cmd_vel_nav')
  ]
  ```
*(Ghi chú cho AI: Nếu người dùng launch bằng file YAML config, hãy sửa tham số `cmd_vel_topic: "cmd_vel_nav"` trong yaml của `controller_server`).*

## Bước 5: Cập nhật Launch File để chạy chung Twist Mux
Mở file `launch` đang chịu trách nhiệm bật `bridge_node` (ví dụ `context_aware_bridge.launch.py`).
- **Thực thi:** Thêm node `twist_mux` vào danh sách các node được launch:
  ```python
  import os
  from ament_index_python.packages import get_package_share_directory
  from launch import LaunchDescription
  from launch_ros.actions import Node

  def generate_launch_description():
      config_dir = os.path.join(get_package_share_directory('wheeltec_ros2_bridge'), 'config')
      
      twist_mux_node = Node(
          package='twist_mux',
          executable='twist_mux',
          name='twist_mux',
          parameters=[os.path.join(config_dir, 'twist_mux.yaml')],
          remappings=[('/cmd_vel_out', '/cmd_vel')] # Xuất lệnh cuối cùng đã phân giải ra /cmd_vel thực của robot
      )

      bridge_node = Node(
          package='wheeltec_ros2_bridge',
          executable='bridge_node',
          name='context_aware_bridge',
          output='screen',
          parameters=[{'jetson_ip': '25.12.4.100'}]
      )

      return LaunchDescription([
          twist_mux_node,
          bridge_node
      ])
  ```

## Bước 6: Build lại và Kiểm tra
Yêu cầu người dùng (hoặc AI thực thi tự động):
1. `cd ~/wheeltec_workspace` (hoặc workspace của Pi).
2. `colcon build --packages-select wheeltec_ros2_bridge`
3. `source install/setup.bash`
4. Chạy Launch file.
5. **Cách verify (Kiểm tra lại):** 
   - Mở 1 terminal mới, gõ: `ros2 topic list`
   - Phải nhìn thấy: `/cmd_vel`, `/cmd_vel_ai`, `/cmd_vel_nav`.
   - Gõ `ros2 topic info /cmd_vel` để xác nhận `twist_mux` là node duy nhất đang Publish vào đó, và Base Controller của robot là node Subscribe.

---
**✅ Tiêu chuẩn Hoàn thành:** Khi robot vừa bật Nav2 (đang chạy tới), nếu Jetson phát hiện thấy mục tiêu ở gần và xuất lệnh `/cmd_vel_ai` yêu cầu lùi lại, robot ngay lập tức phớt lờ lệnh tới của Nav2 và Lùi lại do `cmd_vel_ai` có độ ưu tiên cao hơn (50 > 10). Mất tín hiệu AI 0.5s, robot tự trả lại quyền cho Nav2.
