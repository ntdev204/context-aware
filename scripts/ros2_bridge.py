#!/usr/bin/env python3
"""
Context-Aware ROS2 Bridge (Raspberry Pi)
---------------------------------------
Kết nối Jetson AI Server với ROS2 Nav2.
- Nhận lệnh lái từ Jetson (cổng 5555) -> Xuất ra /cmd_vel
- Đọc Odom/Pin (ROS2) -> Gửi về Jetson (cổng 5560) để gỡ khóa Watchdog.

Chạy trên Raspberry Pi:
  python3 ros2_bridge.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

import zmq
import struct
import threading
import time

JETSON_IP = "192.168.1.100"

class ContextAwareBridge(Node):
    def __init__(self):
        super().__init__('context_aware_bridge')
        
        # ROS2 Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Float32, '/PowerVoltage', self.battery_cb, 10)

        self.robot_state = {
            "vx": 0.0, "vy": 0.0, "vtheta": 0.0,
            "pos_x": 0.0, "pos_y": 0.0, "pos_theta": 0.0,
            "battery": 100.0
        }

        # ZMQ Setup
        self.ctx = zmq.Context()
        
        # 1. ZMQ SUB (Nhận lệnh từ Jetson - Cổng 5555)
        self.sub_sock = self.ctx.socket(zmq.SUB)
        self.sub_sock.setsockopt(zmq.CONFLATE, 1)
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "ai/nav_cmd")
        self.sub_sock.connect(f"tcp://{JETSON_IP}:5555")
        
        # 2. ZMQ PUB (Báo cáo State cho Jetson - Cổng 5560)
        self.pub_sock = self.ctx.socket(zmq.PUB)
        # Bắt buộc phải BIND ở đây, vì Jetson sẽ connect tới cổng này của Pi
        self.pub_sock.bind(f"tcp://0.0.0.0:5560")

        self.get_logger().info(f"ROS2 Bridge started! Listening to Jetson at {JETSON_IP}:5555")

        # Threads & Timers
        self.running = True
        self.recv_thread = threading.Thread(target=self.receive_commands, daemon=True)
        self.recv_thread.start()

        self.create_timer(0.05, self.publish_robot_state) # 20Hz báo cáo state

    def odom_cb(self, msg):
        self.robot_state["vx"] = msg.twist.twist.linear.x
        self.robot_state["vy"] = msg.twist.twist.linear.y
        self.robot_state["vtheta"] = msg.twist.twist.angular.z
        self.robot_state["pos_x"] = msg.pose.pose.position.x
        self.robot_state["pos_y"] = msg.pose.pose.position.y

    def battery_cb(self, msg):
        self.robot_state["battery"] = float(msg.data)

    def publish_robot_state(self):
        """Đóng gói State thành nhị phân (struct) để gửi cho Jetson."""
        # Format (36 bytes): 7 floats (vx, vy, vth, px, py, pth, batt) + 1 double (timestamp)
        payload = struct.pack(
            "!7fd",
            self.robot_state["vx"],
            self.robot_state["vy"],
            self.robot_state["vtheta"],
            self.robot_state["pos_x"],
            self.robot_state["pos_y"],
            self.robot_state["pos_theta"],
            self.robot_state["battery"],
            time.time()
        )
        self.pub_sock.send_multipart([b"robot/state", payload])

    def receive_commands(self):
        """Nhận lệnh NavCmd (struct) từ Jetson và đẩy vào ROS2."""
        poller = zmq.Poller()
        poller.register(self.sub_sock, zmq.POLLIN)
        
        while self.running and rclpy.ok():
            socks = dict(poller.poll(100))
            if self.sub_sock in socks:
                try:
                    parts = self.sub_sock.recv_multipart(flags=zmq.NOBLOCK)
                    if len(parts) >= 2:
                        data = parts[1]
                        # Giải mã cục binary nhận từ Jetson
                        # Format Jetson gửi (không dùng proto): "!iffiffB" (25 bytes)
                        if len(data) == 25:
                            mode, vel_scale, heading_off, follow_id, ts, conf, safety = struct.unpack("!iffiffB", data)
                            
                            # Tính toán Twist cơ bản (Chỉ là ví dụ logic, tuỳ biến nếu cần)
                            twist = Twist()
                            
                            # Nếu an toàn hoặc không bị STOP
                            if mode != 4: # 4 = STOP
                                # Giả định robot max speed = 0.5 m/s
                                twist.linear.x = float(vel_scale * 0.5)
                                twist.angular.z = float(heading_off * 1.5)
                                
                            self.cmd_pub.publish(twist)
                            
                except Exception as e:
                    self.get_logger().error(f"ZMQ Recv Error: {e}")

    def destroy_node(self):
        self.running = False
        self.sub_sock.close()
        self.pub_sock.close()
        self.ctx.term()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ContextAwareBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
