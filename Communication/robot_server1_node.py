#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
import time
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool

class RobotHardwareInterface(Node):
    def __init__(self):
        super().__init__('robot_hardware_interface')
        
        # --- Serial Setup ---
        self.port = '/dev/ttyACM0'  # UPDATED: This is the correct port for Arduino Micro
        self.baud = 115200
        self.ser = None
        self.connect_serial()

        # --- ROS 2 Subscribers ---
        self.pan_tilt_sub = self.create_subscription(
            Vector3, '/robot/pan_tilt_cmd', self.pan_tilt_callback, 10)
        self.shoot_sub = self.create_subscription(
            Bool, '/robot/shoot_cmd', self.shoot_callback, 10)

        self.get_logger().info("Robot Hardware Interface Node has started...")

    def connect_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2) 
            self.get_logger().info(f"Successfully connected to Arduino on {self.port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Could not connect to {self.port}. Error: {e}")
            self.ser = None

    def send_command(self, cmd_type, value):
        if self.ser is not None and self.ser.is_open:
            try:
                msg = f"{cmd_type}{int(value)}\n"
                self.ser.write(msg.encode('utf-8'))
                self.get_logger().debug(f"Sent: {msg.strip()}")
            except serial.SerialException as e:
                self.get_logger().error(f"Serial write error: {e}. Connection lost.")
                self.ser.close()
                self.ser = None 
        else:
            self.get_logger().warn("Serial not connected. Attempting to reconnect...")
            self.connect_serial()

    def pan_tilt_callback(self, msg):
        pan_angle = msg.x
        tilt_angle = msg.y
        self.send_command('p', pan_angle)
        self.send_command('t', tilt_angle)

    def shoot_callback(self, msg):
        value = 1 if msg.data else 0
        self.send_command('s', value)

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b"s0\n") 
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobotHardwareInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
