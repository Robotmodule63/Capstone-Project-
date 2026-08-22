#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float32
import threading
import time
import sys


class RobotCommanderClient(Node):
    def __init__(self):
        super().__init__('robot_commander_client')
        
        # Baseline state (position it was at during the LAST shot cycle)
        self.current_yaw = 80.0
        self.current_pitch = 95.0
        
        # Latest real-time state received from tracker
        self.latest_yaw = 80.0
        self.latest_pitch = 95.0
        
        self.target_detected = False
        self.is_shooting = False
        self.is_delaying = False

        # Timing & Tolerance
        self.ALIGNMENT_TOLERANCE = 0.5   # degrees
        self.SHOOT_DELAY_SEC = 1.5       # Delay before shooting (allow servos to align)
        self.SHOOT_DURATION_SEC = 1.0    # Duration of burst

        # Create Publishers
        self.pan_tilt_pub = self.create_publisher(Vector3, '/robot/pan_tilt_cmd', 10)
        self.shoot_pub = self.create_publisher(Bool, '/robot/shoot_cmd', 10)
        
        # Create Subscribers
        self.target_detected_sub = self.create_subscription(
            Bool, '/robot/target_detected', self.target_detected_callback, 10)
        
        self.target_yaw_sub = self.create_subscription(
            Float32, '/robot/target_yaw', self.target_yaw_callback, 10)
        
        self.target_pitch_sub = self.create_subscription(
            Float32, '/robot/target_pitch', self.target_pitch_callback, 10)

        self.get_logger().info(
            f"Robot Commander Client started with Auto-Shoot "
            f"(±{self.ALIGNMENT_TOLERANCE}° tolerance + {self.SHOOT_DELAY_SEC}s delay)"
        )

    def send_pan_tilt(self, pan, tilt):
        msg = Vector3()
        msg.x = float(pan)
        msg.y = float(tilt)
        msg.z = 0.0
        self.pan_tilt_pub.publish(msg)
        self.get_logger().info(f'Sent Pan: {pan:.2f}°, Tilt: {tilt:.2f}°')

    def send_shoot(self, is_shooting: bool):
        msg = Bool()
        msg.data = bool(is_shooting)
        self.shoot_pub.publish(msg)
        status = "ON" if is_shooting else "OFF"
        self.get_logger().info(f'Shoot Command: {status}')

    def target_detected_callback(self, msg: Bool):
        self.target_detected = msg.data

    def target_yaw_callback(self, msg: Float32):
        self.latest_yaw = msg.data
        self.check_and_shoot()

    def target_pitch_callback(self, msg: Float32):
        self.latest_pitch = msg.data
        self.check_and_shoot()

    def check_and_shoot(self):
        """Check alignment and trigger delay sequence if movement exceeds tolerance"""
        if not self.target_detected or self.is_shooting or self.is_delaying:
            return

        # Compare baseline against the live position
        yaw_diff = abs(self.current_yaw - self.latest_yaw)
        pitch_diff = abs(self.current_pitch - self.latest_pitch)

        if yaw_diff > self.ALIGNMENT_TOLERANCE or pitch_diff > self.ALIGNMENT_TOLERANCE:
            self.is_delaying = True
            self.get_logger().info(
                f"Target moved! Waiting {self.SHOOT_DELAY_SEC}s before shooting... "
                f"(Yaw diff: {yaw_diff:.2f}°, Pitch diff: {pitch_diff:.2f}°)"
            )
            
            # Start delay timer
            threading.Timer(
                self.SHOOT_DELAY_SEC, 
                self.start_shooting_after_delay
            ).start()

    def start_shooting_after_delay(self):
        """Called after delay - lock in and start shooting"""
        if not self.target_detected:
            self.is_delaying = False
            # Ensure baseline is updated so it doesn't instantly re-trigger 
            # if the target is found sitting in the exact same spot later.
            self.current_yaw = self.latest_yaw
            self.current_pitch = self.latest_pitch
            return

        self.is_delaying = False
        self.is_shooting = True
        
        self.get_logger().info("Delay complete -> Firing for 1 second!")
        self.send_shoot(True)
        
        # Stop shooting after duration
        threading.Timer(
            self.SHOOT_DURATION_SEC, 
            self.stop_shooting
        ).start()

    def stop_shooting(self):
        """Stop shooting and lock the new baseline position"""
        self.send_shoot(False)
        self.is_shooting = False
        
        # KEY FIX: Update the baseline AFTER shooting concludes.
        # This ensures the bot will only shoot again when it detects the NEXT movement.
        self.current_yaw = self.latest_yaw
        self.current_pitch = self.latest_pitch
        
        self.get_logger().info(
            f"Shoot complete. Reset baseline position -> "
            f"Yaw: {self.current_yaw:.2f}°, Pitch: {self.current_pitch:.2f}°"
        )


def user_input_loop(node):
    """Handles terminal input in a separate thread."""
    print("\n--- Robot Commander Terminal ---")
    print("Commands:")
    print(" 'p' to move Pan/Tilt manually")
    print(" 's' to toggle Shoot manually")
    print(" 'q' to Quit")
    
    while rclpy.ok():
        try:
            cmd = input("\nEnter command (p/s/q): ").strip().lower()
            
            if cmd == 'q':
                print("Shutting down commander...")
                rclpy.shutdown()
                break
                
            elif cmd == 'p':
                pan_val = input(" Enter Pan angle (25 to 120, center is 80): ")
                tilt_val = input(" Enter Tilt angle (80 to 120, center is 95): ")
                try:
                    node.send_pan_tilt(float(pan_val), float(tilt_val))
                except ValueError:
                    print(" [!] Please enter valid numbers for angles.")
                    
            elif cmd == 's':
                shoot_val = input(" Shoot? (1 for ON, 0 for OFF): ")
                if shoot_val in ['1', '0']:
                    node.send_shoot(shoot_val == '1')
                else:
                    print(" [!] Invalid input. Use 1 or 0.")
            else:
                print(" [!] Unknown command. Use 'p', 's', or 'q'.")
                
        except (EOFError, KeyboardInterrupt):
            rclpy.shutdown()
            break


def main(args=None):
    rclpy.init(args=args)
    node = RobotCommanderClient()
    
    input_thread = threading.Thread(target=user_input_loop, args=(node,), daemon=True)
    input_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
