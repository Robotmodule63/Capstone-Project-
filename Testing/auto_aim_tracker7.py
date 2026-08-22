#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3, Twist
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import time

class AutoAimTracker(Node):
    def __init__(self):
        super().__init__('auto_aim_tracker')
        self.bridge = CvBridge()
        
        # --- Constants & Configuration ---
        self.FOCAL_LENGTH = 750.0
        
        # Physical Shooter Offset (Parallax Compensation)
        self.SHOOTER_OFFSET_X_CM = 1.0
        self.SHOOTER_OFFSET_Y_CM = 7.25
        
        # Fixed Angle Offsets
        self.PAN_OFFSET_DEG = 2.0
        self.TILT_OFFSET_DEG = -5.0
        
        # Servo Constraints & State
        self.PAN_CENTER = 80.0
        self.TILT_CENTER = 95.0
        self.ANGLE_THRESHOLD = 1.0
        
        # Target Distance Thresholds (1.0 meter = 100.0 cm)
        self.TARGET_STOP_DIST_CM = 100.0
        self.NEW_TARGET_ANGLE_THRESHOLD_DEG = 4.0
        
        # Smoothing & Stabilization Trackers
        self.smoothing_alpha = 0.4
        self.filtered_pan = self.PAN_CENTER
        self.filtered_tilt = self.TILT_CENTER
        self.last_sent_pan = self.PAN_CENTER
        self.last_sent_tilt = self.TILT_CENTER
        
        self.tracked_cx = None
        self.tracked_cy = None
        self.tracked_dist = 200.0
        
        # State Tracking for Shooter & Motion Logic
        self.has_shot_target = False
        self.shooting_active = False
        self.last_shot_pan = None
        self.last_shot_tilt = None
        self.shoot_timer = None
        
        # LiDAR Data Store
        self.latest_scan = None
        
        # --- ROS 2 Publishers & Subscribers ---
        self.pan_tilt_pub = self.create_publisher(Vector3, '/robot/pan_tilt_cmd', 10)
        self.target_detected_pub = self.create_publisher(Bool, '/robot/target_detected', 10)
        self.target_yaw_pub = self.create_publisher(Float32, '/robot/target_yaw', 10)
        self.target_pitch_pub = self.create_publisher(Float32, '/robot/target_pitch', 10)
        
        # Velocity and Shooter Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.shoot_cmd_pub = self.create_publisher(Bool, '/robot/shoot_cmd', 10)
        
        self.image_sub = self.create_subscription(
            Image, 'image_raw', self.image_callback, qos_profile=qos_profile_sensor_data
        )
        
        # LiDAR Subscription
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile_sensor_data
        )
        
        self.prev_time = time.time()
        self.get_logger().info("Auto-Aim Tracker Initialized with Dynamic Pursuit & Auto-Shooter.")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def is_circular(self, contour):
        """ Evaluates if a contour is roughly circular """
        peri = cv2.arcLength(contour, True)
        if peri < 30: return False
        area = cv2.contourArea(contour)
        if area == 0: return False
        circularity = 4 * np.pi * (area / (peri * peri))
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return 0.5 < circularity < 1.4 and 0.7 < aspect_ratio < 1.3

    def send_pan_tilt(self, pan, tilt):
        msg = Vector3()
        msg.x = float(pan)
        msg.y = float(tilt)
        msg.z = 0.0
        self.pan_tilt_pub.publish(msg)

    def send_velocity(self, linear_x, angular_z=0.0):
        """ Publishes movement velocity commands """
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def trigger_shooter(self):
        """ Activates shooter for 1 second and marks target as shot """
        if self.shooting_active:
            return
        
        self.get_logger().info("Target within range! Triggering shooter for 1 second.")
        self.shooting_active = True
        self.has_shot_target = True
        self.last_shot_pan = self.filtered_pan
        self.last_shot_tilt = self.filtered_tilt
        
        # Publish Shoot Command = True
        msg = Bool()
        msg.data = True
        self.shoot_cmd_pub.publish(msg)
        
        # Non-blocking ROS Timer to disable shooter after 1 second
        self.shoot_timer = self.create_timer(1.0, self.stop_shooter_callback)

    def stop_shooter_callback(self):
        """ Callback to turn off shooter after 1 second """
        msg = Bool()
        msg.data = False
        self.shoot_cmd_pub.publish(msg)
        
        self.get_logger().info("Shooter deactivated.")
        self.shooting_active = False
        
        if self.shoot_timer:
            self.shoot_timer.cancel()
            self.shoot_timer = None

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        frame_center_x, frame_center_y = width // 2, height // 2
        
        # --- Image Processing ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge((h, s, v_eq))
        enhanced_bgr = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 60, 160)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_circles = []
        for cnt in contours:
            if self.is_circular(cnt):
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    _, radius = cv2.minEnclosingCircle(cnt)
                    valid_circles.append((cx, cy, radius, cnt))
        
        grouped_circles = []
        for cx, cy, r, cnt in valid_circles:
            found_group = False
            for i, group in enumerate(grouped_circles):
                gcx, gcy = group[0], group[1]
                if np.sqrt((cx - gcx)**2 + (cy - gcy)**2) < 10:
                    grouped_circles[i][2].append(r)
                    grouped_circles[i][3].append(cx)
                    grouped_circles[i][4].append(cy)
                    found_group = True
                    break
            if not found_group:
                grouped_circles.append([cx, cy, [r], [cx], [cy]])
        
        # --- Tracking Logic ---
        target_info = {"x": 0, "y": 0, "dist": 0.0, "yaw": 0.0, "pitch": 0.0}
        tracking_active = False
        angle_x = 0.0
        angle_y = 0.0
        
        for gcx, gcy, radii, cxs, cys in grouped_circles:
            if len(radii) >= 3:
                raw_cx = float(np.mean(cxs))
                raw_cy = float(np.mean(cys))
                stable_r = int(np.max(radii))
                
                # 1. Apply sub-pixel stabilization to center coordinates
                if self.tracked_cx is None:
                    self.tracked_cx = raw_cx
                    self.tracked_cy = raw_cy
                else:
                    self.tracked_cx = (0.15 * raw_cx) + (0.85 * self.tracked_cx)
                    self.tracked_cy = (0.15 * raw_cy) + (0.85 * self.tracked_cy)
                
                cx = int(self.tracked_cx)
                cy = int(self.tracked_cy)
                
                # 2. Calculate base camera angles
                dx = cx - frame_center_x
                dy = cy - frame_center_y
                angle_x = np.degrees(np.arctan2(dx, self.FOCAL_LENGTH))
                angle_y = np.degrees(np.arctan2(dy, self.FOCAL_LENGTH))
                
                # 3. LiDAR Distance Calculation
                dist_cm = 200.0  # Fallback
                if self.latest_scan is not None and len(self.latest_scan.ranges) > 0:
                    index_val = angle_x / 0.5
                    if index_val >= 0:
                        idx = int(720 - index_val)
                    else:
                        idx = int(abs(index_val))
                    
                    idx = max(0, min(len(self.latest_scan.ranges) - 1, idx))
                    raw_dist = self.latest_scan.ranges[idx] * 100.0
                    
                    if not np.isinf(raw_dist) and raw_dist > 0:
                        self.tracked_dist = (0.2 * raw_dist) + (0.8 * self.tracked_dist)
                    
                dist_cm = self.tracked_dist

                # 4. Target Angles Calculation & Smoothing
                target_pan = self.PAN_CENTER - angle_x - self.PAN_OFFSET_DEG
                target_tilt = self.TILT_CENTER - angle_y - self.TILT_OFFSET_DEG

                self.filtered_pan = (self.smoothing_alpha * target_pan) + ((1 - self.smoothing_alpha) * self.filtered_pan)
                self.filtered_tilt = (self.smoothing_alpha * target_tilt) + ((1 - self.smoothing_alpha) * self.filtered_tilt) - 4.0
                
                # Clamp Servos
                self.filtered_pan = max(25.0, min(140.0, self.filtered_pan))
                self.filtered_tilt = max(80.0, min(120.0, self.filtered_tilt))
                
                # --- New Target Detection Check (> 4 deg variation) ---
                if self.has_shot_target and self.last_shot_pan is not None and self.last_shot_tilt is not None:
                    pan_diff = abs(self.filtered_pan - self.last_shot_pan)
                    tilt_diff = abs(self.filtered_tilt - self.last_shot_tilt)
                    
                    if pan_diff > self.NEW_TARGET_ANGLE_THRESHOLD_DEG or tilt_diff > self.NEW_TARGET_ANGLE_THRESHOLD_DEG:
                        self.get_logger().info("New target detected (> 4.0 deg shift). Resetting shot flag.")
                        self.has_shot_target = False

                # Servo Commands
                if (abs(self.filtered_pan - self.last_sent_pan) > self.ANGLE_THRESHOLD or
                    abs(self.filtered_tilt - self.last_sent_tilt) > self.ANGLE_THRESHOLD):
                    self.send_pan_tilt(self.filtered_pan, self.filtered_tilt)
                    self.last_sent_pan = self.filtered_pan
                    self.last_sent_tilt = self.filtered_tilt
                
                target_info = {"x": cx, "y": cy, "dist": dist_cm, "yaw": angle_x, "pitch": angle_y}
                tracking_active = True
                
                # --- Pursuit & Shooting Motion Control ---
                if dist_cm > self.TARGET_STOP_DIST_CM:
                    # Target is farther than 1.0 meter (100 cm) -> Move Forward
                    self.send_velocity(linear_x=0.2)
                else:
                    # Target reached (<= 1.0 meter) -> Stop Robot
                    self.send_velocity(linear_x=0.0)
                    
                    # Fire shooter if target hasn't been shot yet
                    if not self.has_shot_target and not self.shooting_active:
                        self.trigger_shooter()

                # Overlay visuals
                cv2.circle(frame, (cx, cy), stable_r, (0, 255, 0), 2)
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                break

        # --- No Target Detected Behavior ---
        if not tracking_active:
            self.tracked_cx = None
            self.tracked_cy = None
            
            # Stop moving when losing target tracking
            self.send_velocity(linear_x=0.0)
            
            # Reset internal filtered states
            self.filtered_pan = self.PAN_CENTER
            self.filtered_tilt = self.TILT_CENTER
            
            if (abs(self.PAN_CENTER - self.last_sent_pan) > self.ANGLE_THRESHOLD or
                abs(self.TILT_CENTER - self.last_sent_tilt) > self.ANGLE_THRESHOLD):
                self.send_pan_tilt(self.PAN_CENTER, self.TILT_CENTER)
                self.last_sent_pan = self.PAN_CENTER
                self.last_sent_tilt = self.TILT_CENTER

        # --- Publish Tracking States ---
        msg_det = Bool()
        msg_det.data = tracking_active
        self.target_detected_pub.publish(msg_det)
        
        if tracking_active:
            msg_yaw = Float32()
            msg_yaw.data = float(self.filtered_pan)
            self.target_yaw_pub.publish(msg_yaw)
            
            msg_pitch = Float32()
            msg_pitch.data = float(self.filtered_tilt)
            self.target_pitch_pub.publish(msg_pitch)

        self.draw_hud(frame, target_info, frame_center_x, frame_center_y, tracking_active)
        return frame

    def draw_hud(self, frame, info, cx, cy, tracking):
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        
        status_str = "SEARCHING"
        if tracking:
            if self.shooting_active:
                status_str = "FIRING SHOOTER"
            elif self.has_shot_target:
                status_str = "TARGET ALREADY SHOT"
            elif info['dist'] <= self.TARGET_STOP_DIST_CM:
                status_str = "IN RANGE (STOPPED)"
            else:
                status_str = "APPROACHING TARGET"

        text_lines = [
            f"FPS: {fps:.2f}",
            f"Yaw Offset: {info['yaw']:+.2f} deg",
            f"Pitch Offset: {info['pitch']:+.2f} deg",
            f"Sent Yaw: {self.last_sent_pan:.2f} deg",
            f"Sent Pitch: {self.last_sent_tilt:.2f} deg",
            f"LiDAR Dist: {info['dist']:.2f} cm",
            f"Status: {status_str}"
        ]
        
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (10, 30 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 140, 60), 2)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            processed_frame = self.process_frame(cv_image)
            cv2.imshow("Auto-Aim Tracker", processed_frame)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    tracker = AutoAimTracker()
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    finally:
        # Emergency stop on shutdown
        tracker.send_velocity(0.0, 0.0)
        tracker.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()