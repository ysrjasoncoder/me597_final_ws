#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self._initialized = False
        self.integral_limit = integral_limit  # 可选积分限幅

    def reset_integral(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self._initialized = False

    def update(self, error, dt):
        if not self._initialized:
            self.prev_error = error
            self._initialized = True
        if dt <= 0.0:
            dt = 1e-3

        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))

        derivative = (error - self.prev_error) / dt
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return out


class BallSearcher(Node):
    def __init__(self):
        super().__init__('ball_searcher')

        # ROS2 interfaces
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Ball detection and tracking
        self.detected_balls = {}  # {color: {'x': x, 'y': y, 'samples': count}}
        self.localized_colors = set()
        self.current_ball_color = None
        self.ball_center = None
        self.ball_radius = 0
        self.image_width = 0
        self.ball_distance = None

        # Color definitions (HSV)
        self.color_ranges = {
            'red': [
                (np.array([0, 150, 100]), np.array([10, 255, 255])),
                (np.array([170, 150, 100]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 150, 100]), np.array([130, 255, 255]))
            ],
            'green': [
                (np.array([40, 150, 100]), np.array([80, 255, 255]))
            ]
        }

        # LIDAR data
        self.last_scan = None

        # Wall following parameters
        self.forward_speed = 0.18
        self.angular_speed = 1.0
        self.d_des_right = 0.6
        self.d_front_stop = 0.5
        self.d_front_slow = 0.8
        self.look_deg = 10.0
        self.right_span = 70.0

        # Wall following gains
        self.kp_wall = 1.0
        self.kp_front = 8.0
        self.kp_rf = 0.6

        # Ball approach parameters
        self.ball_detection_distance = 2.5
        self.min_radius_for_approach = 25

        # ✅ 改成“目标距离控制”
        self.target_ball_distance = 0.60   # 希望停在球前 0.6m（按你场地调）
        self.distance_deadzone = 0.05      # 距离误差 < 5cm 就认为到位

        # 角度控制仍用图像中心
        self.angular_deadzone = 0.05       # 归一化中心误差死区
        self.max_linear_approach = 0.15
        self.max_angular_approach = 0.5

        # PID controllers
        # 角度：误差是归一化像素偏差
        self.ang_pid = PID(kp=1.2, ki=0.0, kd=0.05, integral_limit=0.5)
        # 线速度：误差是“距离误差（米）”
        self.lin_pid = PID(kp=0.8, ki=0.0, kd=0.05, integral_limit=0.5)

        # State machine
        self.state = 'WALL_FOLLOW'
        self.visible_count = 0
        self.invisible_count = 0
        self.visible_enter_frames = 4
        self.visible_exit_frames = 8

        self.localization_samples = []
        self.required_samples = 8
        self.localization_timeout = 0
        self.max_localization_time = 100  # frames (~5 seconds at 20Hz)

        self.last_time = None

        self.get_logger().info('Ball Searcher initialized (no reverse in approach)!')

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        self.last_scan = msg

    def sector_min(self, scan, center_deg, half_width_deg):
        if scan is None:
            return float("inf")
        angle_min = math.degrees(scan.angle_min)
        angle_inc = math.degrees(scan.angle_increment)
        n = len(scan.ranges)

        start_deg = center_deg - half_width_deg
        end_deg = center_deg + half_width_deg

        start_idx = max(0, int((start_deg - angle_min) / angle_inc))
        end_idx = min(n - 1, int((end_deg - angle_min) / angle_inc))

        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        vals = [
            scan.ranges[i]
            for i in range(start_idx, end_idx + 1)
            if not math.isinf(scan.ranges[i]) and not math.isnan(scan.ranges[i])
        ]
        return min(vals) if vals else float("inf")

    def sector_mean(self, center_deg, half_width_deg):
        scan = self.last_scan
        if scan is None:
            return float("inf")
        angle_min = math.degrees(scan.angle_min)
        angle_inc = math.degrees(scan.angle_increment)
        n = len(scan.ranges)

        start_deg = center_deg - half_width_deg
        end_deg = center_deg + half_width_deg

        start_idx = max(0, int((start_deg - angle_min) / angle_inc))
        end_idx = min(n - 1, int((end_deg - angle_min) / angle_inc))

        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        vals = [
            scan.ranges[i]
            for i in range(start_idx, end_idx + 1)
            if not math.isinf(scan.ranges[i]) and not math.isnan(scan.ranges[i])
        ]
        return np.mean(vals) if vals else float("inf")

    def detect_ball_color(self, hsv_image):
        best_color = None
        best_contour = None
        best_area = 0

        for color, ranges in self.color_ranges.items():
            if color in self.localized_colors:
                continue

            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > best_area and area > 500:
                    best_area = area
                    best_contour = largest
                    best_color = color

        return best_color, best_contour

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_width = cv_image.shape[1]

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            color, contour = self.detect_ball_color(hsv)
            seen = False

            if contour is not None:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > self.min_radius_for_approach:
                    self.ball_center = (int(x), int(y))
                    self.ball_radius = int(radius)
                    self.current_ball_color = color
                    seen = True

                    # distance estimate (0.15m ball diameter, ~60deg FOV)
                    focal_length = self.image_width / (2 * math.tan(math.radians(30)))
                    self.ball_distance = (0.15 * focal_length) / (2 * radius)

            if seen:
                self.visible_count += 1
                self.invisible_count = 0
            else:
                self.invisible_count += 1
                self.visible_count = 0

            # Debug visualization
            debug = cv_image.copy()
            if seen:
                color_bgr = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0)}
                cv2.circle(debug, self.ball_center, self.ball_radius,
                           color_bgr.get(color, (255, 255, 255)), 2)
                if self.ball_distance is not None:
                    cv2.putText(
                        debug, f'{color} ({self.ball_distance:.2f}m)',
                        (self.ball_center[0]-60, self.ball_center[1]-self.ball_radius-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr.get(color, (255, 255, 255)), 2
                    )

            cv2.putText(debug, f'State: {self.state}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset = 60
            for c in self.localized_colors:
                info = self.detected_balls.get(c, {})
                cv2.putText(debug, f'{c}: ({info.get("x", 0):.2f}, {info.get("y", 0):.2f}) DONE',
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 25

            cv2.putText(debug, f'Found: {len(self.localized_colors)}/3',
                        (10, cv_image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Ball Searcher", debug)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

    def localize_ball(self):
        if self.ball_distance is None or self.ball_center is None or self.image_width == 0:
            return None

        cx = self.image_width / 2.0
        pixel_offset = self.ball_center[0] - cx
        angle_per_pixel = math.radians(60) / self.image_width
        angle_to_ball = pixel_offset * angle_per_pixel

        ball_angle_world = self.robot_yaw + angle_to_ball
        ball_x = self.robot_x + self.ball_distance * math.cos(ball_angle_world)
        ball_y = self.robot_y + self.ball_distance * math.sin(ball_angle_world)

        return (ball_x, ball_y)

    def wall_follow_control(self):
        if self.last_scan is None:
            return Twist()

        scan = self.last_scan
        front_min = self.sector_min(scan, center_deg=-15.0, half_width_deg=self.look_deg)
        front_mean = self.sector_mean(center_deg=-15.0, half_width_deg=self.look_deg)
        right_front_min = self.sector_min(scan, center_deg=-45.0, half_width_deg=self.look_deg)
        right_min = self.sector_min(scan, center_deg=-90.0, half_width_deg=self.right_span / 2.0)

        twist = Twist()

        if math.isfinite(front_min):
            if front_min < self.d_front_stop:
                speed_scale = 0.0
            elif front_min < self.d_front_slow:
                speed_scale = (front_min - self.d_front_stop) / (self.d_front_slow - self.d_front_stop)
            else:
                speed_scale = 1.0
        else:
            speed_scale = 1.0

        twist.linear.x = self.forward_speed * max(0.0, min(1.0, speed_scale))

        ang = 0.0
        if math.isfinite(front_min) and front_min < self.d_front_stop and \
           math.isfinite(front_mean) and 1.5 * front_min > self.d_front_stop:
            ang = 0.5
        else:
            if math.isfinite(right_min):
                error_wall = right_min - self.d_des_right
                ang += -self.kp_wall * error_wall

            if math.isfinite(right_front_min) and right_front_min < self.d_des_right:
                error_rf = self.d_des_right - right_front_min
                ang += self.kp_rf * self.kp_wall * error_rf

        twist.angular.z = float(np.clip(ang, -self.angular_speed, self.angular_speed))
        return twist

    def approach_ball_control(self, dt):
        """✅ 接近球：角度用图像中心，线速度用距离误差；并且禁止倒车"""
        if self.image_width == 0 or self.ball_center is None or self.ball_distance is None:
            return None

        # dt 防止偶尔跳大
        dt = float(np.clip(dt, 1e-3, 0.10))

        twist = Twist()

        # -------- angular control (center in image) --------
        cx = self.image_width / 2.0
        ball_x = float(self.ball_center[0])
        ang_err_norm = (cx - ball_x) / cx  # left positive => turn left

        if abs(ang_err_norm) <= self.angular_deadzone:
            self.ang_pid.reset_integral()
            ang_err_norm = 0.0

        ang_cmd = self.ang_pid.update(ang_err_norm, dt)
        twist.angular.z = float(np.clip(ang_cmd, -self.max_angular_approach, self.max_angular_approach))

        # -------- linear control (distance to target) --------
        dist_err = self.ball_distance - self.target_ball_distance  # >0 远了要前进；<0 近了
        if abs(dist_err) <= self.distance_deadzone:
            # 到位：停下（不倒车）
            self.lin_pid.reset_integral()
            twist.linear.x = 0.0
        else:
            lin_cmd = self.lin_pid.update(dist_err, dt)

            # ✅ 关键：不允许负速度（不倒车）
            twist.linear.x = float(np.clip(lin_cmd, 0.0, self.max_linear_approach))

        return twist

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now == 0.0:
            return

        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        if len(self.localized_colors) >= 3:
            self.get_logger().info('All 3 balls localized! Mission complete.', throttle_duration_sec=2.0)
            self.cmd_vel_pub.publish(Twist())
            return

        twist = Twist()

        if self.state == 'WALL_FOLLOW':
            if (self.visible_count >= self.visible_enter_frames and
                self.current_ball_color is not None and
                self.ball_distance is not None and
                self.ball_distance < self.ball_detection_distance):

                self.state = 'APPROACH_BALL'
                self.ang_pid.reset()
                self.lin_pid.reset()
                self.get_logger().info(f'Approaching {self.current_ball_color} ball at {self.ball_distance:.2f}m')
            else:
                twist = self.wall_follow_control()

        elif self.state == 'APPROACH_BALL':
            if self.invisible_count >= self.visible_exit_frames:
                self.get_logger().info('Ball lost, returning to wall follow')
                self.state = 'WALL_FOLLOW'
                self.localization_samples = []
                self.localization_timeout = 0
                self.ang_pid.reset()
                self.lin_pid.reset()

            else:
                # 到达目标距离并且大致居中 -> LOCALIZE
                centered = False
                if self.image_width > 0 and self.ball_center is not None:
                    centered = (abs(self.ball_center[0] - self.image_width/2.0) / (self.image_width/2.0) < 0.08)

                close_enough = (self.ball_distance is not None and
                                abs(self.ball_distance - self.target_ball_distance) < 0.08)

                if centered and close_enough:
                    self.state = 'LOCALIZE_BALL'
                    self.localization_timeout = 0
                    self.localization_samples = []
                    self.get_logger().info(f'Starting localization of {self.current_ball_color} ball')
                    twist = Twist()
                else:
                    twist = self.approach_ball_control(dt)
                    if twist is None:
                        twist = Twist()

        elif self.state == 'LOCALIZE_BALL':
            self.localization_timeout += 1

            if self.localization_timeout > self.max_localization_time:
                self.get_logger().warn('Localization timeout, returning to wall follow')
                self.state = 'WALL_FOLLOW'
                self.localization_samples = []
                self.localization_timeout = 0

            elif self.invisible_count >= self.visible_exit_frames:
                self.get_logger().info('Ball lost during localization, returning to wall follow')
                self.state = 'WALL_FOLLOW'
                self.localization_samples = []
                self.localization_timeout = 0

            else:
                ball_pos = self.localize_ball()
                if ball_pos is not None:
                    self.localization_samples.append(ball_pos)

                    if len(self.localization_samples) >= self.required_samples:
                        avg_x = np.mean([p[0] for p in self.localization_samples])
                        avg_y = np.mean([p[1] for p in self.localization_samples])
                        std_x = np.std([p[0] for p in self.localization_samples])
                        std_y = np.std([p[1] for p in self.localization_samples])

                        self.detected_balls[self.current_ball_color] = {
                            'x': float(avg_x), 'y': float(avg_y),
                            'std_x': float(std_x), 'std_y': float(std_y),
                            'samples': len(self.localization_samples)
                        }
                        self.localized_colors.add(self.current_ball_color)

                        self.get_logger().info(
                            f'✓ {self.current_ball_color.upper()} ball localized at '
                            f'({avg_x:.2f}, {avg_y:.2f}) with std=({std_x:.3f}, {std_y:.3f})'
                        )

                        self.localization_samples = []
                        self.localization_timeout = 0
                        self.state = 'WALL_FOLLOW'

                twist = Twist()  # localization 保持静止

        self.cmd_vel_pub.publish(twist)

    def destroy_node(self):
        self.get_logger().info('Shutting down Ball Searcher...')
        self.cmd_vel_pub.publish(Twist())
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BallSearcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
