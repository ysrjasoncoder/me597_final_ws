#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from slam_toolbox.srv import SaveMap

import numpy

class RightWallFollower(Node):
    def __init__(self):
        super().__init__("task1_right_wall_follower")

        # ===== 参数：按你提供的数值 =====
        # 若想在 launch 里改，也可以用 ros2 param 重写
        self.forward_speed = 0.2          # 前进速度
        self.angular_speed = 1.0           # 最大角速度
        self.d_des_right = 0.6             # 期望右墙距离
        self.d_front_stop = 0.6            # 前方小于此距离时停止
        self.d_front_slow = 1            # 前方用于减速的距离（当前和 stop 相同）
        self.look_deg = 10.0               # 前 & 右前扇区半宽
        self.right_span = 70.0             # 右侧扇区总宽度（±35°）
        self.run_seconds = 590.0           # 最大运行时间（秒）

        # 比例控制增益（你可以按效果再调）
        self.kp_wall = 1.0                 # 右侧贴墙控制增益
        self.kp_front = 8                # 前方避障增益
        self.kp_rf = 0.6                   # 右前补偿权重

        # ROS 通信
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        self.last_scan: LaserScan | None = None
        self.start_time = None#self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info("Right-hand wall follower started with tuned parameters.")

    # === Helper function ===
    def sector_min(self, scan: LaserScan, center_deg: float, half_width_deg: float) -> float:
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
    
    def sector(self, center_deg: float, half_width_deg: float):
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
        return vals
    
    def sector_mean(self,center_deg: float, half_width_deg: float):
        return numpy.mean(self.sector(center_deg=center_deg, half_width_deg=half_width_deg))

    # === 回调 ===
    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    # === 主控制循环（纯比例控制）===
    def control_loop(self):
        # 时间到了就停车（不退出 node，方便多次 launch）
        now = self.get_clock().now().nanoseconds / 1e9
        # self.get_logger().info(f'start:{self.start_time}')
        # self.get_logger().info(f'now:{now}')
        # 还没拿到合法的 sim time：直接返回，不做控制
        if now == 0.0:
            return

        # 第一次拿到非 0 的 sim time，当作起点
        if self.start_time is None:
            self.start_time = now
            return  # 下一次再真正开始控制
        
        if now - self.start_time > self.run_seconds:
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
            self.save_map("syn_classroom")
            return

        if self.last_scan is None:
            return

        scan = self.last_scan

        # 使用你给的 helper 函数取三个方向的最近距离
        front_min = self.sector_min(scan, center_deg=-15.0, half_width_deg=self.look_deg)
        front_mean = self.sector_mean(center_deg=-15.0, half_width_deg=self.look_deg)

        right_front_min = self.sector_min(scan, center_deg=-45.0, half_width_deg=self.look_deg)
        # right_span 是总宽度，所以 half_width = right_span / 2
        right_min = self.sector_min(scan, center_deg=-90.0, half_width_deg=self.right_span / 2.0)

        twist = Twist()

        # ===== 1. 线速度：根据前方距离减速/停止 =====
        if math.isfinite(front_min):
            if front_min < self.d_front_stop:
                # 太近了，直接停
                speed_scale = 0.0
            elif front_min < self.d_front_slow:
                # 介于 stop 和 slow 之间可以线性缩放（目前你两个参数一样，相当于全速 or 停）
                speed_scale = front_min / self.d_front_slow
            else:
                speed_scale = 1.0
        else:
            speed_scale = 1.0
        #speed_scale = 1.0
        

        twist.linear.x = self.forward_speed * max(0.0, min(1.0, speed_scale))

        # ===== 2. 角速度：右墙 + 前方 + 右前，纯 P 控制 =====
        ang = 0.0

        

        # 2.2 前方避障：前方距离越小，越向左转（正角速度）
        if math.isfinite(front_min) and front_min < self.d_front_stop and math.isfinite(front_mean) and 1.5 * front_min > self.d_front_stop:
            error_front = self.d_front_stop - front_min
            ang_front = self.kp_front * error_front
            ang = 0.7
        else:
        # 2.1 右手贴墙：right_min 与期望距离的误差
            if math.isfinite(right_min):
                error_wall = right_min - self.d_des_right
                # 右手贴墙：离墙远了（error > 0）应该往右转（负角速度），所以前面加负号
                ang_wall = - self.kp_wall * error_wall
                ang += ang_wall

            # 2.3 右前补偿：右前太近也往左一点，避免刮墙/卡角
            if math.isfinite(right_front_min) and right_front_min < self.d_des_right:
                error_rf = self.d_des_right - right_front_min
                ang_rf = self.kp_rf * self.kp_wall * error_rf
                ang += ang_rf

        # ===== 3. 限制角速度幅度 =====
        if ang > self.angular_speed:
            ang = self.angular_speed
        elif ang < -self.angular_speed:
            ang = -self.angular_speed

        twist.angular.z = ang

        self.cmd_pub.publish(twist)
    
    def save_map(self, map_name: str):
        # 创建 service client
        self.save_map_client = self.create_client(SaveMap,
                                                  '/slam_toolbox/save_map')

        # 等待服务启动
        while not self.save_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /slam_toolbox/save_map service...')
        """ 调用 slam_toolbox 的 SaveMap 服务 """
        req = SaveMap.Request()
        req.name.data = map_name

        self.get_logger().info(f"Saving map as: {map_name} ...")

        future = self.save_map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"Map saved successfully: {map_name}")
        else:
            self.get_logger().error("Failed to save map.")


def main(args=None):
    rclpy.init(args=args)
    node = RightWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down RightWallFollower...")
        node.cmd_pub.publish(Twist())  # 停车
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
