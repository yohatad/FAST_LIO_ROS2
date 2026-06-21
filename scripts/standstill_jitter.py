#!/usr/bin/env python3
"""
Standstill Pose Jitter Logger
-------------------------------
Subscribes to /Odometry (FAST-LIO's nav_msgs/Odometry output) for a fixed
duration while the platform is stationary, and reports the position/yaw
jitter. Use this to compare before/after config changes (acc_cov,
extrinsic_est_en, IMU notch filter, etc.) on a quiet baseline.

Usage:
    python3 standstill_jitter.py [--topic /Odometry] [--duration 60]
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class JitterLogger(Node):

    def __init__(self, topic):
        super().__init__("standstill_jitter_logger")
        self.positions = []
        self.yaws = []
        self.create_subscription(Odometry, topic, self._cb, 50)

    def _cb(self, msg):
        p = msg.pose.pose.position
        self.positions.append((p.x, p.y, p.z))
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.yaws.append(yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/Odometry")
    parser.add_argument("--duration", type=float, default=60.0)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = JitterLogger(args.topic)
    print(f"Logging {args.duration:.0f}s of {args.topic} ... keep the platform stationary.")
    start = time.time()
    try:
        while rclpy.ok() and (time.time() - start) < args.duration:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        print("Interrupted, reporting on whatever was collected so far.")
    node.destroy_node()
    rclpy.shutdown()

    pos = np.array(node.positions)
    yaw = np.array(node.yaws)
    n = len(pos)
    if n < 10:
        print(f"Only collected {n} samples, not enough to analyze.")
        return

    print(f"\nCollected {n} samples")
    mean = pos.mean(axis=0)
    for i, axis in enumerate("xyz"):
        col = pos[:, i]
        print(f"  {axis}: mean={mean[i]:+.4f}  range={col.ptp():.4f} m  std={col.std():.4f} m")

    radial = np.linalg.norm(pos - mean, axis=1)
    print(f"  3D radial deviation from mean: max={radial.max():.4f} m  rms={np.sqrt((radial**2).mean()):.4f} m")

    yaw_unwrapped = np.unwrap(yaw)
    print(f"  yaw: range={np.degrees(yaw_unwrapped.ptp()):.3f} deg  std={np.degrees(yaw_unwrapped.std()):.3f} deg")


if __name__ == "__main__":
    main()
