#!/usr/bin/env python3
"""Publishes odom -> base_footprint from FAST-LIO's odometry (REP-105).

FAST-LIO (configured with publish.publish_tf=false, publish.map_frame=odom,
publish.body_frame=l2lidar_imu) publishes a nav_msgs/Odometry message:

    odom -> l2lidar_imu   (frame_id=odom, child_frame_id=l2lidar_imu)

l2lidar_imu is also a static child of base_footprint
(base_footprint -> l2lidar_frame -> l2lidar_imu, both static transforms).
This node closes the tree

    odom -> base_footprint -> base_link -> l2lidar_frame -> l2lidar_imu

by publishing the single edge odom -> base_footprint:

    odom -> base_footprint = (odom -> l2lidar_imu) * (base_footprint -> l2lidar_imu)^-1

KEY DESIGN POINT
----------------
FAST-LIO's own TF broadcast MUST be disabled (publish.publish_tf=false).
Otherwise l2lidar_imu would get two parents (odom directly, and
base_footprint via the static chain) and the tree would split -- the bug
this node exists to avoid.

Time alignment: the output transform is stamped with the odometry message's
own stamp (never wall-now).

VISUALIZATION-ONLY LEVELING FRAME
----------------------------------
FAST-LIO's odom (camera_init) is fixed to the IMU's raw mounting
orientation at t=0, which is NOT gravity-aligned -- the robot can appear
"lying down" if odom is used as RViz's fixed frame, even though everything
is internally consistent (base_footprint looks correct).

To fix this for visualization without disturbing odom (and therefore
without breaking the consistency between odom -> base_footprint and
FAST-LIO's odom-frame point cloud/path), this node also publishes a single
one-time STATIC transform odom_level -> odom, computed from the first
odometry sample so that odom_level's axes match base_footprint's axes at
t=0 (i.e. odom_level is Z-up, assuming the robot was level/stationary at
startup). Use odom_level as RViz's fixed frame for an upright view; the
TF tree below it (point cloud, path, robot model) is rotated as a whole,
so everything stays consistent.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)


def pose_to_matrix(position, orientation) -> np.ndarray:
    """4x4 homogeneous matrix from a geometry_msgs Point + Quaternion."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        # Degenerate / uninitialized quaternion -> identity rotation.
        m = np.eye(4)
        m[:3, 3] = [position.x, position.y, position.z]
        return m
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    m = np.eye(4)
    m[:3, :3] = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])
    m[:3, 3] = [position.x, position.y, position.z]
    return m


def transform_to_matrix(t: TransformStamped) -> np.ndarray:
    return pose_to_matrix(t.transform.translation, t.transform.rotation)


def matrix_to_translation_quaternion(matrix: np.ndarray):
    """Numerically stable rotation-matrix -> quaternion (largest-diagonal branch)."""
    r = matrix[:3, :3]
    t = matrix[:3, 3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # s = 4*qw
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0  # s = 4*qx
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0  # s = 4*qy
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0  # s = 4*qz
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q)
    return t, q


class LioMapOdomBridge(Node):

    def __init__(self):
        super().__init__('lio_map_odom_bridge')

        # Frames
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        # Physical sensor frame that FAST-LIO's renamed body (l2lidar_imu)
        # corresponds to in the static tree (base_footprint -> ... -> l2lidar_imu).
        self.declare_parameter('lidar_imu_frame', 'l2lidar_imu')
        # FAST-LIO odometry topic (frame_id=odom_frame, child_frame_id=lidar_imu_frame).
        self.declare_parameter('odom_topic', '/Odometry')
        # One-time static odom_level -> odom, for an upright RViz fixed frame.
        self.declare_parameter('level_frame', 'odom_level')

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.lidar_imu_frame = self.get_parameter('lidar_imu_frame').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.level_frame = self.get_parameter('level_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # base_footprint -> l2lidar_imu is static; cache it once we have it.
        self._m_base_imu = None
        # odom_level -> odom is published once, from the first odom sample.
        self._level_published = False

        self.sub = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, 10)

        self.get_logger().info(
            f"lio_map_odom_bridge: consuming {self.odom_topic} "
            f"({self.odom_frame} -> {self.lidar_imu_frame}), publishing "
            f"{self.odom_frame} -> {self.base_frame} and (once) "
            f"{self.level_frame} -> {self.odom_frame}. "
            f"Ensure FAST-LIO's own TF broadcast (publish.publish_tf) is DISABLED.")

    def _lookup_base_imu(self, stamp):
        """Static base_footprint -> l2lidar_imu. Cached after first success."""
        if self._m_base_imu is not None:
            return self._m_base_imu
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.lidar_imu_frame,
                rclpy.time.Time())  # static: latest is fine
            self._m_base_imu = transform_to_matrix(tf)
            return self._m_base_imu
        except (LookupException, ConnectivityException,
                ExtrapolationException) as ex:
            self.get_logger().warn(
                f"Waiting for static {self.base_frame} -> "
                f"{self.lidar_imu_frame}: {ex}",
                throttle_duration_sec=5.0)
            return None

    def _publish_level_frame(self, m_odom_base: np.ndarray):
        """One-time static odom_level -> odom (rotation only).

        Chosen so odom_level's axes match base_footprint's axes at t=0,
        i.e. odom_level is Z-up assuming the robot was level/stationary
        at startup. odom itself is left untouched.
        """
        m_corr = np.eye(4)
        m_corr[:3, :3] = m_odom_base[:3, :3].T  # inverse of a rotation = transpose

        translation, quaternion = matrix_to_translation_quaternion(m_corr)

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.level_frame
        out.child_frame_id = self.odom_frame
        out.transform.translation.x = float(translation[0])
        out.transform.translation.y = float(translation[1])
        out.transform.translation.z = float(translation[2])
        out.transform.rotation.x = float(quaternion[0])
        out.transform.rotation.y = float(quaternion[1])
        out.transform.rotation.z = float(quaternion[2])
        out.transform.rotation.w = float(quaternion[3])
        self.static_tf_broadcaster.sendTransform(out)
        self._level_published = True
        self.get_logger().info(
            f"Published static {self.level_frame} -> {self.odom_frame} "
            f"(one-time leveling correction; use '{self.level_frame}' as "
            f"RViz's fixed frame for an upright view).")

    def on_odom(self, msg: Odometry):
        stamp = msg.header.stamp

        # Optional sanity: warn if FAST-LIO frame ids drift from expectation.
        if msg.header.frame_id and msg.header.frame_id != self.odom_frame:
            self.get_logger().warn(
                f"Odometry frame_id '{msg.header.frame_id}' != odom_frame "
                f"'{self.odom_frame}'.", throttle_duration_sec=10.0)
        if msg.child_frame_id and msg.child_frame_id != self.lidar_imu_frame:
            self.get_logger().warn(
                f"Odometry child_frame_id '{msg.child_frame_id}' != "
                f"lidar_imu_frame '{self.lidar_imu_frame}'.",
                throttle_duration_sec=10.0)

        m_base_imu = self._lookup_base_imu(stamp)
        if m_base_imu is None:
            return

        # odom -> l2lidar_imu straight from the message pose.
        m_odom_imu = pose_to_matrix(msg.pose.pose.position,
                                     msg.pose.pose.orientation)

        # odom -> base_footprint = (odom -> l2lidar_imu) * (base_footprint -> l2lidar_imu)^-1
        m_odom_base = m_odom_imu @ np.linalg.inv(m_base_imu)

        if not self._level_published:
            self._publish_level_frame(m_odom_base)

        translation, quaternion = matrix_to_translation_quaternion(m_odom_base)

        out = TransformStamped()
        out.header.stamp = stamp                 # source time, not wall-now
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.transform.translation.x = float(translation[0])
        out.transform.translation.y = float(translation[1])
        out.transform.translation.z = float(translation[2])
        out.transform.rotation.x = float(quaternion[0])
        out.transform.rotation.y = float(quaternion[1])
        out.transform.rotation.z = float(quaternion[2])
        out.transform.rotation.w = float(quaternion[3])

        self.tf_broadcaster.sendTransform(out)


def main(args=None):
    rclpy.init(args=args)
    node = LioMapOdomBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Benign race: SIGINT can land mid-take_message, tearing down the
        # rcl context while a message is being deserialized.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
