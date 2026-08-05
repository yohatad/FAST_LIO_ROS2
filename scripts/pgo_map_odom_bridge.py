#!/usr/bin/env python3
"""Publishes the REP-105 map -> odom loop-closure correction from PGO.

FAST-LIO owns   odom -> base_footprint  (via lio_map_odom_bridge).
PGO owns        map  -> odom            (this node) -- the loop-closure
correction, so the full tree is

    map -> odom -> base_footprint -> l2lidar_frame -> l2lidar_frame_imu

Both FAST-LIO's /Odometry and PGO's /aft_pgo_odom describe the SAME physical
body (the l2lidar_frame_imu pose), just in two world frames:

    /Odometry      : odom -> l2lidar_frame_imu   (dense, from FAST-LIO)
    /aft_pgo_odom  : map  -> l2lidar_frame_imu   (sparse, latest optimized keyframe)

so the correction at a shared timestamp t_k is

    map -> odom = (map -> l2lidar_frame_imu)_pgo  *  (odom -> l2lidar_frame_imu)_fastlio^-1

We keep a short buffer of FAST-LIO odometry, match each PGO keyframe pose to the
FAST-LIO sample nearest its stamp, and republish the latest correction on every
FAST-LIO odom tick (identity until the first correction arrives). This is the
same pattern nav2/slam_toolbox use: map -> odom updates at loop closures and is
otherwise constant, while odom -> base_footprint stays smooth.

LEVELING (optional, visualization only)
---------------------------------------
map (like odom) is tied to the IMU's mounting tilt at t=0, so it is not
gravity-aligned. When publish_level_frame is true we also publish a single
static map -> map_lidar so map's axes match base_footprint's at t=0
(Z-up, assuming the robot was level at startup). Use map as RViz's fixed
frame for an upright, world-fixed view without disturbing the map->odom chain.
"""

import bisect
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
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
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
    r = matrix[:3, :3]
    t = matrix[:3, 3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q)
    return t, q


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class PgoMapOdomBridge(Node):

    def __init__(self):
        super().__init__('pgo_map_odom_bridge')

        self.declare_parameter('map_frame', 'map_lidar')
        self.declare_parameter('odom_frame', 'odom_lidar')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('level_frame', 'map')
        # Dense FAST-LIO odometry (odom -> l2lidar_frame_imu).
        self.declare_parameter('odom_topic', '/Odometry')
        # Latest optimized keyframe pose (map -> l2lidar_frame_imu).
        self.declare_parameter('pgo_odom_topic', '/aft_pgo_odom')
        self.declare_parameter('publish_level_frame', True)
        # Put the level frame's z=0 on the FLOOR (base_frame's height at t=0)
        # rather than at the LIO start pose (~lidar mount height). base_frame
        # is by definition the ground-plane projection of the robot, so this
        # makes z=0 mean "the floor" for everything downstream -- notably
        # octomap's /projected_map, which hardcodes its grid origin to z=0.
        self.declare_parameter('level_frame_on_floor', True)
        # Physical sensor frame the LIO's body corresponds to in the static
        # tree; only used by level_source='calibration'.
        self.declare_parameter('lidar_imu_frame', 'l2lidar_frame_imu')
        # See lio_map_odom_bridge.py's level_source for the full rationale.
        # 'calibration' reads the leveling straight off the rigid, calibrated
        # base_frame -> lidar_imu_frame mount, so level_frame (and therefore
        # the map_batch.pcd pgo_node saves INTO it) is byte-identical on every
        # run and every backend. 'odometry' is the legacy runtime snapshot,
        # which drifts with however far the robot moved before the first
        # sample. Requires the LIO world frame to start at identity rotation
        # (Point-LIO: mapping.gravity_align must be false).
        self.declare_parameter('level_source', 'calibration')
        # How many recent FAST-LIO odom samples to keep for stamp matching.
        self.declare_parameter('odom_buffer_size', 1000)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.level_frame = self.get_parameter('level_frame').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.pgo_odom_topic = self.get_parameter('pgo_odom_topic').value
        self.publish_level = self.get_parameter('publish_level_frame').value
        self.level_on_floor = self.get_parameter('level_frame_on_floor').value
        self.lidar_imu_frame = self.get_parameter('lidar_imu_frame').value
        self.level_source = self.get_parameter('level_source').value
        if self.level_source not in ('calibration', 'odometry'):
            self.get_logger().error(
                f"level_source '{self.level_source}' invalid; using 'calibration'.")
            self.level_source = 'calibration'
        self.buf_size = int(self.get_parameter('odom_buffer_size').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Latest correction map -> odom (identity until first loop-closure sync).
        self._m_map_odom = np.eye(4)
        self._have_correction = False
        self._level_published = False

        # Time-sorted buffer of FAST-LIO odom: parallel lists of stamps + matrices.
        self._odom_t = []
        self._odom_m = []

        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.on_fastlio_odom, 50)
        self.sub_pgo = self.create_subscription(
            Odometry, self.pgo_odom_topic, self.on_pgo_odom, 50)

        self.get_logger().info(
            f"pgo_map_odom_bridge: correction {self.map_frame} -> "
            f"{self.odom_frame} from {self.pgo_odom_topic} vs {self.odom_topic}"
            f"; leveling {self.level_frame} -> {self.map_frame} "
            f"{'ON' if self.publish_level else 'OFF'}.")

    def on_fastlio_odom(self, msg: Odometry):
        stamp = msg.header.stamp
        t = stamp_to_sec(stamp)
        m = pose_to_matrix(msg.pose.pose.position, msg.pose.pose.orientation)

        # Insert keeping the buffer time-sorted (odom usually arrives in order).
        idx = bisect.bisect_left(self._odom_t, t)
        self._odom_t.insert(idx, t)
        self._odom_m.insert(idx, m)
        if len(self._odom_t) > self.buf_size:
            self._odom_t.pop(0)
            self._odom_m.pop(0)

        # Republish the latest correction, stamped with this odom's time.
        out = TransformStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.odom_frame
        tr, q = matrix_to_translation_quaternion(self._m_map_odom)
        out.transform.translation.x = float(tr[0])
        out.transform.translation.y = float(tr[1])
        out.transform.translation.z = float(tr[2])
        out.transform.rotation.x = float(q[0])
        out.transform.rotation.y = float(q[1])
        out.transform.rotation.z = float(q[2])
        out.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(out)

        if self.publish_level and not self._level_published:
            self._try_publish_level()

    def _nearest_odom(self, t: float):
        """FAST-LIO odom matrix nearest in time to t, or None if buffer empty."""
        if not self._odom_t:
            return None
        i = bisect.bisect_left(self._odom_t, t)
        cands = []
        if i < len(self._odom_t):
            cands.append(i)
        if i > 0:
            cands.append(i - 1)
        best = min(cands, key=lambda k: abs(self._odom_t[k] - t))
        return self._odom_m[best]

    def on_pgo_odom(self, msg: Odometry):
        t = stamp_to_sec(msg.header.stamp)
        m_odom_body = self._nearest_odom(t)
        if m_odom_body is None:
            return  # no FAST-LIO odom buffered yet
        m_map_body = pose_to_matrix(msg.pose.pose.position, msg.pose.pose.orientation)
        # map -> odom = (map -> body) * (odom -> body)^-1
        self._m_map_odom = m_map_body @ np.linalg.inv(m_odom_body)
        self._have_correction = True

    def _try_publish_level(self):
        """One-time static level_frame -> map so level_frame is Z-up at t=0."""
        m_corr = np.eye(4)
        level_z = 0.0

        if self.level_source == 'calibration':
            # The LIO world frame starts at identity rotation, so map's axes ARE
            # the lidar-IMU's axes: the leveling rotation is exactly the static
            # mount rotation, and the mount height IS the floor offset. Pure
            # calibration -- no dependence on odometry timing or robot motion.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.lidar_imu_frame, rclpy.time.Time())
            except (LookupException, ConnectivityException,
                    ExtrapolationException) as ex:
                self.get_logger().warn(
                    f"Waiting for static {self.base_frame} -> "
                    f"{self.lidar_imu_frame} to level {self.level_frame}: {ex}",
                    throttle_duration_sec=5.0)
                return
            m_base_imu = transform_to_matrix(tf)
            m_corr[:3, :3] = m_base_imu[:3, :3]
            if self.level_on_floor:
                level_z = float(m_base_imu[2, 3])
        else:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, rclpy.time.Time())
            except (LookupException, ConnectivityException,
                    ExtrapolationException) as ex:
                self.get_logger().warn(
                    f"Waiting for {self.odom_frame} -> {self.base_frame} to level "
                    f"{self.level_frame}: {ex}", throttle_duration_sec=5.0)
                return

            m_odom_base = transform_to_matrix(tf)
            # map -> base = (map -> odom) * (odom -> base); early on map->odom ~ I.
            m_map_base = self._m_map_odom @ m_odom_base
            m_corr[:3, :3] = m_map_base[:3, :3].T  # inverse rotation -> upright

            # Rotation alone leaves level_frame's origin at the LIO start pose,
            # which sits at roughly lidar-mount height above the ground.
            if self.level_on_floor:
                level_z = -float((m_corr[:3, :3] @ m_map_base[:3, 3])[2])

        tr, q = matrix_to_translation_quaternion(m_corr)
        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.level_frame
        out.child_frame_id = self.map_frame
        out.transform.translation.x = 0.0
        out.transform.translation.y = 0.0
        out.transform.translation.z = level_z
        out.transform.rotation.x = float(q[0])
        out.transform.rotation.y = float(q[1])
        out.transform.rotation.z = float(q[2])
        out.transform.rotation.w = float(q[3])
        self.static_tf_broadcaster.sendTransform(out)
        self._level_published = True
        self.get_logger().info(
            f"Published static {self.level_frame} -> {self.map_frame} "
            f"(one-time leveling from {self.level_source}, z=0 on the "
            f"{'floor' if self.level_on_floor else 'LIO start pose'}, "
            f"offset {level_z:+.3f} m; use '{self.level_frame}' as RViz "
            f"fixed frame).")


def main(args=None):
    rclpy.init(args=args)
    node = PgoMapOdomBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
