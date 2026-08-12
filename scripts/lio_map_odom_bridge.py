#!/usr/bin/env python3
"""Publishes odom -> base_footprint from FAST-LIO's odometry (REP-105).

FAST-LIO (configured with publish.publish_tf=false, publish.map_frame=odom,
publish.body_frame=l2lidar_frame_imu) publishes a nav_msgs/Odometry message:

    odom -> l2lidar_frame_imu   (frame_id=odom, child_frame_id=l2lidar_frame_imu)

l2lidar_frame_imu is also a static child of base_footprint
(base_footprint -> l2lidar_frame -> l2lidar_frame_imu, both static transforms).
This node closes the tree

    odom -> base_footprint -> base_link -> l2lidar_frame -> l2lidar_frame_imu

by publishing the single edge odom -> base_footprint:

    odom -> base_footprint = (odom -> l2lidar_frame_imu) * (base_footprint -> l2lidar_frame_imu)^-1

KEY DESIGN POINT
----------------
FAST-LIO's own TF broadcast MUST be disabled (publish.publish_tf=false).
Otherwise l2lidar_frame_imu would get two parents (odom directly, and
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
one-time STATIC transform odom -> odom_lidar, computed from the first
odometry sample so that odom's axes match base_footprint's axes at
t=0 (i.e. odom is Z-up, assuming the robot was level/stationary at
startup). Use odom as RViz's fixed frame for an upright view; the
TF tree below it (point cloud, path, robot model) is rotated as a whole,
so everything stays consistent.
"""

import time

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


def yaw_only_rotation(r: np.ndarray) -> np.ndarray:
    """Projects a rotation matrix onto pure yaw (about Z), discarding roll/pitch."""
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


class LioMapOdomBridge(Node):

    def __init__(self):
        super().__init__('lio_map_odom_bridge')

        # Frames
        self.declare_parameter('odom_frame', 'odom_lidar')
        self.declare_parameter('base_frame', 'base_footprint')
        # Physical sensor frame that FAST-LIO's renamed body (l2lidar_frame_imu)
        # corresponds to in the static tree (base_footprint -> ... -> l2lidar_frame_imu).
        self.declare_parameter('lidar_imu_frame', 'l2lidar_frame_imu')
        # FAST-LIO odometry topic (frame_id=odom_frame, child_frame_id=lidar_imu_frame).
        self.declare_parameter('odom_topic', '/odom_lio')
        # One-time static odom -> odom_lidar, for an upright RViz fixed frame.
        self.declare_parameter('level_frame', 'odom')
        # Disable when something above owns odom (e.g. PGO's map -> odom), so
        # odom does not end up with two parents (odom AND map).
        self.declare_parameter('publish_level_frame', True)
        # When a higher layer owns odom (a localizer publishing map -> odom),
        # level_frame cannot be odom's PARENT without giving odom two parents.
        # Publishing the inverse as a CHILD is geometrically identical, keeps
        # one parent per frame, and means level_frame -- the frame the whole
        # stack is supposed to standardise on -- exists in EVERY configuration
        # instead of vanishing exactly where a localizer is running.
        self.declare_parameter('level_frame_as_child', False)
        # Put the level frame's z=0 on the FLOOR (base_frame's height at t=0)
        # rather than at the LIO start pose (~lidar mount height). base_frame
        # is by definition the ground-plane projection of the robot, so this
        # makes z=0 mean "the floor" for everything downstream.
        self.declare_parameter('level_frame_on_floor', True)
        # Where the leveling transform comes from:
        #   'calibration' (default) -- read it straight off the static
        #       base_frame -> lidar_imu_frame chain: R_level = R_base_imu,
        #       floor offset = t_base_imu.z. The mount is rigid and calibrated
        #       once, so this is exact, identical on every backend and every
        #       run, and available before any odometry arrives.
        #   'odometry' -- legacy: snapshot odom -> base_frame from the FIRST
        #       odometry message. Equivalent ONLY if the robot has not moved by
        #       then; in practice it has, and that motion leaks in (measured:
        #       0.257 m FAST-LIO vs 0.232 m Point-LIO for the same rig, because
        #       the two backends start publishing at different points in init).
        #
        # 'calibration' REQUIRES the LIO's world frame to start at identity
        # rotation -- true for FAST-LIO always, and for Point-LIO only with
        # mapping.gravity_align:=false (see point_lio/config/l2lidar_node.yaml).
        # If gravity_align is ever re-enabled, the mount rotation would be
        # applied twice; use 'odometry' in that case.
        self.declare_parameter('level_source', 'calibration')
        # Hard flat-floor assumption: zero the LEVELED z/roll/pitch of the
        # published odom -> base_footprint every cycle (keeping x, y, yaw),
        # instead of publishing FAST-LIO's own drifting estimate of them.
        # Off by default -- this discards real 3D pose information, so it's
        # only correct on robots that are always on a genuinely flat floor
        # (ramps/thresholds/uneven terrain would silently be wrong). Requires
        # publish_level_frame: true, since it needs that rotation snapshot to
        # know which direction is actually "up" in the raw (tilted) odom frame.
        self.declare_parameter('flatten_base_frame', False)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.lidar_imu_frame = self.get_parameter('lidar_imu_frame').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.level_frame = self.get_parameter('level_frame').value
        self.publish_level = self.get_parameter('publish_level_frame').value
        self.level_as_child = self.get_parameter('level_frame_as_child').value
        self.level_on_floor = self.get_parameter('level_frame_on_floor').value
        self.level_source = self.get_parameter('level_source').value
        if self.level_source not in ('calibration', 'odometry'):
            self.get_logger().error(
                f"level_source '{self.level_source}' is not 'calibration' or "
                f"'odometry'; falling back to 'calibration'.")
            self.level_source = 'calibration'
        self.flatten_base_frame = self.get_parameter('flatten_base_frame').value

        if self.flatten_base_frame and not self.publish_level:
            self.get_logger().error(
                "flatten_base_frame is true but publish_level_frame is false -- "
                "flattening needs that rotation snapshot to know which way is "
                "up in the raw (tilted) odom frame. Flattening will be skipped "
                "until publish_level_frame is also enabled.")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # base_footprint -> l2lidar_frame_imu is static; cache it once we have it.
        self._m_base_imu = None
        self._base_imu_wait_start = None
        # odom -> odom_lidar is published once, from the first odom sample.
        self._level_published = False
        # Rotation-only part of the odom -> odom_lidar snapshot, cached for
        # flatten_base_frame (see _flatten_to_level).
        self._level_rotation = None
        # Its translation z, i.e. how far the level frame's origin was dropped
        # to land on the floor. Kept so _flatten_to_level clamps to the same
        # datum the published transform uses.
        self._level_z = 0.0

        self.sub = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, 10)

        # With level_source='calibration' the leveling depends only on the
        # static TF, so publish it as soon as that resolves rather than waiting
        # for the first odometry message (removes a startup race for anything
        # that needs level_frame early, e.g. an RViz fixed frame).
        self._level_timer = None
        if self.publish_level and self.level_source == 'calibration':
            self._level_timer = self.create_timer(0.2, self._try_publish_level_early)

        level_msg = (f" and (once) {self.level_frame} -> {self.odom_frame}"
                     if self.publish_level else
                     f" (leveling {self.level_frame} DISABLED; a higher layer "
                     f"owns {self.odom_frame})")
        self.get_logger().info(
            f"lio_map_odom_bridge: consuming {self.odom_topic} "
            f"({self.odom_frame} -> {self.lidar_imu_frame}), publishing "
            f"{self.odom_frame} -> {self.base_frame}{level_msg}. "
            f"Ensure FAST-LIO's own TF broadcast (publish.publish_tf) is DISABLED.")

    def _try_publish_level_early(self):
        """Publish the calibration-derived level frame without odometry."""
        if self._level_published:
            self._level_timer.cancel()
            return
        if self._lookup_base_imu(None) is None:
            return          # static chain not up yet; _lookup_base_imu warns
        self._publish_level_frame()
        self._level_timer.cancel()

    def _lookup_base_imu(self, stamp):
        """Static base_footprint -> l2lidar_frame_imu. Cached after first success."""
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
            if self._base_imu_wait_start is None:
                self._base_imu_wait_start = time.monotonic()
            waited = time.monotonic() - self._base_imu_wait_start

            if waited > 10.0:
                # Past a normal startup race -- this is very likely a missing
                # launch dependency, not a transient timing issue. Keep
                # retrying (the frame may still appear), but make it loud and
                # actionable instead of a quiet WARN forever: this exact
                # silent-hang footgun is what happens if pepper_sensor_tf
                # (or whatever provides this static chain) was never launched.
                self.get_logger().error(
                    f"Still no static {self.base_frame} -> "
                    f"{self.lidar_imu_frame} after {waited:.0f}s: {ex}. "
                    f"This node cannot publish anything until that transform "
                    f"exists -- is 'pepper_sensor_tf.launch.py' (or equivalent) "
                    f"running? No output will appear until it is.",
                    throttle_duration_sec=15.0)
            else:
                self.get_logger().warn(
                    f"Waiting for static {self.base_frame} -> "
                    f"{self.lidar_imu_frame}: {ex}",
                    throttle_duration_sec=5.0)
            return None

    def _publish_level_frame(self, m_odom_base: np.ndarray = None):
        """One-time static odom -> odom_lidar (rotation only).

        Chosen so odom's axes match base_footprint's axes at t=0,
        i.e. odom is Z-up assuming the robot was level/stationary
        at startup. odom itself is left untouched.
        """
        m_corr = np.eye(4)
        if self.level_source == 'calibration':
            # The LIO's world frame starts at identity rotation, so odom's axes
            # ARE the lidar-IMU's axes and the leveling rotation is exactly the
            # static mount rotation. Likewise the level frame's origin sits one
            # mount height above the floor, so that height IS the floor offset.
            # Both read off the calibration -- no runtime measurement, nothing
            # to contaminate.
            m_corr[:3, :3] = self._m_base_imu[:3, :3]
            if self.level_on_floor:
                m_corr[2, 3] = float(self._m_base_imu[2, 3])
        else:
            m_corr[:3, :3] = m_odom_base[:3, :3].T  # inverse of a rotation = transpose
            # Rotation alone leaves the level frame's origin at the LIO start
            # pose, roughly lidar-mount height above the ground. Drop it onto
            # the floor by cancelling base_frame's leveled height at t=0
            # (base_frame is on the ground plane by definition). x/y stay put --
            # only the vertical datum changes.
            if self.level_on_floor:
                m_corr[2, 3] = -float((m_corr[:3, :3] @ m_odom_base[:3, 3])[2])

        self._level_rotation = m_corr[:3, :3].copy()  # cached for flatten_base_frame
        self._level_z = float(m_corr[2, 3])

        # As a child, publish the INVERSE so the geometry is unchanged and only
        # the parent/child roles swap.
        m_pub = np.linalg.inv(m_corr) if self.level_as_child else m_corr
        translation, quaternion = matrix_to_translation_quaternion(m_pub)

        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        if self.level_as_child:
            out.header.frame_id = self.odom_frame
            out.child_frame_id = self.level_frame
        else:
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
            f"Published static "
            f"{self.odom_frame + ' -> ' + self.level_frame if self.level_as_child else self.level_frame + ' -> ' + self.odom_frame} "
            f"(one-time leveling from {self.level_source}, z=0 on the "
            f"{'floor' if self.level_on_floor else 'LIO start pose'}, "
            f"offset {self._level_z:+.3f} m; use '{self.level_frame}' as "
            f"RViz's fixed frame for an upright view).")

    def _flatten_to_level(self, m_odom_base: np.ndarray) -> np.ndarray:
        """Project odom -> base_footprint onto a flat floor (flatten_base_frame).

        Zeros the LEVELED z, roll, and pitch -- i.e. what FAST-LIO's own
        (drifting) 3D estimate says about height and tilt gets discarded and
        replaced with "the robot is exactly where it started, vertically",
        every cycle. x, y, yaw pass through untouched. This is a hard
        assumption, not a sensor-fused correction: it's only valid because
        this robot is known to always be on a genuinely flat floor.
        """
        r_level = self._level_rotation
        p_odom = m_odom_base[:3, 3]
        r_odom = m_odom_base[:3, :3]

        # Raw odom's axes don't correspond to real-world up/forward/side (see
        # the module docstring) -- have to go via the leveled frame to know
        # which component is actually vertical before zeroing it.
        p_level = r_level @ p_odom
        # Clamp to the same vertical datum the published level frame uses, so
        # "flat" means the floor (z=0 in the level frame) rather than the plane
        # through the LIO origin. With level_frame_on_floor off, _level_z is 0
        # and this reduces to the original behaviour.
        p_level[2] = -self._level_z
        r_level_pose = r_level @ r_odom
        r_level_flat = yaw_only_rotation(r_level_pose)

        r_level_inv = r_level.T  # rotation matrix inverse = transpose
        out = np.eye(4)
        out[:3, :3] = r_level_inv @ r_level_flat
        out[:3, 3] = r_level_inv @ p_level
        return out

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

        # odom -> l2lidar_frame_imu straight from the message pose.
        m_odom_imu = pose_to_matrix(msg.pose.pose.position,
                                     msg.pose.pose.orientation)

        # odom -> base_footprint = (odom -> l2lidar_frame_imu) * (base_footprint -> l2lidar_frame_imu)^-1
        m_odom_base = m_odom_imu @ np.linalg.inv(m_base_imu)

        if self.publish_level and not self._level_published:
            self._publish_level_frame(m_odom_base)

        if self.flatten_base_frame and self._level_rotation is not None:
            m_odom_base = self._flatten_to_level(m_odom_base)

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
