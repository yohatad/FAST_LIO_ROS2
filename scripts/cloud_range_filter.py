#!/usr/bin/env python3
"""Clean a scan cloud before octomap/costmap: self-hit cut + outlier removal
+ optional ground-plane removal.

Three filters, applied to an input cloud and republished (the SLAM input, and
any other consumer of the original topic, are untouched):

1. RANGE CUT -- the L2 sits low (~0.26 m above base_footprint) so it sees
   Pepper's own body: a dense cluster within ~0.6 m of the sensor that would
   otherwise be inserted as an obstacle at every pose (a black trail). Keep
   only points with min_range <= range <= max_range (range = norm in the body
   frame, whose origin is the sensor).

2. RADIUS-OUTLIER REMOVAL -- isolated spike returns (few neighbours) become
   stray obstacles and spawn thin clearing-ray "spokes". Drop any point with
   fewer than ror_min_neighbors others within ror_radius. Brute-force via the
   |a-b|^2 = |a|^2+|b|^2-2a.b identity (per-scan clouds are ~1.5k pts, cheap);
   no KDTree/scipy needed. Set ror_min_neighbors=0 to disable.

3. GROUND-PLANE REMOVAL (off by default) -- for feeding nav2's costmap
   voxel_layer, which marks obstacles with a FIXED height band evaluated in
   its global_frame (odom/map). That frame's leveling is a one-time
   snapshot (see lio_map_odom_bridge.py); residual tilt at that instant grows
   with distance/time, so a fixed band eventually mis-marks the real floor as
   an obstacle. base_footprint, by contrast, is republished continuously from
   FAST-LIO's live odometry and tracks the robot's true current
   gravity-referenced attitude (roll/pitch are gravity-observable and don't
   drift the way position/yaw can). So instead of a fixed band, RANSAC-fit
   the dominant near-horizontal, near-z=0 plane per scan in base_footprint
   (same idea as octomap_server's filter_ground_plane used to build this
   map) and drop its inliers -- there is no fixed band left to drift out of.
   Set remove_ground_plane=true to enable; needs a live TF to ground_frame.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)


def _transform_to_matrix(t) -> np.ndarray:
    """4x4 homogeneous matrix from a geometry_msgs TransformStamped."""
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    m = np.eye(4)
    if n < 1e-12:
        return m
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    m[:3, :3] = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])
    m[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
    return m


class CloudRangeFilter(Node):

    def __init__(self):
        super().__init__('cloud_range_filter')
        self.declare_parameter('input_topic', '/cloud_registered_body')
        self.declare_parameter('output_topic', '/cloud_registered_body_filtered')
        self.declare_parameter('min_range', 0.8)     # drop self-hits closer than this
        self.declare_parameter('max_range', 0.0)     # 0 = no upper limit
        # Radius-outlier removal. OFF by default (0): the sparse per-scan L2 cloud
        # loses real far returns at neighbourly settings. Loose radius if enabled.
        self.declare_parameter('ror_min_neighbors', 0)   # 0 disables ROR
        self.declare_parameter('ror_radius', 0.5)        # m

        # Ground-plane removal. OFF by default -- only needed when feeding a
        # fixed-height-band consumer (nav2 costmap voxel_layer); octomap's own
        # /cloud_registered_body path already gets adaptive ground handling
        # from octomap_server's filter_ground_plane, so leave this off there.
        self.declare_parameter('remove_ground_plane', False)
        self.declare_parameter('ground_frame', 'base_footprint')
        self.declare_parameter('ground_distance_thresh', 0.05)   # inlier band around fitted plane
        self.declare_parameter('ground_angle_thresh', 0.15)      # rad, normal vs +Z tolerance
        self.declare_parameter('ground_z_thresh', 0.12)          # fitted plane must sit within this of ground_frame z=0
        self.declare_parameter('ground_ransac_iterations', 60)
        self.declare_parameter('ground_min_points', 30)          # below this, skip filtering (keep all points)

        in_topic = self.get_parameter('input_topic').value
        self.out_topic = self.get_parameter('output_topic').value
        self.min_r = float(self.get_parameter('min_range').value)
        self.max_r = float(self.get_parameter('max_range').value)
        self.ror_k = int(self.get_parameter('ror_min_neighbors').value)
        self.ror_r = float(self.get_parameter('ror_radius').value)

        self.remove_ground = bool(self.get_parameter('remove_ground_plane').value)
        self.ground_frame = self.get_parameter('ground_frame').value
        self.ground_dist = float(self.get_parameter('ground_distance_thresh').value)
        self.ground_angle = float(self.get_parameter('ground_angle_thresh').value)
        self.ground_z = float(self.get_parameter('ground_z_thresh').value)
        self.ground_iters = int(self.get_parameter('ground_ransac_iterations').value)
        self.ground_min_pts = int(self.get_parameter('ground_min_points').value)
        self.tf_buffer = None
        self.tf_listener = None
        if self.remove_ground:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, qos)
        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb, qos)
        ror = (f", ROR >={self.ror_k} nbrs in {self.ror_r} m"
               if self.ror_k > 0 else ", ROR off")
        ground = (f", ground-plane removal vs {self.ground_frame}"
                  if self.remove_ground else ", ground-plane removal off")
        self.get_logger().info(
            f"cloud_range_filter: {in_topic} -> {self.out_topic}, keep "
            f"{self.min_r} m <= range" +
            (f" <= {self.max_r} m" if self.max_r > 0 else " (no max)") + ror + ground + ".")

    def _radius_outlier_keep(self, pts: np.ndarray) -> np.ndarray:
        """Keep points with >= ror_k neighbours within ror_r (self excluded)."""
        n = pts.shape[0]
        if self.ror_k <= 0 or n <= self.ror_k:
            return np.ones(n, dtype=bool)
        # squared pairwise distances via |a-b|^2 = |a|^2 + |b|^2 - 2 a.b
        sq = np.einsum('ij,ij->i', pts, pts)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
        neighbors = (d2 <= self.ror_r * self.ror_r).sum(axis=1) - 1  # drop self
        return neighbors >= self.ror_k

    def _ground_plane_keep(self, pts: np.ndarray, stamp, frame_id: str) -> np.ndarray:
        """Boolean keep-mask dropping the dominant near-horizontal, near-z=0
        plane's inliers (fit in ground_frame). Fails safe: on a missing TF or
        an inconclusive fit, keeps every point (never invents obstacles, but
        also never silently strips more than we're confident is floor)."""
        n = pts.shape[0]
        if n < 3:
            return np.ones(n, dtype=bool)
        try:
            tf = self.tf_buffer.lookup_transform(self.ground_frame, frame_id, stamp)
        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().warn(
                f"TF {self.ground_frame} <- {frame_id} unavailable, skipping "
                f"ground-plane removal for this scan: {ex}", throttle_duration_sec=5.0)
            return np.ones(n, dtype=bool)
        m = _transform_to_matrix(tf)
        pts_h = np.hstack([pts, np.ones((n, 1))])
        pts_ground = (pts_h @ m.T)[:, :3]

        rng = np.random.default_rng()
        best_inliers = None
        best_count = 0
        cos_thresh = np.cos(self.ground_angle)
        for _ in range(self.ground_iters):
            idx = rng.choice(n, size=3, replace=False)
            p0, p1, p2 = pts_ground[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm
            if normal[2] < 0:
                normal = -normal
            if normal[2] < cos_thresh:          # not near-horizontal, skip
                continue
            d = -normal @ p0
            if abs(d) > self.ground_z:           # plane not near ground_frame z=0, skip
                continue
            dist = np.abs(pts_ground @ normal + d)
            inliers = dist <= self.ground_dist
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
        if best_inliers is None or best_count < self.ground_min_pts:
            return np.ones(n, dtype=bool)
        return ~best_inliers

    def cb(self, msg: PointCloud2):
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts.shape[0] == 0:
            return
        r = np.linalg.norm(pts, axis=1)
        keep = r >= self.min_r
        if self.max_r > 0:
            keep &= r <= self.max_r
        pts = pts[keep]
        if pts.shape[0]:
            pts = pts[self._radius_outlier_keep(pts)]
        if self.remove_ground and pts.shape[0]:
            before = pts.shape[0]
            pts = pts[self._ground_plane_keep(pts, msg.header.stamp, msg.header.frame_id)]
            self.get_logger().info(
                f"ground-plane filter: {before} -> {pts.shape[0]} pts "
                f"({before - pts.shape[0]} dropped as floor)",
                throttle_duration_sec=2.0)
        out = pc2.create_cloud_xyz32(msg.header, pts)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CloudRangeFilter()
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
