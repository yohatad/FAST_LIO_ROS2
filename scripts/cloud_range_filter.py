#!/usr/bin/env python3
"""Clean a scan cloud before octomap: self-hit range cut + radius-outlier removal.

Two filters, applied to /cloud_registered_body and republished for octomap only
(FAST-LIO's SLAM input is untouched):

1. RANGE CUT -- the L2 sits low (~0.26 m above base_footprint) so it sees
   Pepper's own body: a dense cluster within ~0.6 m of the sensor that octomap
   would insert as an obstacle at every pose (a black trail). Keep only points
   with min_range <= range <= max_range (range = norm in the body frame, whose
   origin is the sensor).

2. RADIUS-OUTLIER REMOVAL -- isolated spike returns (few neighbours) become
   stray obstacles and spawn thin clearing-ray "spokes". Drop any point with
   fewer than ror_min_neighbors others within ror_radius. Brute-force via the
   |a-b|^2 = |a|^2+|b|^2-2a.b identity (per-scan clouds are ~1.5k pts, cheap);
   no KDTree/scipy needed. Set ror_min_neighbors=0 to disable.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


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

        in_topic = self.get_parameter('input_topic').value
        self.out_topic = self.get_parameter('output_topic').value
        self.min_r = float(self.get_parameter('min_range').value)
        self.max_r = float(self.get_parameter('max_range').value)
        self.ror_k = int(self.get_parameter('ror_min_neighbors').value)
        self.ror_r = float(self.get_parameter('ror_radius').value)

        qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, qos)
        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb, qos)
        ror = (f", ROR >={self.ror_k} nbrs in {self.ror_r} m"
               if self.ror_k > 0 else ", ROR off")
        self.get_logger().info(
            f"cloud_range_filter: {in_topic} -> {self.out_topic}, keep "
            f"{self.min_r} m <= range" +
            (f" <= {self.max_r} m" if self.max_r > 0 else " (no max)") + ror + ".")

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
