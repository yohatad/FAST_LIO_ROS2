// FAST-LOCALIZATION for ROS 2 -- localization against a prior map, ported from
// YWL0720/FAST-LOCALIZATION (ROS 1) onto this workspace's FAST-LIO2.
//
// The prior map is loaded straight into the ikd-Tree that FAST-LIO's iEKF
// registers against, so every scan is constrained by the map INSIDE the filter
// at scan rate. There is no map -> odom correction, and so nothing to jump.
//
// Initialization is automatic: ScanContext matches the current scan against the
// map's keyframe descriptors, then two-stage ICP refines against that
// keyframe's cloud. Candidates must clear an overlap gate and agree twice
// within init_agree_dist before being accepted. After the lock the map is
// READ-ONLY: map_incremental stops, so drifting scans cannot contaminate it.
//
// MAP FORMAT (built by utils/pgo_to_scancontext_map.py from a PGO run):
//   <map_dir>/pose.json    one line per keyframe: tx ty tz qw qx qy qz
//   <map_dir>/pcd/<N>.pcd  that keyframe's cloud, in ITS OWN frame
//
#include <atomic>
#include <std_msgs/msg/float32.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <cstdio>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "Scancontext/Scancontext.h"
#include <pcl/registration/icp.h>
#include <queue>

#include "lio_core.hpp"

// The estimator core -- globals, sensor callbacks, scan/IMU sync, the
// ikd-Tree local map, h_share_model, and the cloud/path publishers -- is
// shared with fastlio_mapping and lives in lio_core.hpp. It used to be
// duplicated here verbatim; ~1165 lines were byte-identical, so a fix in
// one node silently did not reach the other.
//
// What remains below is localization-specific: the prior map and its
// ScanContext DB, the TF-child resolution, publish_odometry (which unlike
// mapping's also emits /localization/pose with a lever-arm-corrected
// twist), the initial-pose search thread, and the lifecycle node.

/*** FAST-LOCALIZATION state ***/
SCManager scManager;                       // ScanContext descriptor DB of the prior map
PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
KD_TREE<PointType>::Ptr ikdtree_global(new KD_TREE<PointType>());  // prior map as an ikd-Tree, moved into ikdtree on lock
std::vector<V3D, Eigen::aligned_allocator<V3D>> position_map;   // keyframe positions (map)
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pose_map;
// Odometry trail during the init phase. A ScanContext match names a SCAN, not
// "now", so its pose must be carried forward by the odometry accumulated since.
std::vector<V3D, Eigen::aligned_allocator<V3D>> position_init;
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pose_init;
std::queue<std::pair<int, PointCloudXYZI::Ptr>> init_feats_down_bodys;
std::mutex init_feats_mutex, init_state_mutex;
// Sliding window of accepted init candidates (see global_localization_thread).
// Global so rearm_search() can drop it on /relocalize.
std::vector<int> candidate_ids;
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> candidate_poses;
std::mutex candidate_mutex;
bool  global_localization_finish = false;  // a lock has been found
bool  global_update = false;               // the current lock has been APPLIED
// Cleared by on_deactivate() to stop global_localization_thread. Distinct from
// global_localization_finish ("found a lock", not "stop running").
bool  keep_searching = true;
bool  map_swapped   = false;               // ikdtree already holds the prior map
std::atomic<bool> relocalize_requested{false};
// A pose handed in on /initialpose, waiting to be applied by the scan pipeline.
// Applied as-is, unlike a ScanContext lock, which names a past scan and must be
// carried forward by the odometry since.
std::mutex seed_mutex;
Eigen::Matrix4d pending_seed = Eigen::Matrix4d::Identity();
std::atomic<bool> has_pending_seed{false};
int   init_count = 0;
std::pair<int, Eigen::Matrix4d> init_result;
// pose.json is the map's identity and lives tracked in pepper_navigation; the
// keyframe clouds are bulk binary and are addressed separately.
std::string map_dir_param;                 // holds the pose file
std::string map_pose_file_param;           // pose file; bare name or absolute path
std::string map_scan_dir_param;            // holds <N>.pcd; defaults to <map_dir>/pcd
int    init_agree_count = 2;               // independent locks that must agree
double init_agree_dist  = 2.0;             // metres they must agree within
double init_icp_coarse  = 5.0, init_icp_fine = 1.0;
// ScanContext descriptor geometry -- sized to THIS sensor, not upstream's
// 64-beam car lidar. See the note in Scancontext.h.
double sc_lidar_height = 0.5, sc_max_radius = 10.0, sc_dist_thres = 0.15;
// Gates on a candidate lock. init_min_overlap is the fraction of the scan that
// must land within init_overlap_dist of the prior map; the tolerance is tight
// because a loose one saturates (at 1.0 m a lock 41 m out still scored ~100%).
// init_require_motion additionally demands the agreeing estimates come from
// scans the robot moved between -- two near-identical scans agree trivially.
// Default OFF: it blocks a stationary start, wrong when /initialpose is used.
bool   init_require_motion = false;
double init_motion_min = 0.50;    // metres of odometry between the two scans
double init_min_overlap = 0.70;
double init_overlap_dist = 0.20;  // metres; a point nearer than this is "on the map"
pcl::KdTreeFLANN<PointType>::Ptr global_map_kdtree;
int    sc_num_ring = 12, sc_num_sector = 40;
double prior_map_view_leaf = 0.20;   // display-only downsample for /prior_map
// TF child frame. FAST-LIO natively broadcasts map -> <body_frame> (the IMU),
// but REP-105 wants map -> base_footprint, and /tf_static already parents the
// IMU frame -- broadcasting it too would split the tree.
std::string tf_child_frame;                 // "" => broadcast body_frame as-is
bool  tf_child_resolved = false;
M3D   R_body_to_tfchild(Eye3d);
V3D   t_body_to_tfchild(0, 0, 0);
std::shared_ptr<tf2_ros::Buffer> tf_buffer_g;
// Logging from the free functions goes through lio_core.hpp's lio_logger()/
// lio_clock(), pointed at this node in on_configure(). This used to be a
// node pointer plus a local lio_logger()/this_clock() pair, duplicating what
// the shared header already provides.
rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_g;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_g;
bool   map_loaded = false;

/*** Resolve the STATIC body -> tf_child extrinsic, once, and cache it.
 ***
 *** Called from publish_odometry() and from the /initialpose handler. It used
 *** to live inside publish_odometry()'s `if (publish_tf_en)` block, which meant
 *** /initialpose -- which hard-rejects until tf_child_resolved -- was DEAD
 *** whenever publish_tf was false or tf_child_frame named the body frame. ***/
bool resolve_tf_child()
{
    if (tf_child_resolved) return true;
    // No distinct child frame: the filter's body frame is already the target.
    if (tf_child_frame.empty() || tf_child_frame == body_frame) {
        R_body_to_tfchild = Eye3d;
        t_body_to_tfchild = V3D(0, 0, 0);
        tf_child_resolved = true;
        return true;
    }
    if (!tf_buffer_g) return false;
    try {
        auto tfs = tf_buffer_g->lookupTransform(
            body_frame, tf_child_frame, tf2::TimePointZero);
        const auto &q = tfs.transform.rotation;
        const auto &v = tfs.transform.translation;
        Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
        R_body_to_tfchild = eq.toRotationMatrix();
        t_body_to_tfchild = V3D(v.x, v.y, v.z);
        tf_child_resolved = true;
        return true;
    } catch (const tf2::TransformException &ex) {
        // Throttled, not silent: returning quietly here shows up only as nav2
        // waiting forever for a map frame.
        RCLCPP_WARN_THROTTLE(lio_logger(), lio_clock(), 5000,
            "cannot resolve %s -> %s yet (%s)",
            body_frame.c_str(), tf_child_frame.c_str(), ex.what());
        return false;
    }
}

void publish_odometry(const rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = body_frame;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);

    // Linear velocity: state_point.vel is the IKFOM state's own filtered
    // estimate, in the world frame. nav_msgs/Odometry's twist is conventionally
    // in child_frame_id (REP 103), so rotate it by the inverse orientation.
    // Published so consumers get the EKF's smoothed velocity rather than
    // differencing poses, which amplifies scan-matching jitter.
    vect3 vel_body = state_point.rot.conjugate() * state_point.vel;
    odomAftMapped.twist.twist.linear.x = vel_body[0];
    odomAftMapped.twist.twist.linear.y = vel_body[1];
    odomAftMapped.twist.twist.linear.z = vel_body[2];

    // Angular velocity: the bias-corrected gyro from the last IMU propagation,
    // already in the body frame that child_frame_id names. Never populated
    // before, which left this twist -- and the lever-arm term in
    // /localization/pose below, which reads it -- silently and permanently zero.
    const V3D w_body = p_imu->get_angvel_last();
    odomAftMapped.twist.twist.angular.x = w_body[0];
    odomAftMapped.twist.twist.angular.y = w_body[1];
    odomAftMapped.twist.twist.angular.z = w_body[2];

    // state_ikfom declares pos FIRST (use-ikfom.hpp), so the filter's tangent
    // indices are pos 0-2, rot 3-5 -- the same order geometry_msgs uses. The
    // previous swap (k = i<3 ? i+3 : i-3) therefore published the rotation
    // block where position belongs and vice versa. Inherited from upstream.
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            odomAftMapped.pose.covariance[i*6 + j] = P(i, j);
    // vel occupies state indices 12-14 (see use-ikfom.hpp / get_f). World-frame,
    // not rotated to match vel_body above.
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            odomAftMapped.twist.covariance[i*6 + j] = P(12 + i, 12 + j);

    // Publish only once every field above is populated, or the message ships
    // the PREVIOUS cycle's covariance.
    pubOdomAftMapped->publish(odomAftMapped);

    if (publish_tf_en)
    {
        geometry_msgs::msg::TransformStamped trans;
        trans.header.frame_id = map_frame;
        trans.header.stamp = odomAftMapped.header.stamp;

        // Nothing is broadcast until the extrinsic resolves -- an edge computed
        // from a missing one would be silently wrong rather than absent.
        if (!tf_child_frame.empty() && tf_child_frame != body_frame) {
            if (!resolve_tf_child()) return;
            const Eigen::Quaterniond q_mb(odomAftMapped.pose.pose.orientation.w,
                                          odomAftMapped.pose.pose.orientation.x,
                                          odomAftMapped.pose.pose.orientation.y,
                                          odomAftMapped.pose.pose.orientation.z);
            const M3D R_mb = q_mb.toRotationMatrix();
            const V3D p_mb(odomAftMapped.pose.pose.position.x,
                           odomAftMapped.pose.pose.position.y,
                           odomAftMapped.pose.pose.position.z);
            const M3D R_mc = R_mb * R_body_to_tfchild;
            const V3D p_mc = R_mb * t_body_to_tfchild + p_mb;
            const Eigen::Quaterniond q_mc(R_mc);
            trans.child_frame_id = tf_child_frame;
            trans.transform.translation.x = p_mc(0);
            trans.transform.translation.y = p_mc(1);
            trans.transform.translation.z = p_mc(2);
            trans.transform.rotation.w = q_mc.w();
            trans.transform.rotation.x = q_mc.x();
            trans.transform.rotation.y = q_mc.y();
            trans.transform.rotation.z = q_mc.z();
        } else {
            trans.child_frame_id = body_frame;
            trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
            trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
            trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
            trans.transform.rotation = odomAftMapped.pose.pose.orientation;
        }
        tf_br->sendTransform(trans);

        // /localization/pose -- the SAME pose and twist, but genuinely in
        // tf_child_frame (base_footprint), matching the TF edge just sent.
        // /Odometry keeps stock FAST-LIO semantics (child_frame_id is the IMU),
        // so anything assuming it is the robot base reads a velocity rotated by
        // the mount. lio_localization's transform_fusion fixes the same thing.
        if (pub_localization_g && tf_child_resolved) {
            nav_msgs::msg::Odometry loc;
            loc.header = odomAftMapped.header;
            loc.child_frame_id = tf_child_frame;
            loc.pose.pose.position.x = trans.transform.translation.x;
            loc.pose.pose.position.y = trans.transform.translation.y;
            loc.pose.pose.position.z = trans.transform.translation.z;
            loc.pose.pose.orientation = trans.transform.rotation;
            loc.pose.covariance = odomAftMapped.pose.covariance;

            // Twist is in child_frame_id, so the base origin's offset from the
            // body origin adds a lever-arm term:
            //   w_base = R * w_body
            //   v_base = R * v_body + w_base x (R * t)
            // with R = R_base<-body and t = base origin in body coords.
            const M3D R_bb = R_body_to_tfchild.transpose();
            const V3D r_b  = R_bb * t_body_to_tfchild;
            const auto &tw = odomAftMapped.twist.twist;
            const V3D v_b_in(tw.linear.x, tw.linear.y, tw.linear.z);
            const V3D w_b_in(tw.angular.x, tw.angular.y, tw.angular.z);
            const V3D w_o = R_bb * w_b_in;
            const V3D v_o = R_bb * v_b_in + w_o.cross(r_b);
            loc.twist.twist.linear.x = v_o(0);
            loc.twist.twist.linear.y = v_o(1);
            loc.twist.twist.linear.z = v_o(2);
            loc.twist.twist.angular.x = w_o(0);
            loc.twist.twist.angular.y = w_o(1);
            loc.twist.twist.angular.z = w_o(2);
            loc.twist.covariance = odomAftMapped.twist.covariance;
            pub_localization_g->publish(loc);
        }
    }
}



/*** Load the prior map: per-keyframe clouds + their poses, and build the
 *** ScanContext descriptor DB. Clouds are stored in their OWN frame (what
 *** ScanContext needs), and transformed into map only for global_map. ***/
bool load_prior_map(const rclcpp::Logger &log)
{
    // Named per run, not a bare pose.json: this directory also holds the other
    // stack's poses, and undated names make them silently interchangeable.
    const std::string pose_path =
        map_pose_file_param.find('/') != std::string::npos
            ? map_pose_file_param
            : map_dir_param + "/" + map_pose_file_param;
    std::ifstream pose_file(pose_path);
    if (!pose_file.is_open()) {
        RCLCPP_ERROR(log, "cannot open %s", pose_path.c_str());
        return false;
    }
    double tx, ty, tz, w, x, y, z;
    int count = 0;
    while (pose_file >> tx >> ty >> tz >> w >> x >> y >> z)
    {
        Eigen::Quaterniond q(w, x, y, z);
        q.normalize();
        V3D pos(tx, ty, tz);
        const std::string pcd = map_scan_dir_param + "/" + std::to_string(count) + ".pcd";
        PointCloudXYZI::Ptr temp(new PointCloudXYZI());
        if (pcl::io::loadPCDFile(pcd, *temp) < 0) {
            RCLCPP_ERROR(log, "cannot read %s (pose.json has %d entries so far)",
                         pcd.c_str(), count);
            return false;
        }
        position_map.push_back(pos);
        pose_map.push_back(q);
        scManager.makeAndSaveScancontextAndKeys(*temp);   // in the keyframe's own frame
        PointCloudXYZI::Ptr in_map(new PointCloudXYZI());
        pcl::transformPointCloud(*temp, *in_map, pos, q);
        *global_map += *in_map;
        count++;
    }
    if (count == 0) { RCLCPP_ERROR(log, "%s is empty", pose_path.c_str()); return false; }
    RCLCPP_INFO(log, "Prior map: %d keyframes, %zu pts", count, global_map->size());
    return true;
}

/*** Odometry pose at an init-phase scan, as map-agnostic odom <- IMU.
 *** Read under init_feats_mutex: the main thread appends while the search
 *** thread reads, and /relocalize clears these outright. ***/
bool odom_at(int id, Eigen::Matrix4d &T)
{
    std::lock_guard<std::mutex> lk(init_feats_mutex);
    if (id < 0 || id >= (int)position_init.size() || id >= (int)pose_init.size())
        return false;
    T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = pose_init[id].toRotationMatrix();
    T.block<3,1>(0,3) = position_init[id];
    return true;
}

/*** Fraction of a scan that lands on the prior map at a proposed pose.
 ***
 *** The agreement check alone is NOT sufficient: two wrong matches to the same
 *** wrong place agree with each other perfectly, so agreement measures
 *** self-consistency, not correctness. This asks the map instead -- put the
 *** scan where the candidate says and see how much of it lands on something. ***/
double map_overlap(const PointCloudXYZI::Ptr &scan_body, const Eigen::Matrix4d &T_map_body)
{
    if (!global_map_kdtree || scan_body->empty()) return 0.0;
    PointCloudXYZI::Ptr in_map(new PointCloudXYZI());
    pcl::transformPointCloud(*scan_body, *in_map, T_map_body.cast<float>());
    const double r2 = init_overlap_dist * init_overlap_dist;
    std::vector<int> idx(1); std::vector<float> d2(1);
    size_t hit = 0;
    for (const auto &pt : in_map->points) {
        if (global_map_kdtree->nearestKSearch(pt, 1, idx, d2) > 0 && d2[0] <= r2) ++hit;
    }
    return double(hit) / double(in_map->size());
}

/*** Background thread: find where we are, using ScanContext + ICP.
 *** Runs until a lock is accepted, consuming the scans the main loop queues
 *** during the init phase. Requires init_agree_count independent estimates
 *** agreeing within init_agree_dist. ***/
void global_localization_thread(rclcpp::Logger log)
{
    rclcpp::Rate rate(20);
    while (rclcpp::ok())
    {
        bool already_locked, keep_going;
        {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            already_locked = global_localization_finish;
            keep_going = keep_searching;
        }
        // on_deactivate() clears this and joins us -- exit promptly rather than
        // idle-sleeping through a transition that is waiting on us.
        if (!keep_going) return;
        // Idle, not finished: /relocalize clears this flag to re-arm the
        // search, so returning here would make relocalization impossible.
        if (already_locked) { rate.sleep(); continue; }
        if (!map_loaded) { rate.sleep(); continue; }

        auto candidate_count = []() {
            std::lock_guard<std::mutex> lk(candidate_mutex);
            return (int)candidate_ids.size();
        };
        auto still_wanted = []() {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            return keep_searching;
        };
        // keep_searching is checked HERE too, not only in the outer loop. While
        // searching -- the normal state -- the thread lives in this inner loop,
        // and on_deactivate() stops the scan timer before joining, so the
        // candidate count can never advance. Without this check the loop spins
        // forever (rclcpp::ok() is still true during a lifecycle transition) and
        // on_deactivate()/on_shutdown() block in join() permanently.
        while (candidate_count() < init_agree_count && rclcpp::ok() && still_wanted())
        {
            std::pair<int, PointCloudXYZI::Ptr> item;
            {
                std::lock_guard<std::mutex> lk(init_feats_mutex);
                if (init_feats_down_bodys.empty()) { item.second = nullptr; }
                else { item = init_feats_down_bodys.front(); init_feats_down_bodys.pop(); }
            }
            if (!item.second) { rate.sleep(); continue; }

            PointCloudXYZI::Ptr scan(new PointCloudXYZI());
            pcl::copyPointCloud(*item.second, *scan);

            scManager.makeAndSaveScancontextAndKeys(*scan);
            auto hit = scManager.detectLoopClosureID();
            const int   match_id = hit.first;
            const float yaw_init = hit.second;
            scManager.dropBackScancontextAndKeys();       // do not grow the DB with live scans
            if (match_id == -1) { continue; }

            // ScanContext resolves yaw only; undo it, then ICP for the rest.
            Eigen::Matrix4d T_sc = Eigen::Matrix4d::Identity();
            T_sc.block<3,3>(0,0) = Eigen::Matrix3d(
                Eigen::AngleAxisd(-yaw_init, V3D(0,0,1)));
            pcl::transformPointCloud(*scan, *scan, T_sc);

            PointCloudXYZI::Ptr kf_cloud(new PointCloudXYZI());
            const std::string kf_pcd = map_scan_dir_param + "/" + std::to_string(match_id) + ".pcd";
            if (pcl::io::loadPCDFile(kf_pcd, *kf_cloud) < 0) { continue; }

            // Coarse then fine: coarse survives a hit that is the right PLACE
            // but metres off; fine is the answer.
            Eigen::Matrix4d T_corr = T_sc;
            pcl::PointCloud<PointType>::Ptr unused(new pcl::PointCloud<PointType>());
            for (double maxd : {init_icp_coarse, init_icp_fine}) {
                pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(maxd);
                icp.setInputSource(scan);
                icp.setInputTarget(kf_cloud);
                icp.align(*unused);
                if (!icp.hasConverged()) { T_corr.setZero(); break; }
                Eigen::Matrix4d Ti = icp.getFinalTransformation().cast<double>();
                pcl::transformPointCloud(*scan, *scan, Ti);
                T_corr = (Ti * T_corr).eval();
            }
            if (T_corr.isZero()) { continue; }

            Eigen::Matrix4d T_kf = Eigen::Matrix4d::Identity();
            T_kf.block<3,3>(0,0) = pose_map[match_id].toRotationMatrix();
            T_kf.block<3,1>(0,3) = position_map[match_id];

            Eigen::Matrix4d T_i_l = Eigen::Matrix4d::Identity();
            T_i_l.block<3,3>(0,0) = Lidar_R_wrt_IMU;
            T_i_l.block<3,1>(0,3) = Lidar_T_wrt_IMU;

            // map <- IMU at the scan this estimate came from
            const Eigen::Matrix4d T_cand = T_kf * T_corr * T_i_l.inverse();

            // Does the scan actually fit the map there? Checked against the
            // ORIGINAL scan, in the lidar frame the pose describes.
            const Eigen::Matrix4d T_cand_lidar = T_cand * T_i_l;
            const double ov = map_overlap(item.second, T_cand_lidar);
            if (ov < init_min_overlap) {
                RCLCPP_WARN(log, "[init] discarded keyframe %d: only %.0f%% of the "
                                 "scan lands on the map there (need %.0f%%)",
                            match_id, 100.0 * ov, 100.0 * init_min_overlap);
                continue;
            }
            // Motion gate: the candidate must come from a scan the robot has
            // actually travelled from, or it is not independent evidence.
            int oldest_id = -1;
            {
                std::lock_guard<std::mutex> lk(candidate_mutex);
                if (!candidate_ids.empty()) oldest_id = candidate_ids.front();
            }
            if (init_require_motion && oldest_id != -1) {
                Eigen::Matrix4d T0, Tn;
                if (!odom_at(oldest_id, T0) || !odom_at(item.first, Tn)) continue;
                const double moved =
                    (Tn.block<3,1>(0,3) - T0.block<3,1>(0,3)).norm();
                if (moved < init_motion_min) {
                    RCLCPP_INFO(log, "[init] holding: only %.2f m travelled since "
                                     "the oldest kept estimate (need %.2f) -- move the robot",
                                moved, init_motion_min);
                    continue;
                }
            }
            int kept_count;
            {
                std::lock_guard<std::mutex> lk(candidate_mutex);
                candidate_poses.push_back(T_cand);
                candidate_ids.push_back(item.first);
                kept_count = (int)candidate_ids.size();
            }
            RCLCPP_INFO(log, "[init] candidate %d/%d: matched map keyframe %d "
                             "(%.0f%% overlap)",
                        kept_count, init_agree_count, match_id, 100.0 * ov);
        }

        std::vector<int> ids;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
        {
            std::lock_guard<std::mutex> lk(candidate_mutex);
            ids = candidate_ids;
            poses = candidate_poses;
        }
        // A concurrent /relocalize can have cleared the window mid-flight;
        // re-check rather than trust the count that satisfied the condition.
        if ((int)ids.size() < init_agree_count) continue;

        // Each pose is the map pose at ITS OWN scan, so transport each back to
        // the first scan's instant before comparing, or a correct pair is
        // penalised for the distance covered between them:
        //     predicted_0 = pose_i * T_odom(id_i)^-1 * T_odom(id_0)
        double spread = 0.0;
        for (size_t i = 1; i < poses.size(); ++i) {
            Eigen::Matrix4d Pi = poses[i];
            if (init_require_motion) {
                Eigen::Matrix4d T0, Ti;
                if (odom_at(ids[0], T0) && odom_at(ids[i], Ti)) {
                    Pi = poses[i] * Ti.inverse() * T0;
                } else {
                    RCLCPP_WARN(log, "[init] odometry trail unavailable; comparing "
                                     "estimates uncompensated");
                }
            }
            spread = std::max(spread,
                (Pi.block<3,1>(0,3) - poses[0].block<3,1>(0,3)).norm());
        }

        if (spread < init_agree_dist) {
            init_result.first  = ids[0];
            init_result.second = poses[0];
            {
                std::lock_guard<std::mutex> lk(init_state_mutex);
                global_localization_finish = true;
            }
            {
                std::lock_guard<std::mutex> lk(init_feats_mutex);
                std::queue<std::pair<int, PointCloudXYZI::Ptr>> empty;
                std::swap(init_feats_down_bodys, empty);
            }
            {
                std::lock_guard<std::mutex> lk(candidate_mutex);
                candidate_ids.clear();
                candidate_poses.clear();
            }
            RCLCPP_INFO(log, "[init] LOCKED: %d estimates agree to %.2f m (limit %.2f)",
                        init_agree_count, spread, init_agree_dist);
            continue;   // idle until /relocalize re-arms us
        }
        // Drop only the OLDEST candidate, not the whole window: a persistently
        // good match should survive being paired with one bad one.
        RCLCPP_WARN(log, "[init] rejected: estimates disagree by %.2f m (limit %.2f) "
                         "-- ambiguous place, dropping the oldest and retrying",
                    spread, init_agree_dist);
        {
            std::lock_guard<std::mutex> lk(candidate_mutex);
            if (!candidate_ids.empty()) {
                candidate_ids.erase(candidate_ids.begin());
                candidate_poses.erase(candidate_poses.begin());
            }
        }
    }
}

class LaserMappingNode : public rclcpp_lifecycle::LifecycleNode
{
public:
    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp_lifecycle::LifecycleNode("fast_lio_localization", options)
    {
        // Heavy setup in on_configure(); resource activation (timers, init
        // thread, lifecycle publishers) in on_activate().
    }

    // Supports the standard configure -> activate <-> deactivate cycle. While
    // inactive the scan timer does not exist and no publisher emits.
    //
    // on_cleanup() is DELIBERATELY NOT SUPPORTED: the prior map, ikd-Trees and
    // filter state are process-global, so there is no safe way to release and
    // reload them. A different map means relaunching the process.
    CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<bool>("publish.publish_tf", true);
        this->declare_parameter<string>("publish.map_frame", "camera_init");
        this->declare_parameter<string>("publish.body_frame", "body");

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        // FAST-LOCALIZATION parameters
        this->declare_parameter<std::string>("localization.map_dir", "");
        this->declare_parameter<std::string>("localization.map_scan_dir", "");
        this->declare_parameter<std::string>("localization.map_pose_file", "pose.json");
        this->declare_parameter<int>("localization.init_agree_count", 2);
        this->declare_parameter<double>("localization.init_agree_dist", 2.0);
        this->declare_parameter<double>("localization.init_icp_coarse", 5.0);
        this->declare_parameter<double>("localization.init_icp_fine", 1.0);
        this->get_parameter("localization.map_dir", map_dir_param);
        this->get_parameter("localization.map_scan_dir", map_scan_dir_param);
        this->get_parameter("localization.map_pose_file", map_pose_file_param);
        if (map_pose_file_param.empty()) map_pose_file_param = "pose.json";
        if (map_scan_dir_param.empty()) map_scan_dir_param = map_dir_param + "/pcd";
        this->get_parameter("localization.init_agree_count", init_agree_count);
        this->get_parameter("localization.init_agree_dist", init_agree_dist);
        this->get_parameter("localization.init_icp_coarse", init_icp_coarse);
        this->get_parameter("localization.init_icp_fine", init_icp_fine);
        this->declare_parameter<double>("localization.sc_lidar_height", 0.5);
        this->declare_parameter<double>("localization.sc_max_radius", 10.0);
        this->declare_parameter<double>("localization.sc_dist_thres", 0.15);
        this->declare_parameter<int>("localization.sc_num_ring", 12);
        this->declare_parameter<int>("localization.sc_num_sector", 40);
        this->get_parameter("localization.sc_lidar_height", sc_lidar_height);
        this->get_parameter("localization.sc_max_radius", sc_max_radius);
        this->get_parameter("localization.sc_dist_thres", sc_dist_thres);
        this->get_parameter("localization.sc_num_ring", sc_num_ring);
        this->get_parameter("localization.sc_num_sector", sc_num_sector);
        this->declare_parameter<bool>("localization.init_require_motion", true);
        this->declare_parameter<double>("localization.init_motion_min", 0.50);
        this->get_parameter("localization.init_require_motion", init_require_motion);
        this->get_parameter("localization.init_motion_min", init_motion_min);
        this->declare_parameter<double>("localization.init_min_overlap", 0.60);
        this->declare_parameter<double>("localization.init_overlap_dist", 1.0);
        this->get_parameter("localization.init_min_overlap", init_min_overlap);
        this->get_parameter("localization.init_overlap_dist", init_overlap_dist);
        this->declare_parameter<double>("localization.prior_map_view_leaf", 0.20);
        this->get_parameter("localization.prior_map_view_leaf", prior_map_view_leaf);
        // Post-lock health check: re-runs map_overlap() against the LIVE tracked
        // pose (see health_check_callback). A wrong-but-self-consistent lock
        // produces no other symptom -- effct_feat_num stays healthy.
        this->declare_parameter<double>("localization.health_min_overlap", 0.45);
        this->declare_parameter<double>("localization.health_bad_duration", 5.0);
        this->declare_parameter<double>("localization.health_check_period", 1.0);
        this->declare_parameter<bool>("localization.auto_relocalize", true);
        this->get_parameter("localization.health_min_overlap", health_min_overlap_);
        this->get_parameter("localization.health_bad_duration", health_bad_duration_);
        this->get_parameter("localization.health_check_period", health_check_period_);
        this->get_parameter("localization.auto_relocalize", auto_relocalize_);
        // Covariance the filter is reset to after an /initialpose teleport, in
        // m^2 and rad^2. Defaults are a loose ~0.7 m / ~11 deg one-sigma, which
        // is about how well an operator can place a pose in RViz.
        this->declare_parameter<double>("localization.seed_pos_cov", 0.5);
        this->declare_parameter<double>("localization.seed_rot_cov", 0.04);
        this->get_parameter("localization.seed_pos_cov", seed_pos_cov_);
        this->get_parameter("localization.seed_rot_cov", seed_rot_cov_);
        this->declare_parameter<std::string>("publish.tf_child_frame", "base_footprint");
        this->get_parameter("publish.tf_child_frame", tf_child_frame);
        // Point the shared core's logger/clock at this node, so warnings from
        // lio_core.hpp carry the node name and can be throttled.
        lio_logger_ = this->get_logger();
        lio_clock_  = this->get_clock();
        pub_localization_g =
            this->create_publisher<nav_msgs::msg::Odometry>("/localization/pose", 10);
        pubLocalizationOverlap_ =
            this->create_publisher<std_msgs::msg::Float32>("/localization/overlap", 10);
        pubDiagnostics_ =
            this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
        tf_buffer_g = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        // spin_thread=true: the listener needs its OWN thread. On this node's
        // single-threaded executor the scan callback runs the whole iEKF update
        // and blocks it, so /tf_static starves and the lookup never resolves.
        tf_listener_g = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_g, this, true);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<bool>("publish.publish_tf", publish_tf_en, true);
        this->get_parameter_or<string>("publish.map_frame", map_frame, "camera_init");
        this->get_parameter_or<string>("publish.body_frame", body_frame, "body");

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = map_frame;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
#ifdef HAVE_LIVOX
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
#endif
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        // SensorDataQoS (BEST_EFFORT), not a plain depth: a plain depth yields
        // the RELIABLE default, which matches NOTHING against the BEST_EFFORT
        // publisher every real IMU driver offers. rmw then silently delivers no
        // IMU at all -- FAST-LIO waits forever for init and prints no error.
        // Masked on bag replay by config/play_qos.yaml, which re-offers
        // /imu/data as RELIABLE, so it only appeared on the real robot.
        // keep_last(200): SensorDataQoS defaults to depth 5, far too shallow for
        // a 250 Hz IMU feeding a loop that stalls for tens of ms.
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::SensorDataQoS().keep_last(200), imu_cbk);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        //------------------------------------------------------------------------------------------------------
        // Load the prior map and arm the search BEFORE the scan pipeline starts.
        // A missing map is fatal: this node has nothing to do without one.
        if (map_dir_param.empty()) {
            RCLCPP_FATAL(this->get_logger(),
                "localization.map_dir is required -- point it at a directory "
                "holding pose.json and pcd/.");
            return CallbackReturn::FAILURE;
        }
        // Before load_prior_map: the map DB and the live scans must share the
        // same descriptor geometry or they are not comparable at all.
        scManager.set_geometry(sc_lidar_height, sc_max_radius,
                               sc_num_ring, sc_num_sector, sc_dist_thres);
        // The DB is a static prior map plus one live query scan, so exclude only
        // that query. Upstream's 50 is for online SLAM and here would make the
        // last 50 map keyframes permanently unmatchable -- silently fatal if the
        // robot starts where the mapping run ended.
        scManager.set_exclude_recent(1);
        RCLCPP_INFO(this->get_logger(),
            "ScanContext: %d rings x %d sectors over %.1f m, lidar height %.2f m, "
            "dist thresh %.2f", sc_num_ring, sc_num_sector, sc_max_radius,
            sc_lidar_height, sc_dist_thres);
        RCLCPP_INFO(this->get_logger(), "Map: poses %s/%s  scans %s/",
                    map_dir_param.c_str(), map_pose_file_param.c_str(),
                    map_scan_dir_param.c_str());
        if (!load_prior_map(this->get_logger())) {
            RCLCPP_FATAL(this->get_logger(),
                "failed to load prior map from %s", map_dir_param.c_str());
            return CallbackReturn::FAILURE;
        }
        global_map_kdtree.reset(new pcl::KdTreeFLANN<PointType>());
        global_map_kdtree->setInputCloud(global_map);
        ikdtree_global->set_downsample_param(filter_size_map_min);
        ikdtree_global->Build(global_map->points);
        map_loaded = true;
        RCLCPP_INFO(this->get_logger(),
            "Prior map ready (%zu pts). Searching for initial pose: ScanContext + "
            "ICP, %d estimates must agree within %.2f m.",
            global_map->size(), init_agree_count, init_agree_dist);
        // Prior map display cloud, LATCHED so RViz shows it on connect.
        // Downsampled for display only; the ikd-Tree keeps the full cloud.
        // Published from on_activate() -- a LifecyclePublisher drops publish()
        // calls made before activation.
        {
            rclcpp::QoS qos(1);
            qos.transient_local().reliable();
            pubPriorMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/prior_map", qos);
            PointCloudXYZI::Ptr shown(new PointCloudXYZI());
            pcl::VoxelGrid<PointType> vg;
            vg.setLeafSize(prior_map_view_leaf, prior_map_view_leaf, prior_map_view_leaf);
            vg.setInputCloud(global_map);
            vg.filter(*shown);
            pcl::toROSMsg(*shown, prior_map_msg_);
            prior_map_msg_.header.frame_id = map_frame;
            RCLCPP_INFO(this->get_logger(),
                "Prior map display cloud ready on %s (%zu pts at %.2f m leaf)",
                map_frame.c_str(), shown->size(), prior_map_view_leaf);
        }

        // /relocalize -- "I do not trust where I think I am". Re-arms the
        // ScanContext search from scratch.
        //
        // The prior map STAYS in the ikd-Tree while searching: the search thread
        // does not use it, and rebuilding a 4.5M-point tree on a service call is
        // not worth it. While lost the filter finds no correspondences and
        // coasts on IMU. That does not corrupt the answer -- the new lock is
        // applied relative to the odometry at lock, so the error cancels.
        srv_relocalize_ = this->create_service<std_srvs::srv::Trigger>(
            "/relocalize",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                if (!map_loaded) {
                    res->success = false;
                    res->message = "prior map not loaded";
                    return;
                }
                rearm_search();
                RCLCPP_WARN(this->get_logger(),
                    "/relocalize: searching again. The pose is NOT trustworthy "
                    "until the next 'Localized' line.");
                res->success = true;
                res->message = "Global search re-armed; watch the log for [init].";
            });

        // /initialpose -- seeded initialization, an alternative to the ScanContext
        // search rather than a replacement; whichever locks first wins.
        // RViz publishes map <- base_footprint while the filter state is
        // map <- body, so the static extrinsic is composed out. Applied through
        // the SAME path as a lock, so gravity and velocity are rotated correctly
        // and the prior map is swapped in exactly once.
        sub_initialpose_ = this->create_subscription<
            geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 1,
            [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                if (!map_loaded) {
                    RCLCPP_WARN(this->get_logger(),
                        "/initialpose ignored: prior map not loaded yet");
                    return;
                }
                if (!resolve_tf_child()) {
                    RCLCPP_WARN(this->get_logger(),
                        "/initialpose ignored: %s -> %s extrinsic not resolved yet, "
                        "so the pose cannot be converted to the filter's body frame",
                        body_frame.c_str(), tf_child_frame.c_str());
                    return;
                }
                const auto &q = msg->pose.pose.orientation;
                const auto &t = msg->pose.pose.position;
                Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
                eq.normalize();
                Eigen::Matrix4d T_map_base = Eigen::Matrix4d::Identity();
                T_map_base.block<3,3>(0,0) = eq.toRotationMatrix();
                T_map_base.block<3,1>(0,3) = V3D(t.x, t.y, t.z);

                Eigen::Matrix4d T_body_base = Eigen::Matrix4d::Identity();
                T_body_base.block<3,3>(0,0) = R_body_to_tfchild;
                T_body_base.block<3,1>(0,3) = t_body_to_tfchild;

                const Eigen::Matrix4d T_map_body = T_map_base * T_body_base.inverse();

                {
                    std::lock_guard<std::mutex> lk(seed_mutex);
                    pending_seed = T_map_body;
                }
                has_pending_seed = true;
                {
                    std::lock_guard<std::mutex> lk(init_state_mutex);
                    global_localization_finish = true; // stand the search down
                }
                RCLCPP_INFO(this->get_logger(),
                    "/initialpose accepted: seeding at x=%.2f y=%.2f yaw=%.1f deg "
                    "(map -> %s)", t.x, t.y,
                    std::atan2(2.0*(eq.w()*eq.z()+eq.x()*eq.y()),
                               1.0-2.0*(eq.y()*eq.y()+eq.z()*eq.z()))*180.0/M_PI,
                    tf_child_frame.c_str());
            });

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Configured. Waiting for activate() to start "
                                         "processing and searching for a lock.");
        return CallbackReturn::SUCCESS;
    }

    // Starts the timers (scan-rate, map, health), the background init-search
    // thread, and the lifecycle publishers -- inactive publishers drop
    // everything, so activating them is what makes publish() calls emit.
    CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
    {
        pubLaserCloudFull_->on_activate();
        pubLaserCloudFull_body_->on_activate();
        pubLaserCloudEffect_->on_activate();
        pubLaserCloudMap_->on_activate();
        pubOdomAftMapped_->on_activate();
        pubPath_->on_activate();
        pubPriorMap_->on_activate();
        pubLocalizationOverlap_->on_activate();
        pubDiagnostics_->on_activate();
        pub_localization_g->on_activate();

        prior_map_msg_.header.stamp = this->get_clock()->now();
        pubPriorMap_->publish(prior_map_msg_);

        {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            keep_searching = true;
        }
        init_thread_ = std::thread(global_localization_thread, this->get_logger());

        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms,
            std::bind(&LaserMappingNode::timer_callback, this));
        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms,
            std::bind(&LaserMappingNode::map_publish_callback, this));
        auto health_period_ms = std::chrono::milliseconds(
            static_cast<int64_t>(health_check_period_ * 1000.0));
        health_timer_ = rclcpp::create_timer(this, this->get_clock(), health_period_ms,
            std::bind(&LaserMappingNode::health_check_callback, this));

        RCLCPP_INFO(this->get_logger(), "Activated. Searching for a lock.");
        return CallbackReturn::SUCCESS;
    }

    // Inverse of on_activate(): stops the timers, signals and joins the search
    // thread, and deactivates the publishers. Subscriptions and the service
    // entry points stay alive -- harmless while inactive, and this avoids
    // re-subscribing on every cycle.
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
    {
        timer_.reset();
        map_pub_timer_.reset();
        health_timer_.reset();

        {
            // NOT global_localization_finish = true here: the thread checks
            // keep_searching first, so clearing it alone makes the thread exit
            // within one tick. Forcing a "locked" state never actually reached
            // would leave init_result holding garbage for the next activate.
            std::lock_guard<std::mutex> lk(init_state_mutex);
            keep_searching = false;
        }
        if (init_thread_.joinable()) init_thread_.join();

        pubLaserCloudFull_->on_deactivate();
        pubLaserCloudFull_body_->on_deactivate();
        pubLaserCloudEffect_->on_deactivate();
        pubLaserCloudMap_->on_deactivate();
        pubOdomAftMapped_->on_deactivate();
        pubPath_->on_deactivate();
        pubPriorMap_->on_deactivate();
        pubLocalizationOverlap_->on_deactivate();
        pubDiagnostics_->on_deactivate();
        pub_localization_g->on_deactivate();

        RCLCPP_INFO(this->get_logger(), "Deactivated.");
        return CallbackReturn::SUCCESS;
    }

    // NOT SUPPORTED -- see the class-level comment above on_configure(). The
    // prior map, both ikd-Trees and the filter state are process-global, shared
    // with free functions inherited from upstream FAST-LIO, so there is no
    // member-scoped state to release that would make a second on_configure()
    // safe. Refusing beats silently reconfiguring onto stale globals.
    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_ERROR(this->get_logger(),
            "on_cleanup is not supported by fast_lio_localization: the loaded "
            "prior map and filter state are process-global and cannot be safely "
            "released and reloaded in-process. Kill and relaunch the process to "
            "localize against a different map.");
        return CallbackReturn::FAILURE;
    }

    CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override
    {
        if (state.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
            on_deactivate(state);
        }
        return CallbackReturn::SUCCESS;
    }

    ~LaserMappingNode()
    {
        {   // see on_deactivate() for why this doesn't also force
            // global_localization_finish -- harmless here since the process
            // exits right after, but kept consistent.
            std::lock_guard<std::mutex> lk(init_state_mutex);
            keep_searching = false;
        }
        if (init_thread_.joinable()) init_thread_.join();
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

private:
    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();

            // THE HANDOVER. A lock has been found but not yet applied: move the
            // filter state into map coordinates and give it the prior map to
            // register against. From here the estimate IS the map pose -- there
            // is no map -> odom correction, which is the point of this node.
            //
            // The lock names the scan it came from and odometry has run on
            // since, so carry it forward:
            //   T_map_now = T_map_at_lock * T_odom_at_lock^-1 * T_odom_now
            // A seed takes precedence and is applied directly, with no
            // carry-forward, so it is correct with or without an existing lock.
            if (has_pending_seed.exchange(false))
            {
                Eigen::Matrix4d T_map_now;
                {
                    std::lock_guard<std::mutex> lk(seed_mutex);
                    T_map_now = pending_seed;
                }
                const M3D R_old = state_point.rot.toRotationMatrix();
                const M3D R_new = T_map_now.block<3,3>(0,0);
                const M3D R_delta = R_new * R_old.transpose();
                state_ikfom gs = state_point;
                gs.pos  = T_map_now.block<3,1>(0,3);
                gs.rot  = R_new;
                gs.vel  = R_delta * state_point.vel;
                gs.grav = S2(V3D(R_delta * state_point.grav.vec));
                kf.change_x(gs);
                // An operator's RViz click is easily a metre and several degrees
                // off, but change_x() alone leaves P at the pre-jump track's
                // confidence -- so the filter resists the map correcting the
                // seed. Reset pos (0-2) and rot (3-5) to the seed's own
                // uncertainty and let the next few scans pull it in.
                {
                    auto P = kf.get_P();
                    P.block<3, 3>(0, 0) = M3D::Identity() * seed_pos_cov_;
                    P.block<3, 3>(3, 3) = M3D::Identity() * seed_rot_cov_;
                    kf.change_P(P);
                }
                state_point = kf.get_x();
                if (!map_swapped) {
                    ikdtree = std::move(ikdtree_global);
                    map_swapped = true;
                }
                global_update = true;
                RCLCPP_INFO(this->get_logger(),
                    "Seeded from /initialpose: filter is now at x=%.2f y=%.2f z=%.2f",
                    gs.pos(0), gs.pos(1), gs.pos(2));
            }

            {
                std::unique_lock<std::mutex> lk(init_state_mutex);
                const bool locked = global_localization_finish;
                lk.unlock();
                if (locked && !global_update)
                {
                    const int id = init_result.first;
                    Eigen::Matrix4d T_odom_lock = Eigen::Matrix4d::Identity();
                    T_odom_lock.block<3,3>(0,0) = pose_init[id].toRotationMatrix();
                    T_odom_lock.block<3,1>(0,3) = position_init[id];

                    Eigen::Matrix4d T_odom_now = Eigen::Matrix4d::Identity();
                    T_odom_now.block<3,3>(0,0) = state_point.rot.toRotationMatrix();
                    T_odom_now.block<3,1>(0,3) = state_point.pos;

                    const Eigen::Matrix4d T_map_now =
                        init_result.second * T_odom_lock.inverse() * T_odom_now;

                    // The handover is a change of WORLD FRAME, not just a pose
                    // edit, so every world-frame quantity has to rotate with it
                    // -- not only pos/rot.
                    //
                    // Upstream sets pos and rot alone, which is only safe when
                    // the map and LIO start frames nearly coincide. This map is
                    // levelled ~90 deg off the LIO start attitude, so leaving
                    // grav behind points gravity sideways in the new frame and
                    // the filter diverges within a few scans.
                    const M3D R_old = state_point.rot.toRotationMatrix();
                    const M3D R_new = T_map_now.block<3,3>(0,0);
                    const M3D R_delta = R_new * R_old.transpose();

                    state_ikfom gs = state_point;
                    gs.pos  = T_map_now.block<3,1>(0,3);
                    gs.rot  = R_new;
                    gs.vel  = R_delta * state_point.vel;      // world-frame velocity
                    gs.grav = S2(V3D(R_delta * state_point.grav.vec));
                    // bg/ba are body-frame biases and offset_R/T_L_I is the
                    // lidar-IMU extrinsic; none are world-frame.
                    kf.change_x(gs);
                    state_point = kf.get_x();

                    // Hand the filter the prior map -- but only the FIRST time.
                    // ikdtree_global is moved-from afterwards.
                    if (!map_swapped) {
                        ikdtree = std::move(ikdtree_global);
                        map_swapped = true;
                    }
                    global_update = true;

                    RCLCPP_INFO(this->get_logger(),
                        "Localized: filter is now in the map frame at "
                        "x=%.2f y=%.2f z=%.2f; prior map is read-only from here.",
                        gs.pos(0), gs.pos(1), gs.pos(2));
                }
            }
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            // Only while the tree holds a live map. Against the prior map this
            // deletes points permanently (map_incremental no longer runs to
            // re-add them), so revisiting a trimmed area would find no map
            // there. Latent at the shipped cube_side_length, not at a smaller
            // one -- so guarded rather than relied upon.
            if (!map_swapped) lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree->Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree->set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree->Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree->validnum();
            kdtree_size_st = ikdtree->size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree->PCL_Storage);
                ikdtree->flatten(ikdtree->Root_Node, ikdtree->PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree->PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            // Gated on map_swapped, NOT global_update. rearm_search() clears
            // global_update, so gating on it re-opened the prior map for writing
            // for the whole duration of a /relocalize or auto-relocalize --
            // merging in exactly the scans whose pose was just declared
            // untrustworthy. Once the prior map is in the tree it stays
            // read-only for the life of the process.
            if (!map_swapped) {
                map_incremental();
            }
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            // Feed the init thread while still searching: the undistorted,
            // downsampled scan plus the odometry pose it was taken at, so a lock
            // found several scans later can be carried forward to now.
            if (!global_localization_finish)
            {
                PointCloudXYZI::Ptr snapshot(new PointCloudXYZI());
                pcl::copyPointCloud(*feats_down_body, *snapshot);
                {
                    // The trail and the queue are ONE unit: the id queued with a
                    // scan indexes into these vectors, so appending outside the
                    // lock would race the search thread and /relocalize's clear.
                    position_init.push_back(state_point.pos);
                    pose_init.push_back(state_point.rot);
                    // Bounded: ScanContext + two ICP passes is slower than the
                    // scan rate, so an unbounded queue would only grow staler.
                    if (init_feats_down_bodys.size() < 5)
                        init_feats_down_bodys.push({init_count, snapshot});
                }
                init_count++;
            }

            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree->size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en) publish_map(pubLaserCloudMap_);
    }

    // Shared by /relocalize and the auto-triggered path in health_check_callback:
    // drop the odometry trail (it indexes scans from the OLD search) and stand
    // the current lock down so global_localization_thread starts over.
    void rearm_search()
    {
        {
            std::lock_guard<std::mutex> lk(init_feats_mutex);
            std::queue<std::pair<int, PointCloudXYZI::Ptr>> empty;
            std::swap(init_feats_down_bodys, empty);
            position_init.clear();
            pose_init.clear();
            init_count = 0;
        }
        {
            // Otherwise a candidate accepted before this rearm could pair with
            // a fresh one and produce a bogus instant "agreement".
            std::lock_guard<std::mutex> lk(candidate_mutex);
            candidate_ids.clear();
            candidate_poses.clear();
        }
        {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            global_localization_finish = false;   // re-arm the search
        }
        global_update = false;                    // allow a new teleport
    }

    // Runs at health_check_period_ Hz, only once locked. Re-scores the CURRENT
    // tracked pose with the same map_overlap() used during init and publishes
    // it on /localization/overlap, giving a continuous confidence signal.
    //
    // A single low reading is NOT acted on -- an unmapped side room, a person
    // crossing the scan, or the map's edge all dip overlap briefly on a correct
    // lock. Only overlap below health_min_overlap_ for the full
    // health_bad_duration_ window is treated as a wrong lock.
    // Also published as a DiagnosticArray so rqt_robot_monitor shows this node
    // continuously, not just via log lines.
    void publish_diagnostic(uint8_t level, const std::string &message,
                             const std::string &overlap_value = "")
    {
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.level = level;
        status.name = "fastlio_localization: pose lock";
        status.hardware_id = "fast_lio_localization";
        status.message = message;
        if (!overlap_value.empty()) {
            diagnostic_msgs::msg::KeyValue kv;
            kv.key = "map_overlap";
            kv.value = overlap_value;
            status.values.push_back(kv);
        }
        diagnostic_msgs::msg::DiagnosticArray arr;
        arr.header.stamp = this->get_clock()->now();
        arr.status.push_back(status);
        pubDiagnostics_->publish(arr);
    }

    void health_check_callback()
    {
        if (!map_loaded) return;
        if (!global_update) {
            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                                "Searching for initial pose (not yet localized)");
            return;
        }
        if (feats_down_body->empty()) return;

        // map <- lidar now, built the same way pointBodyToWorld does: state_point
        // is map <- IMU post-handover, and offset_R_L_I/offset_T_L_I is the
        // filter's live (possibly online-calibrated) IMU <- lidar extrinsic.
        Eigen::Matrix4d T_body_now = Eigen::Matrix4d::Identity();
        T_body_now.block<3,3>(0,0) = state_point.rot.toRotationMatrix();
        T_body_now.block<3,1>(0,3) = state_point.pos;
        Eigen::Matrix4d T_i_l = Eigen::Matrix4d::Identity();
        T_i_l.block<3,3>(0,0) = state_point.offset_R_L_I.toRotationMatrix();
        T_i_l.block<3,1>(0,3) = state_point.offset_T_L_I;
        const Eigen::Matrix4d T_map_lidar = T_body_now * T_i_l;

        const double ov = map_overlap(feats_down_body, T_map_lidar);
        last_overlap_ = ov;

        std_msgs::msg::Float32 ov_msg;
        ov_msg.data = static_cast<float>(ov);
        pubLocalizationOverlap_->publish(ov_msg);

        char ov_str[16];
        std::snprintf(ov_str, sizeof(ov_str), "%.2f", ov);

        const rclcpp::Time now = this->get_clock()->now();
        if (ov < health_min_overlap_) {
            if (!overlap_bad_) { overlap_bad_ = true; bad_since_ = now; }
            const double bad_for = (now - bad_since_).seconds();
            RCLCPP_WARN(this->get_logger(),
                "[health] overlap %.0f%% (need %.0f%%), bad for %.1f s (limit %.1f)",
                100.0 * ov, 100.0 * health_min_overlap_, bad_for, health_bad_duration_);
            if (auto_relocalize_ && bad_for >= health_bad_duration_) {
                RCLCPP_ERROR(this->get_logger(),
                    "[health] overlap stayed below %.0f%% for %.1f s -- this lock "
                    "looks wrong. Re-arming the global search (auto_relocalize).",
                    100.0 * health_min_overlap_, bad_for);
                publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                    "Lock looks wrong; auto-relocalize re-armed the search", ov_str);
                rearm_search();
                overlap_bad_ = false;
                return;
            }
            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                "Overlap below threshold; watching before acting", ov_str);
        } else {
            overlap_bad_ = false;
            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::OK,
                "Localized; tracking the prior map", ov_str);
        }
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
#ifdef HAVE_LIVOX
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
#endif

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::thread init_thread_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap_;
    sensor_msgs::msg::PointCloud2 prior_map_msg_;   // built once in on_configure, published in on_activate
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_relocalize_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
        sub_initialpose_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    // Post-lock health check (see health_check_callback).
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr pubLocalizationOverlap_;
    rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pubDiagnostics_;
    rclcpp::TimerBase::SharedPtr health_timer_;
    double health_min_overlap_   = 0.45;
    double health_bad_duration_  = 5.0;
    double health_check_period_  = 1.0;
    bool   auto_relocalize_      = true;
    bool   overlap_bad_          = false;
    double seed_pos_cov_         = 0.5;
    double seed_rot_cov_         = 0.04;
    double last_overlap_         = 1.0;
    rclcpp::Time bad_since_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    // This node supports being driven by an external lifecycle_manager, but
    // nothing wires that up yet, so self-drive configure -> activate here to
    // keep standalone launches (localization_l2.launch.py) working exactly as
    // before: up and searching for a lock as soon as the process starts.
    auto node = std::make_shared<LaserMappingNode>();
    auto configured = node->configure();
    if (configured.id() != lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
        RCLCPP_FATAL(node->get_logger(), "configure() failed (state: %s); exiting.",
                     configured.label().c_str());
        rclcpp::shutdown();
        return 1;
    }
    auto activated = node->activate();
    if (activated.id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
        RCLCPP_FATAL(node->get_logger(), "activate() failed (state: %s); exiting.",
                     activated.label().c_str());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::spin(node->get_node_base_interface());

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
