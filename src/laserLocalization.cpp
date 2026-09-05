// FAST-LOCALIZATION for ROS 2 -- localization against a prior map, ported from
// YWL0720/FAST-LOCALIZATION (ROS 1) onto this workspace's FAST-LIO2.
//
// WHY THIS EXISTS, AND HOW IT DIFFERS FROM lio_localization.
//
// lio_localization keeps FAST-LIO's own map and bolts a SEPARATE ICP node
// beside it, which registers the scan against a prior .pcd every ~0.5 s and
// emits a discrete map -> odom correction. That correction is a step, and the
// step IS the jump: MEASURED on slam_20260823_aligned, 383 correction attempts,
// 223 rejected by the innovation gate (58%), 100 forced through by the
// 3-strike escape hatch, the largest 49.72 m, growing over the run -- i.e. it
// diverged rather than settled. Fitness cannot catch this: inlierFitness is an
// inlier COUNT at max_corr_dist (1.0 m) while the gate rejects at 0.30 m, so a
// correction the gate calls implausible costs 0.000 fitness (measured: fitness
// stays 1.000 out to a full 1.0 m of deliberate offset).
//
// This node removes the correction entirely. The prior map is loaded straight
// into the ikd-Tree that FAST-LIO's iEKF registers against, so every scan is
// constrained by the map INSIDE the filter at scan rate. There is no map->odom
// step to jump, because there is no separate correction.
//
// INITIALIZATION is automatic: ScanContext matches the current scan against the
// map's keyframe descriptors, then two-stage ICP (5 m then 1 m correspondence)
// refines against that keyframe's cloud. It is required to agree TWICE within
// init_agree_dist before being accepted -- a single descriptor hit in a
// corridor is exactly the confident-wrong-lock this workspace keeps hitting.
//
// After the lock the map is READ-ONLY: map_incremental stops, so drifting scans
// cannot contaminate the prior.
//
// MAP FORMAT (built by utils/pgo_to_scancontext_map.py from a PGO run):
//   <map_dir>/pose.json    one line per keyframe: tx ty tz qw qx qy qz
//   <map_dir>/pcd/<N>.pcd  that keyframe's cloud, in ITS OWN frame
//
#include <omp.h>
#include <mutex>
#include <atomic>
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#ifdef HAVE_LIVOX
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif
#include "preprocess.h"
#include "Scancontext/Scancontext.h"
#include <pcl/registration/icp.h>
#include <thread>
#include <queue>
#include <mutex>
#include <fstream>
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   publish_tf_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string map_frame, body_frame;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** FAST-LOCALIZATION state ***/
SCManager scManager;                       // ScanContext descriptor DB of the prior map
PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
KD_TREE<PointType> ikdtree_global;         // prior map as an ikd-Tree, swapped in on lock
std::vector<V3D, Eigen::aligned_allocator<V3D>> position_map;   // keyframe positions (map)
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pose_map;
// Odometry trail during the init phase. The ScanContext match names a SCAN, not
// "now", so the pose it yields has to be carried forward by the odometry that
// accumulated since -- hence keeping the whole trail rather than just the latest.
std::vector<V3D, Eigen::aligned_allocator<V3D>> position_init;
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pose_init;
std::queue<std::pair<int, PointCloudXYZI::Ptr>> init_feats_down_bodys;
std::mutex init_feats_mutex, init_state_mutex;
bool  global_localization_finish = false;  // a lock has been found
bool  global_update = false;               // the current lock has been APPLIED
bool  map_swapped   = false;               // ikdtree already holds the prior map
std::atomic<bool> relocalize_requested{false};
// A pose handed in on /initialpose, waiting to be applied by the scan pipeline.
// Kept separate from init_result because it is applied DIFFERENTLY: a
// ScanContext lock names a past scan and must be carried forward by the odometry
// since, whereas a seed means "you are here, NOW" and is applied as-is. Feeding
// a seed through the carry-forward is wrong once a lock already exists, because
// the state is then in map while the odometry trail is in the LIO frame --
// MEASURED, that sent a correctly-localized filter from x=13.09 to x=64.34
// z=49.80.
std::mutex seed_mutex;
Eigen::Matrix4d pending_seed = Eigen::Matrix4d::Identity();
std::atomic<bool> has_pending_seed{false};
int   init_count = 0;
std::pair<int, Eigen::Matrix4d> init_result;
// The map is TWO things with very different natures, so they are addressed
// separately. pose.json is 233 KB and IS the map's identity -- it belongs in
// pepper_navigation beside pepper_map_lc_poses.txt, tracked, where the pairing
// between the two localization stacks' maps is visible. The 2735 keyframe
// clouds are 75 MB of binary that would be gitignored anyway, and putting them
// in a ROS package means install(DIRECTORY) copies all 2735 on every build.
std::string map_dir_param;                 // holds the pose file
std::string map_pose_file_param;           // pose file; bare name or absolute path
std::string map_scan_dir_param;            // holds <N>.pcd; defaults to <map_dir>/pcd
int    init_agree_count = 2;               // independent locks that must agree
double init_agree_dist  = 2.0;             // metres they must agree within
double init_icp_coarse  = 5.0, init_icp_fine = 1.0;
// ScanContext descriptor geometry -- sized to THIS sensor, not upstream's 64-beam
// car lidar. See the note in Scancontext.h: the L2's downsampled keyframes are
// ~1600 pts with 90% inside 2.9 m, so a 80 m radius leaves most of the descriptor
// empty. Radius is set from the data; rings/sectors are reduced so the bins that
// remain actually hold points (20x60 = 1200 bins for 1600 points is ~1 pt/bin).
double sc_lidar_height = 0.5, sc_max_radius = 10.0, sc_dist_thres = 0.15;
// Minimum fraction of the scan that must land on the prior map, at the pose a
// candidate proposes, for that candidate to be believed at all.
// TUNED, not guessed. At 1.0 m tolerance a wrong lock 41 m from truth still
// scored 97-100%: the map is dense (4.5M pts) and the scan sparse (~1600 pts
// mostly within 3 m), so indoors almost any pose puts most points within a
// metre of something. That is the same saturation that made the old stack's
// ICP fitness useless. At 0.20 m the wrong candidates score 50-67% and the
// right ones 99-100% -- a clean separation.
// Require the agreeing estimates to come from scans the robot actually MOVED
// between, and compare them with the odometry between compensated out.
//
// Without this, agreement is close to vacuous when starting from a standstill:
// two "independent" estimates are then taken from near-identical scans, so of
// course they agree. MEASURED at a stationary start mid-corridor: keyframes
// 2625 and 2636 -- eleven apart, the same place -- both scored 93-100% overlap
// and agreed to 0.35 m, and the lock was 40 m from truth.
//
// With it, a spurious match has to stay consistent ACROSS the robot moving,
// which is a much harder thing to be accidentally right about. The cost is that
// initialization cannot finish while stationary -- arguably correct, since one
// viewpoint genuinely cannot disambiguate a corridor.
// Default OFF. It makes an unseeded ScanContext lock far more reliable (see
// above), but it also means initialization cannot finish until the robot has
// driven init_motion_min, which is the wrong trade when an operator is going to
// supply the pose via /initialpose anyway -- a seeded start needs no
// disambiguation. Turn it ON for unattended startup with no seed.
bool   init_require_motion = false;
double init_motion_min = 0.50;    // metres of odometry between the two scans
double init_min_overlap = 0.70;
double init_overlap_dist = 0.20;  // metres; a point nearer than this is "on the map"
pcl::KdTreeFLANN<PointType>::Ptr global_map_kdtree;
int    sc_num_ring = 12, sc_num_sector = 40;
double prior_map_view_leaf = 0.20;   // display-only downsample for /prior_map
// TF child frame. FAST-LIO natively broadcasts map -> <body_frame>, which here
// is camera_imu_optical_frame -- the IMU on the mast. That is the wrong thing to
// hand a bag or nav2 for two reasons: REP-105 wants map -> base_footprint, and
// the bag's own /tf_static already publishes base_footprint -> camera_imu_-
// optical_frame, so broadcasting the IMU edge too would give that frame TWO
// parents and split the tree. Composing the (static) body -> base extrinsic on
// first lookup and broadcasting map -> base_footprint keeps one parent each.
std::string tf_child_frame;                 // "" => broadcast body_frame as-is
bool  tf_child_resolved = false;
M3D   R_body_to_tfchild(Eye3d);
V3D   t_body_to_tfchild(0, 0, 0);
std::shared_ptr<tf2_ros::Buffer> tf_buffer_g;
rclcpp::Node *node_g = nullptr;            // for logging from the free functions
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_g;
static rclcpp::Logger this_logger() {
    return node_g ? node_g->get_logger() : rclcpp::get_logger("fast_lio_localization");
}
static rclcpp::Clock::SharedPtr this_clock() {
    static auto fallback = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
    return node_g ? node_g->get_clock() : fallback;
}
std::shared_ptr<tf2_ros::TransformListener> tf_listener_g;
bool   map_loaded = false;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
#ifdef HAVE_LIVOX
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
#endif // HAVE_LIVOX

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = map_frame;
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = body_frame;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = map_frame;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = map_frame;
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = body_frame;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);

    // Linear velocity: state_point.vel is the IKFOM state's own filtered
    // velocity estimate (get_f() integrates it straight into pos, i.e. it's
    // expressed in the map/world frame). nav_msgs/Odometry's twist is
    // conventionally in child_frame_id (body frame, REP 103), so rotate it
    // by the inverse of the current orientation before publishing.
    // This is populated so downstream consumers get the EKF's own smoothed
    // velocity instead of differencing consecutive /Odometry poses
    // themselves -- differencing amplifies the ~1-2cm scan-matching jitter
    // into a badly noisy velocity (same effect that inflated raw-rate
    // "distance traveled" 2x in the travel-distance analysis).
    vect3 vel_body = state_point.rot.conjugate() * state_point.vel;
    odomAftMapped.twist.twist.linear.x = vel_body[0];
    odomAftMapped.twist.twist.linear.y = vel_body[1];
    odomAftMapped.twist.twist.linear.z = vel_body[2];

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    // vel occupies state indices 12-14 (see use-ikfom.hpp / get_f: res(i+12)
    // is vel's derivative) -- this is the world-frame vel covariance, not
    // rotated to match vel_body above, but still a useful relative measure
    // of how well-determined the velocity estimate currently is.
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            odomAftMapped.twist.covariance[i*6 + j] = P(12 + i, 12 + j);

    // publish only once every field above is populated -- previously this
    // call happened before the covariance loop, so every message shipped
    // the PREVIOUS cycle's covariance instead of its own.
    pubOdomAftMapped->publish(odomAftMapped);

    if (publish_tf_en)
    {
        geometry_msgs::msg::TransformStamped trans;
        trans.header.frame_id = map_frame;
        trans.header.stamp = odomAftMapped.header.stamp;

        // Resolve body -> tf_child once. It is a STATIC extrinsic, so a single
        // successful lookup is cached for the life of the node; until it
        // succeeds nothing is broadcast, because a map -> base edge computed
        // from a missing extrinsic would be silently wrong rather than absent.
        if (!tf_child_frame.empty() && tf_child_frame != body_frame) {
            if (!tf_child_resolved) {
                if (!tf_buffer_g) return;
                try {
                    auto tfs = tf_buffer_g->lookupTransform(
                        body_frame, tf_child_frame, tf2::TimePointZero);
                    const auto &q = tfs.transform.rotation;
                    const auto &v = tfs.transform.translation;
                    Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
                    R_body_to_tfchild = eq.toRotationMatrix();
                    t_body_to_tfchild = V3D(v.x, v.y, v.z);
                    tf_child_resolved = true;
                } catch (const tf2::TransformException &ex) {
                    // Throttled, not silent: this used to return quietly and the
                    // only visible effect was nav2 waiting forever for a map
                    // frame that was never going to arrive.
                    RCLCPP_WARN_THROTTLE(this_logger(), *this_clock(), 5000,
                        "cannot resolve %s -> %s yet (%s); NOT broadcasting "
                        "%s -> %s until it does",
                        body_frame.c_str(), tf_child_frame.c_str(), ex.what(),
                        map_frame.c_str(), tf_child_frame.c_str());
                    return;
                }
            }
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
        //
        // /Odometry keeps stock FAST-LIO semantics: child_frame_id is body_frame,
        // the IMU on the mast, and the twist is in those axes. Anything that
        // assumes child_frame_id is the robot base -- nav2's odom_topic, for
        // one -- then reads a velocity rotated by the mount, ~64 deg of yaw on
        // this rig. lio_localization's transform_fusion documents and fixes the
        // same thing; this mirrors it so both stacks publish the same contract.
        if (pub_localization_g && tf_child_resolved) {
            nav_msgs::msg::Odometry loc;
            loc.header = odomAftMapped.header;
            loc.child_frame_id = tf_child_frame;
            loc.pose.pose.position.x = trans.transform.translation.x;
            loc.pose.pose.position.y = trans.transform.translation.y;
            loc.pose.pose.position.z = trans.transform.translation.z;
            loc.pose.pose.orientation = trans.transform.rotation;
            loc.pose.covariance = odomAftMapped.pose.covariance;

            // Twist is expressed in child_frame_id, so rotating alone is not
            // enough -- the base origin sits off the body origin, so it also
            // picks up the lever-arm term:
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

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = map_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}


/*** Load the prior map: per-keyframe clouds + their poses, and build the
 *** ScanContext descriptor DB. Clouds are stored in their OWN frame (that is
 *** what ScanContext needs -- a descriptor of the local structure as seen from
 *** that pose), and transformed into map only when accumulating global_map. ***/
bool load_prior_map(const rclcpp::Logger &log)
{
    // Named per run, not a bare pose.json: this directory also holds
    // pepper_map_lc_poses.txt for the other stack, and a second map would drop
    // a second pose file beside it. An undated name makes those silently
    // interchangeable -- the same failure as a .pcd paired with the wrong poses.
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
 *** Read under init_feats_mutex: the main thread appends to these vectors while
 *** the search thread reads them, and /relocalize clears them outright. ***/
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
 *** The agreement check alone is NOT sufficient. MEASURED starting mid-bag in
 *** the corridor: ScanContext matched keyframes 203 and 191, which are adjacent
 *** to each other near the origin, so the two estimates agreed to 1.86 m and
 *** passed a 2 m limit -- while the robot was 41 m away. Two wrong matches to
 *** the same wrong place agree with each other perfectly. Agreement measures
 *** self-consistency, not correctness.
 ***
 *** This asks the map instead: put the scan where the candidate says, and see
 *** how much of it lands on something. A pose 41 m out overlaps almost nothing,
 *** and no amount of internal consistency can fake that. ***/
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
 ***
 *** Runs only until a lock is accepted. It consumes the undistorted scans the
 *** main loop queues during the init phase, and requires init_agree_count
 *** independent locks agreeing within init_agree_dist -- one descriptor hit in a
 *** corridor is not evidence, several that agree are. ***/
void global_localization_thread(rclcpp::Logger log)
{
    rclcpp::Rate rate(20);
    while (rclcpp::ok())
    {
        bool already_locked;
        {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            already_locked = global_localization_finish;
        }
        // Idle, not finished: /relocalize clears this flag to re-arm the search,
        // so returning here would make relocalization impossible for the life
        // of the process.
        if (already_locked) { rate.sleep(); continue; }
        if (!map_loaded) { rate.sleep(); continue; }

        std::vector<int> ids;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
        while ((int)ids.size() < init_agree_count && rclcpp::ok())
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

            // Coarse then fine: the coarse pass has to survive a ScanContext hit
            // that is the right PLACE but metres off; the fine pass is the answer.
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
            // ORIGINAL scan, not the ICP-transformed copy, and in the lidar
            // frame the pose describes.
            const Eigen::Matrix4d T_cand_lidar = T_cand * T_i_l;
            const double ov = map_overlap(item.second, T_cand_lidar);
            if (ov < init_min_overlap) {
                RCLCPP_WARN(log, "[init] discarded keyframe %d: only %.0f%% of the "
                                 "scan lands on the map there (need %.0f%%)",
                            match_id, 100.0 * ov, 100.0 * init_min_overlap);
                continue;
            }
            // Motion gate: the new candidate must come from a scan the robot
            // has actually travelled from, or it is not independent evidence.
            if (init_require_motion && !ids.empty()) {
                Eigen::Matrix4d T0, Tn;
                if (!odom_at(ids[0], T0) || !odom_at(item.first, Tn)) continue;
                const double moved =
                    (Tn.block<3,1>(0,3) - T0.block<3,1>(0,3)).norm();
                if (moved < init_motion_min) {
                    RCLCPP_INFO(log, "[init] holding: only %.2f m travelled since "
                                     "the first estimate (need %.2f) -- move the robot",
                                moved, init_motion_min);
                    continue;
                }
            }
            poses.push_back(T_cand);
            ids.push_back(item.first);
            RCLCPP_INFO(log, "[init] candidate %zu/%d: matched map keyframe %d "
                             "(%.0f%% overlap)",
                        ids.size(), init_agree_count, match_id, 100.0 * ov);
        }
        if ((int)ids.size() < init_agree_count) continue;

        // Each pose is the map pose at ITS OWN scan, so comparing them raw
        // penalises a correct pair for the distance the robot covered between
        // them -- the old 2 m tolerance was quietly absorbing that. Transport
        // each estimate back to the first scan's instant using the odometry
        // between, and the comparison becomes a true consistency test:
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
            RCLCPP_INFO(log, "[init] LOCKED: %d estimates agree to %.2f m (limit %.2f)",
                        init_agree_count, spread, init_agree_dist);
            continue;   // idle until /relocalize re-arms us
        }
        RCLCPP_WARN(log, "[init] rejected: estimates disagree by %.2f m (limit %.2f) "
                         "-- ambiguous place, retrying", spread, init_agree_dist);
        ids.clear(); poses.clear();
    }
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("fast_lio_localization", options)
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
        this->declare_parameter<std::string>("publish.tf_child_frame", "base_footprint");
        this->get_parameter("publish.tf_child_frame", tf_child_frame);
        node_g = this;
        pub_localization_g =
            this->create_publisher<nav_msgs::msg::Odometry>("/localization/pose", 10);
        tf_buffer_g = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        // spin_thread=true: the listener gets its OWN thread. Sharing this
        // node's single-threaded executor does not work -- the scan timer
        // callback runs the whole iEKF update and blocks it, so /tf_static
        // callbacks starve and the buffer stays empty however long you wait.
        // Symptom was a lookup that never resolved, silently, while the frames
        // were being published the whole time.
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
        // SensorDataQoS (BEST_EFFORT), not a plain depth. A plain depth yields
        // the DEFAULT profile, i.e. RELIABLE, and a RELIABLE subscriber matches
        // NOTHING against a BEST_EFFORT publisher -- which is what every real
        // IMU driver offers, l2lidar_node included. rmw then silently delivers
        // no IMU at all: FAST-LIO waits forever for IMU init, never emits
        // /Odometry, and prints no error. The lidar subscription above already
        // used SensorDataQoS, which is why /points worked while /imu/data did
        // not, and why this only ever showed up as a broken TF tree
        // (odom -> base_footprint missing, because lio_odom_bridge had no
        // odometry to close it with).
        //
        // This was masked on bag replay by config/play_qos.yaml, which
        // re-offers /imu/data as RELIABLE. There is no such override when the
        // driver is live, so the bug only appeared on the real robot.
        // BEST_EFFORT readers match BOTH kinds of writer, so this is strictly
        // more compatible than what it replaces.
        //
        // keep_last(200): SensorDataQoS defaults to depth 5, far too shallow
        // for a 250 Hz IMU feeding a mapping loop that stalls for tens of ms.
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
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        // Load the prior map and arm the search BEFORE the scan pipeline starts,
        // so no scan is processed against an empty map. A missing or unreadable
        // map is fatal here: this node localizes, it has nothing to do without one.
        if (map_dir_param.empty()) {
            RCLCPP_FATAL(this->get_logger(),
                "localization.map_dir is required -- point it at a directory "
                "holding pose.json and pcd/.");
            throw std::runtime_error("localization.map_dir not set");
        }
        // Before load_prior_map: the map DB and the live scans must be described
        // with identical geometry or the descriptors are not comparable at all.
        scManager.set_geometry(sc_lidar_height, sc_max_radius,
                               sc_num_ring, sc_num_sector, sc_dist_thres);
        RCLCPP_INFO(this->get_logger(),
            "ScanContext: %d rings x %d sectors over %.1f m, lidar height %.2f m, "
            "dist thresh %.2f", sc_num_ring, sc_num_sector, sc_max_radius,
            sc_lidar_height, sc_dist_thres);
        RCLCPP_INFO(this->get_logger(), "Map: poses %s/%s  scans %s/",
                    map_dir_param.c_str(), map_pose_file_param.c_str(),
                    map_scan_dir_param.c_str());
        if (!load_prior_map(this->get_logger())) {
            throw std::runtime_error("failed to load prior map from " + map_dir_param);
        }
        global_map_kdtree.reset(new pcl::KdTreeFLANN<PointType>());
        global_map_kdtree->setInputCloud(global_map);
        ikdtree_global.set_downsample_param(filter_size_map_min);
        ikdtree_global.Build(global_map->points);
        map_loaded = true;
        RCLCPP_INFO(this->get_logger(),
            "Prior map ready (%zu pts). Searching for initial pose: ScanContext + "
            "ICP, %d estimates must agree within %.2f m.",
            global_map->size(), init_agree_count, init_agree_dist);
        // Publish the prior map once, LATCHED (transient local) so RViz shows it
        // on connect rather than only if it happens to be listening at startup.
        // Downsampled for display only -- the ikd-Tree keeps the full cloud.
        {
            rclcpp::QoS qos(1);
            qos.transient_local().reliable();
            pubPriorMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/prior_map", qos);
            PointCloudXYZI::Ptr shown(new PointCloudXYZI());
            pcl::VoxelGrid<PointType> vg;
            vg.setLeafSize(prior_map_view_leaf, prior_map_view_leaf, prior_map_view_leaf);
            vg.setInputCloud(global_map);
            vg.filter(*shown);
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*shown, msg);
            msg.header.frame_id = map_frame;
            msg.header.stamp = this->get_clock()->now();
            pubPriorMap_->publish(msg);
            RCLCPP_INFO(this->get_logger(),
                "Published /prior_map on %s (%zu pts at %.2f m leaf, latched)",
                map_frame.c_str(), shown->size(), prior_map_view_leaf);
        }

        // /relocalize -- "I do not trust where I think I am". Re-arms the
        // ScanContext search from scratch.
        //
        // The prior map STAYS in the ikd-Tree while searching. That is
        // deliberate: the search thread does not use the ikd-Tree (it works off
        // global_map_kdtree and the per-keyframe clouds), and swapping a local
        // map back in would mean destroying and rebuilding a 4.5M-point tree on
        // a service call. While lost the filter simply finds no correspondences
        // and coasts on IMU, which is what being lost IS.
        //
        // Coasting does not corrupt the answer either: the new lock is applied
        // as T_map_at_lock * T_odom_at_lock^-1 * T_odom_now, all relative, so
        // however wrong the pose is when the fix arrives, it cancels.
        srv_relocalize_ = this->create_service<std_srvs::srv::Trigger>(
            "/relocalize",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                if (!map_loaded) {
                    res->success = false;
                    res->message = "prior map not loaded";
                    return;
                }
                {   // drop the odometry trail: it indexes scans from the OLD
                    // search, and the new lock will index into this vector.
                    std::lock_guard<std::mutex> lk(init_feats_mutex);
                    std::queue<std::pair<int, PointCloudXYZI::Ptr>> empty;
                    std::swap(init_feats_down_bodys, empty);
                    position_init.clear();
                    pose_init.clear();
                    init_count = 0;
                }
                {
                    std::lock_guard<std::mutex> lk(init_state_mutex);
                    global_localization_finish = false;   // re-arm the search
                }
                global_update = false;                    // allow a new teleport
                RCLCPP_WARN(this->get_logger(),
                    "/relocalize: searching again. The pose is NOT trustworthy "
                    "until the next 'Localized' line.");
                res->success = true;
                res->message = "Global search re-armed; watch the log for [init].";
            });

        // /initialpose -- seeded initialization, as an alternative to the
        // ScanContext search rather than a replacement for it. The search keeps
        // running until something locks; whichever arrives first wins.
        //
        // The pose is map <- base_footprint (that is what RViz's 2D Pose
        // Estimate publishes), while the filter state is map <- body, so the
        // static body->base extrinsic is composed out. It is applied through the
        // SAME path as a ScanContext lock -- init_result plus the teleport in
        // the scan pipeline -- so gravity and velocity are rotated correctly and
        // the prior map is swapped in exactly once.
        sub_initialpose_ = this->create_subscription<
            geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 1,
            [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                if (!map_loaded) {
                    RCLCPP_WARN(this->get_logger(),
                        "/initialpose ignored: prior map not loaded yet");
                    return;
                }
                if (!tf_child_resolved) {
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

        init_thread_ = std::thread(global_localization_thread, this->get_logger());

        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        {   // unblock the init thread if it is still searching
            std::lock_guard<std::mutex> lk(init_state_mutex);
            global_localization_finish = true;
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
            // is no map -> odom correction, which is the whole point of this node.
            //
            // The lock names the scan it was computed from, and odometry has run
            // on since, so carry it forward:
            //   T_map_now = T_map_at_lock * T_odom_at_lock^-1 * T_odom_now
            // A seed takes precedence and is applied directly: no odometry
            // carry-forward, so it is correct whether or not a lock exists.
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
                    // edit, so every world-frame quantity in the state has to
                    // rotate with it -- not only pos/rot.
                    //
                    // Upstream sets pos and rot alone. That is only safe when the
                    // map frame and the LIO start frame nearly coincide, which is
                    // true upstream and FALSE here: this map is levelled (the
                    // map <- pgo_init transform is ~90 deg off the LIO start
                    // attitude). Leaving grav behind therefore leaves gravity
                    // pointing sideways in the new frame, the IMU prediction is
                    // then wrong by ~1 g in the horizontal plane, and the filter
                    // diverges within a few scans. MEASURED before this fix: the
                    // first scans matched the prior map exactly (nearest
                    // neighbour distance 0.000), then z climbed to 96 m and every
                    // scan reported "No Effective Points".
                    const M3D R_old = state_point.rot.toRotationMatrix();
                    const M3D R_new = T_map_now.block<3,3>(0,0);
                    const M3D R_delta = R_new * R_old.transpose();

                    state_ikfom gs = state_point;
                    gs.pos  = T_map_now.block<3,1>(0,3);
                    gs.rot  = R_new;
                    gs.vel  = R_delta * state_point.vel;      // world-frame velocity
                    gs.grav = S2(V3D(R_delta * state_point.grav.vec));
                    // bg/ba are body-frame biases and offset_R/T_L_I is the
                    // lidar-IMU extrinsic; none of those are world-frame, so they
                    // carry over untouched.
                    kf.change_x(gs);
                    state_point = kf.get_x();

                    // Hand the filter the prior map -- but only the FIRST time.
                    // ikdtree_global is moved-from afterwards, and on a
                    // /relocalize the tree already holds the prior map, so a
                    // second swap would install an empty tree.
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
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
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
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
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
            // Only while still searching. After the lock the prior map is
            // READ-ONLY: adding live scans would let drift contaminate the very
            // thing being localized against.
            if (!global_update) {
                map_incremental();
            }
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            // Feed the init thread while still searching: the undistorted,
            // downsampled scan for ScanContext, and the odometry pose it was
            // taken at, so a lock found several scans later can be carried
            // forward to now. Indices into position_init/pose_init ARE the id
            // queued alongside the cloud, so the two cannot drift apart.
            if (!global_localization_finish)
            {
                PointCloudXYZI::Ptr snapshot(new PointCloudXYZI());
                pcl::copyPointCloud(*feats_down_body, *snapshot);
                {
                    // The trail and the queue are ONE unit: the id queued
                    // alongside a scan indexes into these vectors, so appending
                    // outside the lock let the search thread (and /relocalize's
                    // clear) race against the append.
                    std::lock_guard<std::mutex> lk(init_feats_mutex);
                    position_init.push_back(state_point.pos);
                    pose_init.push_back(state_point.rot);
                    // Bounded: ScanContext + two ICP passes is slower than the
                    // scan rate, so an unbounded queue would grow without limit
                    // and the thread would work on ever-staler scans.
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
                kdtree_size_end = ikdtree.size();
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
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
#ifdef HAVE_LIVOX
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
#endif

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::thread init_thread_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_relocalize_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
        sub_initialpose_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

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

    rclcpp::spin(std::make_shared<LaserMappingNode>());

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
