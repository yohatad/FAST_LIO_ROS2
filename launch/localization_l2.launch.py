# FAST-LOCALIZATION on the L2 rig: localize against a prior map built by
# fastlio_lc_pgo and converted with utils/pgo_to_scancontext_map.py.
#
#   ros2 launch fast_lio localization_l2.launch.py \
#       map_dir:=<dir holding pose.json and pcd/>
#
# then, for a bag:
#   ros2 bag play <bag> --clock \
#       --qos-profile-overrides-path config/play_qos.yaml \
#       --read-ahead-queue-size 2000
#
# The QoS overrides are REQUIRED: /imu/data and /camera/imu were recorded
# BEST_EFFORT and a RELIABLE subscriber matches nothing against them.
#
# map_frame defaults to 'map', not stock FAST-LIO's 'camera_init': after the
# handover the filter state IS in the prior map's frame.
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('fast_lio')

    args = [
        # Split deliberately: pose.json is small and git-tracked; the keyframe
        # clouds are 75 MB of gitignored binary.
        DeclareLaunchArgument('map_dir',
            default_value=os.path.join(
                get_package_share_directory('pepper_navigation'), 'pcd'),
            description='Directory holding the pose file.'),
        DeclareLaunchArgument('map_pose_file', default_value='sc_pose_20260823.json',
            description='Pose file within map_dir (or an absolute path). Named '
                        'per run: this directory also holds the OTHER stack\'s '
                        'pepper_map_lc_poses.txt, and a second map would drop a '
                        'second pose file beside it.'),
        # Named after the RUN, not a bare pcd/: pose.json indexes these by
        # number, so two undifferentiated folders would be silently
        # interchangeable.
        DeclareLaunchArgument('map_scan_dir',
            default_value=os.path.join(
                get_package_share_directory('pepper_navigation'),
                'pcd', 'sc_pcd_20260823'),
            description='Directory holding the per-keyframe <N>.pcd clouds. '
                        'Both are written by utils/pgo_to_scancontext_map.py.'),
        DeclareLaunchArgument('config_file', default_value='l2_rsimu.yaml',
            description='FAST-LIO config. Must be the SAME one the map was '
                        'built with -- the lidar/IMU extrinsic enters the '
                        'initial pose composition.'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
            description='true for bag replay (this launch is bag-oriented).'),
        DeclareLaunchArgument('rviz', default_value='true',
            description='Open RViz2 pre-configured for this stack.'),
        DeclareLaunchArgument('map_frame', default_value='map',
            description='Frame name for the published map -> body edge.'),
        # Standalone, this node is the only authority on map -> body, so
        # publish_tf defaults true here. Set false if running alongside
        # lio_odom_bridge, which owns those frames in the mapping stack.
        DeclareLaunchArgument('tf_child_frame', default_value='base_footprint',
            description='Child of the broadcast map edge. The body->child '
                        'extrinsic is read from /tf_static once and cached.'),
        DeclareLaunchArgument('publish_tf', default_value='true',
            description='Broadcast map -> body. Required standalone; turn off '
                        'if another node owns those frames.'),
        # ScanContext descriptor geometry. Defaults are sized to the L2, whose
        # keyframes hold ~1600 pts with 90% inside 3.3 m and only 0.08% beyond
        # 10 m -- upstream's 80 m / 20x60 leaves most of the descriptor empty.
        DeclareLaunchArgument('sc_max_radius', default_value='10.0',
            description='ScanContext descriptor radius, metres.'),
        DeclareLaunchArgument('sc_num_ring', default_value='12',
            description='ScanContext descriptor ring count.'),
        DeclareLaunchArgument('sc_num_sector', default_value='40',
            description='ScanContext descriptor sector count.'),
        DeclareLaunchArgument('sc_lidar_height', default_value='0.5',
            description='Lidar height above ground, metres.'),
        DeclareLaunchArgument('sc_dist_thres', default_value='0.15',
            description='ScanContext match distance threshold.'),
        # A single ScanContext hit in a corridor is not evidence. Require this
        # many independent locks agreeing within init_agree_dist.
        DeclareLaunchArgument('init_agree_count', default_value='2',
            description='Independent ScanContext locks required to agree.'),
        DeclareLaunchArgument('init_agree_dist', default_value='2.0',
            description='Metres within which agreeing locks must match.'),
        # Agreement alone can't catch two matches that agree on the SAME wrong
        # place (MEASURED: 41 m off in a corridor). require_motion forces the
        # two estimates apart in space so agreement means something; off by
        # default since a seeded /initialpose start doesn't need it. Turn on
        # for unattended startup with no seed.
        DeclareLaunchArgument('init_require_motion', default_value='false',
            description='Require motion between agreeing estimates before '
                        'accepting a lock (guards against vacuous standstill '
                        'agreement).'),
        DeclareLaunchArgument('init_motion_min', default_value='0.50',
            description='Metres of odometry required between the agreeing '
                        'estimates. Only used when init_require_motion.'),
        DeclareLaunchArgument('init_min_overlap', default_value='0.70',
            description='Minimum fraction of the scan that must overlap the '
                        'map at the proposed pose.'),
        DeclareLaunchArgument('init_overlap_dist', default_value='0.20',
            description='Metres. Keep TIGHT -- looser values let a wrong lock '
                        'still score a high overlap.'),
    ]

    node = Node(
        package='fast_lio', executable='fastlio_localization',
        name='fast_lio_localization', output='screen',
        parameters=[
            os.path.join(share, 'config', 'l2_rsimu.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'publish.map_frame': LaunchConfiguration('map_frame'),
             # Explicit type: a bare LaunchConfiguration arrives as a string,
             # which the node's bool parameter rejects.
             'publish.publish_tf': ParameterValue(
                 LaunchConfiguration('publish_tf'), value_type=bool),
             'publish.tf_child_frame': LaunchConfiguration('tf_child_frame'),
             'localization.map_dir': LaunchConfiguration('map_dir'),
             'localization.map_scan_dir': LaunchConfiguration('map_scan_dir'),
             'localization.map_pose_file': LaunchConfiguration('map_pose_file'),
             'localization.sc_max_radius': LaunchConfiguration('sc_max_radius'),
             'localization.sc_num_ring': LaunchConfiguration('sc_num_ring'),
             'localization.sc_num_sector': LaunchConfiguration('sc_num_sector'),
             'localization.sc_lidar_height': LaunchConfiguration('sc_lidar_height'),
             'localization.sc_dist_thres': LaunchConfiguration('sc_dist_thres'),
             'localization.init_agree_count': LaunchConfiguration('init_agree_count'),
             'localization.init_agree_dist': LaunchConfiguration('init_agree_dist'),
             'localization.init_require_motion': ParameterValue(
                 LaunchConfiguration('init_require_motion'), value_type=bool),
             'localization.init_motion_min': LaunchConfiguration('init_motion_min'),
             'localization.init_min_overlap': LaunchConfiguration('init_min_overlap'),
             'localization.init_overlap_dist': LaunchConfiguration('init_overlap_dist')},
        ])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['-d', os.path.join(share, 'rviz', 'fastlio_localization.rviz')])

    return LaunchDescription(args + [node, rviz])
