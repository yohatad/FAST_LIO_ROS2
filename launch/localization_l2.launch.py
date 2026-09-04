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
# map_frame defaults to 'map', not stock FAST-LIO's 'camera_init'. That is not
# cosmetic: after the handover the filter state IS in the prior map's frame, so
# 'camera_init' (the LIO's own start frame) would name it wrongly -- the same
# misnomer FRAMES.md warns about elsewhere in this workspace.
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
        DeclareLaunchArgument('map_dir', default_value='',
            description='REQUIRED. Directory holding pose.json and pcd/, from '
                        'utils/pgo_to_scancontext_map.py.'),
        DeclareLaunchArgument('config_file', default_value='l2_rsimu.yaml',
            description='FAST-LIO config. Must be the SAME one the map was '
                        'built with -- the lidar/IMU extrinsic enters the '
                        'initial pose composition.'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
            description='true for bag replay (this launch is bag-oriented).'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        # l2_rsimu.yaml sets publish_tf:=false, because in the MAPPING stack
        # lio_odom_bridge owns map/odom/base_footprint and a second broadcaster
        # would fight it. Standalone, this node is the only authority on
        # map -> body, so nothing publishes the 'map' frame unless it does --
        # RViz then reports "frame [map] does not exist" and shows nothing.
        # Set false again if you later run this alongside lio_odom_bridge.
        # REP-105: map -> base_footprint, not map -> the IMU on the mast. The
        # bag's /tf_static already owns base_footprint -> camera_imu_optical_-
        # frame, so broadcasting the IMU edge here as well would give it two
        # parents and split the TF tree.
        DeclareLaunchArgument('tf_child_frame', default_value='base_footprint',
            description='Child of the broadcast map edge. The body->child '
                        'extrinsic is read from /tf_static once and cached.'),
        DeclareLaunchArgument('publish_tf', default_value='true',
            description='Broadcast map -> body. Required standalone; turn off '
                        'if another node owns those frames.'),
        # ScanContext descriptor geometry. Defaults are sized to the L2, whose
        # keyframes hold ~1600 pts with 90% inside 3.3 m and only 0.08% beyond
        # 10 m -- upstream's 80 m / 20x60 leaves most of the descriptor empty.
        DeclareLaunchArgument('sc_max_radius', default_value='10.0'),
        DeclareLaunchArgument('sc_num_ring', default_value='12'),
        DeclareLaunchArgument('sc_num_sector', default_value='40'),
        DeclareLaunchArgument('sc_lidar_height', default_value='0.5'),
        DeclareLaunchArgument('sc_dist_thres', default_value='0.15'),
        # A single ScanContext hit in a corridor is not evidence. Require this
        # many independent locks agreeing within init_agree_dist.
        DeclareLaunchArgument('init_agree_count', default_value='2'),
        DeclareLaunchArgument('init_agree_dist', default_value='2.0'),
        # Agreement alone cannot catch a wrong lock: two matches to the SAME
        # wrong place agree perfectly. MEASURED starting mid-corridor, it locked
        # 41 m from truth on two mutually-consistent matches. These ask the map
        # instead -- what fraction of the scan lands on it at the proposed pose.
        DeclareLaunchArgument('init_min_overlap', default_value='0.70'),
        DeclareLaunchArgument('init_overlap_dist', default_value='0.20',
            description='Metres. Keep this TIGHT: at 1.0 m a pose 41 m out still '
                        'scored 97% against this dense map.'),
    ]

    node = Node(
        package='fast_lio', executable='fastlio_localization',
        name='fast_lio_localization', output='screen',
        parameters=[
            os.path.join(share, 'config', 'l2_rsimu.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'publish.map_frame': LaunchConfiguration('map_frame'),
             # ParameterValue with an explicit type: the node declares this as a
             # bool, and a bare LaunchConfiguration arrives as the STRING
             # 'true', which ROS 2 rejects on type -- silently leaving
             # publish_tf at the config's false, so no 'map' frame ever appears.
             'publish.publish_tf': ParameterValue(
                 LaunchConfiguration('publish_tf'), value_type=bool),
             'publish.tf_child_frame': LaunchConfiguration('tf_child_frame'),
             'localization.map_dir': LaunchConfiguration('map_dir'),
             'localization.sc_max_radius': LaunchConfiguration('sc_max_radius'),
             'localization.sc_num_ring': LaunchConfiguration('sc_num_ring'),
             'localization.sc_num_sector': LaunchConfiguration('sc_num_sector'),
             'localization.sc_lidar_height': LaunchConfiguration('sc_lidar_height'),
             'localization.sc_dist_thres': LaunchConfiguration('sc_dist_thres'),
             'localization.init_agree_count': LaunchConfiguration('init_agree_count'),
             'localization.init_agree_dist': LaunchConfiguration('init_agree_dist'),
             'localization.init_min_overlap': LaunchConfiguration('init_min_overlap'),
             'localization.init_overlap_dist': LaunchConfiguration('init_overlap_dist')},
        ])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['-d', os.path.join(share, 'rviz', 'fastlio_localization.rviz')])

    return LaunchDescription(args + [node, rviz])
