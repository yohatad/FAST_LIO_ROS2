import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node


# The odom -> base_footprint bridge is NOT started here any more.
#
# It was Pepper-specific glue in a launch file shared by every FAST-LIO sensor
# config in this workspace (mid360, velodyne, avia, ouster...), and it duplicated
# point_lio's identical copy -- same six parameters, same /odom_lio -- each with
# its own body-frame resolver and hardcoded {config_file: frame} table.
#
# It now lives once, in pepper_slam/launch/lio_odom_bridge.launch.py, which reads
# publish.body_frame from the config instead of mirroring it. Include that
# alongside this file; pepper_slam/launch/fastlio_odometry.launch.py shows how.
#
# This file therefore no longer declares lidar_imu_frame, bridge_level_frame,
# level_frame_as_child or flatten_base_frame -- they belong to the bridge.

def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_path = os.path.join(package_path, 'config')
    default_rviz_config_path = os.path.join(
        package_path, 'rviz', 'fastlio.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    bridge_level_frame = LaunchConfiguration('bridge_level_frame')
    flatten_base_frame = LaunchConfiguration('flatten_base_frame')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        description='Config file (e.g. l2.yaml, mid360.yaml) -- required, no '
                     'default: this launch file serves many different '
                     'sensors, and silently defaulting to one would load the '
                     'wrong config for everyone else. The bridge in\n'
                     'pepper_slam reads publish.body_frame from whatever is\n'
                     'selected here, so this choice also fixes the TF frame\n'
                     'contract downstream.'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time}],
        # Standard odometry topic across every LIO variant: FAST-LIO publishes
        # /Odometry natively, Point-LIO and FAST-LIVO2 /aft_mapped_to_init.
        # Each mapping launch remaps its own to /odom_lio so consumers need not
        # know which estimator is running.
        remappings=[('/Odometry', '/odom_lio')],
        output='screen'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        # Without this RViz runs on the WALL clock while every message and
        # transform carries bag time. Its TF cache then holds ~10 s around
        # today, the clouds arrive stamped whenever the bag was recorded, and
        # it silently drops all of them:
        #   "Message Filter dropping message: ... the timestamp on the message
        #    is earlier than all the data in the transform cache"
        # Localization is unaffected -- only the view is. Every other node here
        # already gets use_sim_time; this one was missed.
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz_use)
    )

    # Publishes odom -> base_footprint from FAST-LIO's /Odometry message
    # (odom -> l2lidar_frame_imu, with FAST-LIO's own TF broadcast disabled),
    # closing the tree per REP-105 (odom -> base_footprint -> ... -> l2lidar_frame_imu).
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)

    # AFTER every DeclareLaunchArgument: the resolver reads config_file and
    # lidar_imu_frame, which do not exist in the context until their
    # declares have run.
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)

    return ld
