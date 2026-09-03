import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node


# FAST-LIO mapping, shared by every sensor config in this workspace.
#
# The odom -> base_footprint bridge is NOT started here: it was Pepper glue in a
# file serving mid360, velodyne, avia and ouster too. It lives in
# pepper_slam/launch/lio_odom_bridge.launch.py, which reads publish.body_frame
# from the config; include it alongside this file, as
# pepper_slam/launch/fastlio_odometry.launch.py does. lidar_imu_frame,
# bridge_level_frame, level_frame_as_child and flatten_base_frame belong to it.

def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_path = os.path.join(package_path, 'config')
    default_rviz_config_path = os.path.join(
        package_path, 'rviz', 'fastlio.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
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
        # Every LIO variant remaps its native topic to /odom_lio, so consumers
        # need not know which estimator is running.
        remappings=[('/Odometry', '/odom_lio')],
        output='screen'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        # Required: without it RViz runs on the WALL clock while the data
        # carries bag time, so its TF cache silently drops every cloud
        # ("Message Filter dropping message"). Only the view is affected.
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz_use)
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)

    return ld
