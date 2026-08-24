import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node


# Which static frame FAST-LIO's publish.body_frame corresponds to is a PROPERTY
# OF THE CONFIG, not an independent choice: l2.yaml estimates the L2's IMU,
# l2_rsimu.yaml estimates the RealSense's. lio_map_odom_bridge uses it to close
# odom -> base_footprint, and a mismatch is silent -- the tree still resolves,
# it is just wrong. So derive it, and let an explicit lidar_imu_frame override
# for configs this table does not know.

def _resolve_lidar_imu_frame(context, *args, **kwargs):
    """Which frame FAST-LIO's pose estimate refers to.

    lio_map_odom_bridge needs this to look up base_footprint -> <frame> and
    close odom -> base_footprint. It MUST equal the estimator's own
    publish.body_frame, or the bridge composes through the wrong rigid offset
    and produces a plausible-looking pose that is simply wrong.

    So read it from the config rather than keeping a second copy. This used to
    be a hardcoded {config_file: frame} table in this file -- a hand-maintained
    mirror of the yaml that had to be edited every time a config was added, and
    that could silently disagree with the file it mirrored. The yaml is the one
    source of truth; the node reads publish.body_frame from it too.
    """
    from launch.actions import SetLaunchConfiguration
    explicit = LaunchConfiguration('lidar_imu_frame').perform(context)
    if explicit:
        return [SetLaunchConfiguration('resolved_lidar_imu_frame', explicit)]

    import yaml
    path = os.path.join(LaunchConfiguration('config_path').perform(context),
                        os.path.basename(
                            LaunchConfiguration('config_file').perform(context)))
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise RuntimeError(
            f"cannot read FAST-LIO config '{path}' to determine the body frame "
            f"({exc}). Pass lidar_imu_frame:=<frame> explicitly.")

    frame = None
    for top in doc.values():
        if isinstance(top, dict):
            params = top.get('ros__parameters', top)
            if isinstance(params, dict):
                pub = params.get('publish')
                if isinstance(pub, dict) and pub.get('body_frame'):
                    frame = pub['body_frame']
                    break

    # Matches LaserMappingNode's own declare_parameter default, so a config that
    # omits publish.body_frame still lines up with what the node will stamp.
    return [SetLaunchConfiguration('resolved_lidar_imu_frame', frame or 'body')]


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
    declare_bridge_level_frame_cmd = DeclareLaunchArgument(
        'bridge_level_frame', default_value='true',
        description='Have lio_map_odom_bridge publish the static odom -> lio_init '
                    'leveling frame. Set false when a higher layer owns odom '
                    '(e.g. PGO publishing map -> odom), so odom keeps one parent.'
    )
    declare_level_frame_as_child_cmd = DeclareLaunchArgument(
        'level_frame_as_child', default_value='false',
        description='Publish the leveling transform as lio_init -> odom '
                    '(child) instead of odom -> lio_init (parent). Use with '
                    'bridge_level_frame:=true when a localizer already owns '
                    'map -> odom, so odom still exists without giving '
                    'odom two parents.'
    )
    declare_flatten_base_frame_cmd = DeclareLaunchArgument(
        'flatten_base_frame', default_value='false',
        description='Zero the leveled z/roll/pitch of odom -> base_footprint '
                    'every cycle (keep x, y, yaw) -- a hard flat-floor '
                    'assumption, not a sensor-fused correction. Off by '
                    'default: only correct on robots that are always on '
                    'genuinely flat floor. Requires bridge_level_frame: true.'
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
                     'wrong config (and the wrong TF frame contract via '
                     'lio_map_odom_bridge) for everyone else.'
    )
    # DERIVED from config_file when left empty -- see _resolve_lidar_imu_frame.
    # It used to default to 'l2lidar_frame_imu' unconditionally, which made the
    # RealSense-IMU config a two-argument change that five callers got wrong by
    # omission: they passed config_file and inherited the L2 frame, silently
    # producing a wrong odom -> base_footprint with no error anywhere.
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='',
        description='Static-tree frame that FAST-LIO\'s publish.body_frame '
                    'corresponds to, used by lio_map_odom_bridge to close '
                    'odom -> base_footprint. MUST match the config: l2.yaml '
                    'uses l2lidar_frame_imu, l2_rsimu.yaml uses '
                    'camera_imu_optical_frame. A mismatch silently yields a '
                    'wrong odom -> base_footprint.'
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
    lio_map_odom_bridge = Node(
        package='fast_lio',
        executable='lio_map_odom_bridge.py',
        name='lio_map_odom_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_level_frame': bridge_level_frame,
            'level_frame_as_child': LaunchConfiguration('level_frame_as_child'),
            'flatten_base_frame': flatten_base_frame,
            'lidar_imu_frame': LaunchConfiguration('resolved_lidar_imu_frame'),
            'odom_topic': '/odom_lio',
        }]
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    ld.add_action(declare_level_frame_as_child_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_lidar_imu_frame_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)

    # AFTER every DeclareLaunchArgument: the resolver reads config_file and
    # lidar_imu_frame, which do not exist in the context until their
    # declares have run.
    ld.add_action(OpaqueFunction(function=_resolve_lidar_imu_frame))
    ld.add_action(fast_lio_node)
    ld.add_action(lio_map_odom_bridge)
    ld.add_action(rviz_node)

    return ld
