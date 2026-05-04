"""
Bridge nodes launcher for use when Isaac Sim is already running.

Counterpart to isaac_sim_launch.py, which spawns Isaac Sim itself plus these
nodes. Use this one if you start run.py separately.

om_path is not launched here — it comes from OM1-ros2-sdk's sensor_launch.py
that the all-in-one launcher invokes. Uncomment the block below to run it
standalone.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Isaac Sim) clock if true",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch rviz2"
    )

    go2_sport_node = Node(
        package="go2_sim",
        executable="go2_sport_node",
        name="go2_sport_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    go2_remapping_node = Node(
        package="go2_sim",
        executable="go2_remapping_node",
        name="go2_remapping_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Use the Isaac-Sim-specific lowstate node, not gazebo's go2_lowstate_node.
    lowstate_node = Node(
        package="isaac_sim",
        executable="lowstate_node",
        name="go2_lowstate_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # om_path_node = Node(
    #     package="om_path",
    #     executable="om_path",
    #     name="om_path",
    #     output="screen",
    #     parameters=[
    #         {"use_sim_time": use_sim_time},
    #         {"use_sim": True},
    #     ],
    # )

    # Republish renamed topics so consumers using either naming work.
    relay_color_image = Node(
        package="topic_tools",
        executable="relay",
        name="relay_color_image",
        arguments=[
            "/camera/realsense2_camera_node/color/image_raw",
            "/camera/realsense2_camera_node/color/image_isaac_sim_raw",
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )
    relay_depth_image = Node(
        package="topic_tools",
        executable="relay",
        name="relay_depth_image",
        arguments=[
            "/camera/realsense2_camera_node/depth/image_rect_raw",
            "/camera/realsense2_camera_node/depth/image_rect_isaac_sim_raw",
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )
    relay_unitree_lidar = Node(
        package="topic_tools",
        executable="relay",
        name="relay_unitree_lidar",
        arguments=["/unitree_lidar/points", "/unitree_lidar"],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )
    relay_go2_camera = Node(
        package="topic_tools",
        executable="relay",
        name="relay_go2_camera",
        arguments=["/camera/image_raw", "/camera/go2/image_raw"],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_rviz,
            go2_sport_node,
            go2_remapping_node,
            lowstate_node,
            relay_color_image,
            relay_depth_image,
            relay_unitree_lidar,
            relay_go2_camera,
            rviz2,
        ]
    )
