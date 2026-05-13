"""
Bridge nodes launcher for use when Isaac Sim is already running.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Bridge + topic-relay nodes for an externally-started Isaac Sim."""
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Isaac Sim) clock if true",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch rviz2"
    )

    lowstate_node = Node(
        package="go2_gazebo_sim",
        executable="go2_lowstate_node",
        name="go2_lowstate_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

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
            lowstate_node,
            relay_color_image,
            relay_depth_image,
            relay_unitree_lidar,
            relay_go2_camera,
            rviz2,
        ]
    )
