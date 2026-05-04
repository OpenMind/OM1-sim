"""
ROS2 helper nodes for the Isaac Sim flow.

Run this AFTER starting Isaac Sim via `python isaac_sim/run.py ...` in a separate
terminal. Isaac Sim publishes /cmd_vel (sub), /scan, /odom, /tf, /tf_static,
/clock, /joint_states, /imu/data, /unitree_lidar/points, the camera topics, and
auto-spawns lowstate_node.py (which publishes /lowstate, /lf/lowstate,
/utlidar/robot_pose, /utlidar/cloud_deskewed).

This launch file starts the OM1-sim ROS2 nodes that bridge Isaac Sim to OM1:
  - go2_sport_node:    /api/sport/request -> /cmd_vel
  - go2_remapping_node: /odom -> /utlidar/robot_pose with charging-aware z-offset
                       /unitree_lidar/points -> /utlidar/cloud_deskewed
  - om_path_node:       /scan -> /om/paths*

go2_lowstate_node is intentionally NOT started here: Isaac Sim's run.py auto-
spawns its own lowstate_node.py subprocess, so launching go2_lowstate_node
here would create duplicate publishers on /lowstate and /lf/lowstate.
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

    om_path_node = Node(
        package="om_path",
        executable="om_path",
        name="om_path",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"use_sim": True},
        ],
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
            om_path_node,
            rviz2,
        ]
    )
