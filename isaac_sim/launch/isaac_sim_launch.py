import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import (
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for Isaac Sim simulation with OM1.

    Launches:
      1. Isaac Sim (run.py) in its own Python 3.11 venv
      2. Bridge nodes: go2_remapping, go2_sport, go2_lowstate (from go2_sim)
      3. Sensor nodes: om_path, obstacle detection, traversability (from OM1-ros2-sdk)

    Usage:
      ros2 launch isaac_sim isaac_sim_launch.py
      ros2 launch isaac_sim isaac_sim_launch.py robot_type:=g1
      ros2 launch isaac_sim isaac_sim_launch.py policy_dir:=/path/to/policy

    After launching, run Zenoh bridge and OM1 separately:
      zenoh-bridge-ros2dds -c zenoh/zenoh_bridge_config.json5
      cd ~/Documents/GitHub/OM1 && uv run src/run.py go2_sim_autonomy
    """

    # --- Paths ----------------------------------------------------------------

    # Resolve OM1-sim workspace root.
    # This launch file lives at:
    #   source:  <workspace>/isaac_sim/launch/isaac_sim_launch.py
    #   install: <workspace>/install/isaac_sim/share/isaac_sim/launch/isaac_sim_launch.py
    # We search upward for cyclonedds/ directory to find the workspace root.
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    om1_sim_dir = _this_dir
    for _ in range(6):
        om1_sim_dir = os.path.dirname(om1_sim_dir)
        if os.path.isdir(os.path.join(om1_sim_dir, "cyclonedds")):
            break

    isaac_sim_src = os.path.join(om1_sim_dir, "isaac_sim")
    isaac_sim_venv = os.path.join(isaac_sim_src, "env_isaacsim")
    isaac_sim_run_py = os.path.join(isaac_sim_src, "run.py")

    cyclonedds_xml = os.path.join(om1_sim_dir, "cyclonedds", "cyclonedds.xml")

    ros2_bridge_lib = os.path.join(
        isaac_sim_venv,
        "lib/python3.11/site-packages/isaacsim/exts/"
        "isaacsim.ros2.bridge/humble/lib",
    )

    # OM1-ros2-sdk (sibling directory)
    om1_ros2_sdk_dir = os.environ.get(
        "OM1_ROS2_SDK_DIR",
        os.path.join(os.path.dirname(om1_sim_dir), "OM1-ros2-sdk"),
    )
    om1_ros2_sdk_setup = os.path.join(om1_ros2_sdk_dir, "install", "setup.bash")
    has_ros2_sdk = os.path.isfile(om1_ros2_sdk_setup)

    # --- Launch Arguments -----------------------------------------------------

    robot_type = LaunchConfiguration("robot_type")
    policy_dir = LaunchConfiguration("policy_dir")
    launch_sensors = LaunchConfiguration("launch_sensors")

    declare_robot_type = DeclareLaunchArgument(
        "robot_type",
        default_value="go2",
        description="Robot type: go2 or g1",
    )
    declare_policy_dir = DeclareLaunchArgument(
        "policy_dir",
        default_value="",
        description="Path to policy directory (uses default if empty)",
    )
    declare_launch_sensors = DeclareLaunchArgument(
        "launch_sensors",
        default_value="true" if has_ros2_sdk else "false",
        description="Launch sensor nodes from OM1-ros2-sdk (om_path, etc.)",
    )

    # --- Environment ----------------------------------------------------------

    set_rmw = SetEnvironmentVariable(
        name="RMW_IMPLEMENTATION", value="rmw_cyclonedds_cpp"
    )
    set_cyclonedds = SetEnvironmentVariable(
        name="CYCLONEDDS_URI", value="file://" + cyclonedds_xml
    )

    # --- 1. Isaac Sim ---------------------------------------------------------

    # Isaac Sim must run inside its own Python 3.11 venv with special env vars.
    # We use ExecuteProcess with bash to activate the venv and run run.py.
    isaac_sim_cmd = [
        FindExecutable(name="bash"),
        "-c",
        [
            "source ", isaac_sim_venv, "/bin/activate && "
            "export ROS_DISTRO=humble && "
            "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
            "export CYCLONEDDS_URI=file://", cyclonedds_xml, " && "
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:", ros2_bridge_lib, " && "
            "cd ", isaac_sim_src, " && "
            "python3 run.py --robot_type ", robot_type,
        ],
    ]

    isaac_sim_process = ExecuteProcess(
        cmd=isaac_sim_cmd,
        name="isaac_sim",
        output="screen",
        shell=False,
    )

    # --- 2. Bridge Nodes (delayed to let Isaac Sim start) ---------------------

    go2_remapping_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package="go2_sim",
                executable="go2_remapping_node",
                name="go2_remapping_node",
                output="screen",
            ),
        ],
    )

    go2_sport_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package="go2_sim",
                executable="go2_sport_node",
                name="go2_sport_node",
                output="screen",
            ),
        ],
    )

    go2_lowstate_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package="go2_sim",
                executable="go2_lowstate_node",
                name="go2_lowstate_node",
                output="screen",
            ),
        ],
    )

    # --- 3. Sensor Nodes (from OM1-ros2-sdk, delayed further) -----------------

    # The sensor nodes (om_path, d435_obstacle_dector, local_traversability,
    # d435_isaac_sim_scaler) come from OM1-ros2-sdk's go2_sdk package.
    # We launch them via ExecuteProcess so we can source the correct workspace.

    sensor_entities = []
    if has_ros2_sdk:
        sensor_launch_cmd = [
            FindExecutable(name="bash"),
            "-c",
            [
                "source /opt/ros/humble/setup.bash && "
                "source ", om1_ros2_sdk_setup, " && "
                "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
                "export CYCLONEDDS_URI=file://", cyclonedds_xml, " && "
                "ros2 launch go2_sdk sensor_launch.py use_sim:=true",
            ],
        ]

        sensor_launch_process = TimerAction(
            period=20.0,
            actions=[
                ExecuteProcess(
                    cmd=sensor_launch_cmd,
                    name="sensor_nodes",
                    output="screen",
                    shell=False,
                ),
            ],
        )
        sensor_entities.append(sensor_launch_process)

    # --- Launch Description ---------------------------------------------------

    return LaunchDescription(
        [
            # Arguments
            declare_robot_type,
            declare_policy_dir,
            declare_launch_sensors,
            # Environment
            set_rmw,
            set_cyclonedds,
            # Isaac Sim
            isaac_sim_process,
            # Bridge nodes (after 15s delay for Isaac Sim startup)
            go2_remapping_node,
            go2_sport_node,
            go2_lowstate_node,
        ]
        + sensor_entities
    )
