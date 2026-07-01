import os
import subprocess

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import FindExecutable, LaunchConfiguration


def _detect_cyclonedds_interface() -> str:
    """Pick a CycloneDDS network interface.

    CycloneDDS can't bind to `lo` for typical multi-process discovery (lo is
    not multicast-capable), so fall back to the first UP non-loopback
    interface. Honor `CYCLONEDDS_INTERFACE` if the user already set one.
    """
    explicit = os.environ.get("CYCLONEDDS_INTERFACE")
    if explicit:
        return explicit
    try:
        out = subprocess.check_output(
            ["ip", "-br", "link"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return "lo"
    candidates = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        if name == "lo" or state != "UP":
            continue
        if name.startswith(("docker", "br-", "veth")):
            continue
        candidates.append(name)
    for prefix in ("en", "eth", "wl"):
        for c in candidates:
            if c.startswith(prefix):
                return c
    return candidates[0] if candidates else "lo"


def generate_launch_description():
    """
    Launch file for Isaac Sim simulation with OM1 (OM1-ros2-sdk layout).

    Launches:
      1. Isaac Sim (run.py) in its own Python venv
      2. Sensor + bridge nodes: go2_sdk/sensor_launch.py use_sim:=true
         (which brings up go2_lowstate_node from go2_gazebo_sim)

    The sensor nodes require the sibling OM1-ros2-sdk workspace (it provides
    the go2_sdk package). This is auto-detected at a directory named
    "OM1-ros2-sdk" next to this workspace, or via the OM1_ROS2_SDK_WS env
    var; its install/setup.bash is sourced automatically before launching
    sensor nodes, so you don't need to source it yourself.

    Usage:
      ros2 launch isaac_sim isaac_sim_launch.py
      ros2 launch isaac_sim isaac_sim_launch.py robot_type:=g1
      ros2 launch isaac_sim isaac_sim_launch.py isaac_sim_venv:=/path/to/venv
    """
    # --- Paths ----------------------------------------------------------------

    # Resolve OM1-sim workspace root (this repo).
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = _this_dir
    for _ in range(8):
        workspace_root = os.path.dirname(workspace_root)
        if os.path.isdir(os.path.join(workspace_root, "cyclonedds")):
            break

    isaac_sim_src = os.path.join(workspace_root, "isaac_sim") # "unitree" removed
    cyclonedds_xml = os.path.join(workspace_root, "cyclonedds", "cyclonedds.xml")
    om1_sim_setup = os.path.join(workspace_root, "install", "setup.bash")

    # Resolve the sibling OM1-ros2-sdk workspace, which provides go2_sdk and
    # the sensor/path nodes it depends on (go2_gazebo_sim, om_path, etc.).
    # Honor OM1_ROS2_SDK_WS if set, else look for a directory named
    # "OM1-ros2-sdk" next to this workspace.
    _ros2_sdk_candidates = [
        os.environ.get("OM1_ROS2_SDK_WS", ""),
        os.path.join(os.path.dirname(workspace_root), "OM1-ros2-sdk"),
    ]
    _ros2_sdk_ws = next(
        (
            p
            for p in _ros2_sdk_candidates
            if p and os.path.isfile(os.path.join(p, "install", "setup.bash"))
        ),
        "",
    )
    ros2_sdk_setup = (
        os.path.join(_ros2_sdk_ws, "install", "setup.bash") if _ros2_sdk_ws else ""
    )

    # Venv location: prefer ISAAC_SIM_VENV env var, then ~/env_isaacsim, then
    # <isaac_sim_src>/env_isaacsim.
    default_venv_candidates = [
        os.environ.get("ISAAC_SIM_VENV", ""),
        os.path.expanduser("~/env_isaacsim"),
        os.path.join(isaac_sim_src, "env_isaacsim"),
    ]
    default_venv = next(
        (p for p in default_venv_candidates if p and os.path.isdir(p)), ""
    )

    # --- Launch Arguments -----------------------------------------------------

    robot_type = LaunchConfiguration("robot_type")
    isaac_sim_venv = LaunchConfiguration("isaac_sim_venv")
    launch_sensors = LaunchConfiguration("launch_sensors")
    human = LaunchConfiguration("human")
    human_auto_walk = LaunchConfiguration("human_auto_walk")

    declare_robot_type = DeclareLaunchArgument(
        "robot_type",
        default_value="go2",
        description="Robot type: go2, g1, or tron1",
    )
    declare_human = DeclareLaunchArgument(
        "human",
        default_value="false",
        description="Spawn a human pedestrian model (true/false)",
    )
    declare_human_auto_walk = DeclareLaunchArgument(
        "human_auto_walk",
        default_value="false",
        description="Human walks autonomously in a patrol pattern (true/false)",
    )
    declare_policy_dir = DeclareLaunchArgument(
        "policy_dir",
        default_value="",
        description="Path to policy directory (uses default if empty)",
    )
    declare_venv = DeclareLaunchArgument(
        "isaac_sim_venv",
        default_value=default_venv,
        description="Path to Isaac Sim Python venv (must have run.py-compatible isaacsim install)",
    )
    declare_launch_sensors = DeclareLaunchArgument(
        "launch_sensors",
        default_value="true" if ros2_sdk_setup else "false",
        description="Launch sensor nodes from go2_sdk (om_path, obstacle detector, etc.)",
    )
    declare_cyclonedds_iface = DeclareLaunchArgument(
        "cyclonedds_interface",
        default_value=_detect_cyclonedds_interface(),
        description="Network interface for CycloneDDS (e.g. eth0, wlan0)",
    )
    cyclonedds_iface = LaunchConfiguration("cyclonedds_interface")

    # --- Environment ----------------------------------------------------------

    set_rmw = SetEnvironmentVariable(
        name="RMW_IMPLEMENTATION", value="rmw_cyclonedds_cpp"
    )
    set_cyclonedds = SetEnvironmentVariable(
        name="CYCLONEDDS_URI", value="file://" + cyclonedds_xml
    )
    set_cyclonedds_iface = SetEnvironmentVariable(
        name="CYCLONEDDS_INTERFACE", value=cyclonedds_iface
    )

    # --- 1. Isaac Sim ---------------------------------------------------------

    # Isaac Sim must run inside its own Python venv with the isaacsim package.
    # ros2_bridge_lib is added to LD_LIBRARY_PATH so the in-process ROS2 bridge
    # extension can resolve its shared libs.
    isaac_sim_cmd = [
        FindExecutable(name="bash"),
        "-c",
        [
            "source ",
            isaac_sim_venv,
            "/bin/activate && "
            "export ROS_DISTRO=humble && "
            "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
            "export CYCLONEDDS_INTERFACE=",
            cyclonedds_iface,
            " && export CYCLONEDDS_URI=file://",
            cyclonedds_xml,
            " && PY_VER=$(python3 -c \"import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')\") && BRIDGE_DIR=",
            isaac_sim_venv,
            "/lib/$PY_VER/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble"
            " && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BRIDGE_DIR/lib"
            " && export PYTHONPATH=$BRIDGE_DIR/rclpy:$PYTHONPATH"
            " && cd ",
            isaac_sim_src,
            " && python3 run.py --robot_type ",
            robot_type,
            " $([ '",
            human,
            "' = 'true' ] && echo '--human' || echo '')",
            " $([ '",
            human_auto_walk,
            "' = 'true' ] && echo '--human_auto_walk' || echo '')",
        ],
    ]

    isaac_sim_process = ExecuteProcess(
        cmd=isaac_sim_cmd,
        name="isaac_sim",
        output="screen",
        shell=False,
    )

    # --- 2. Sensor Nodes (from go2_sdk, delayed to let Isaac Sim start) -------
    sensor_entities = []
    if ros2_sdk_setup:
        sensor_launch_cmd = [
            FindExecutable(name="bash"),
            "-c",
            [
                "export PATH=/usr/bin:$PATH && "
                "source /opt/ros/humble/setup.bash && source ",
                ros2_sdk_setup,
                " && source ",
                om1_sim_setup,
                " && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
                " && export CYCLONEDDS_INTERFACE=",
                cyclonedds_iface,
                " && export CYCLONEDDS_URI=file://",
                cyclonedds_xml,
                " && ros2 launch go2_sdk sensor_launch.py use_sim:=true",
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
            condition=IfCondition(launch_sensors),
        )
        sensor_entities.append(sensor_launch_process)

    # --- Launch Description ---------------------------------------------------

    return LaunchDescription(
        [
            declare_robot_type,
            declare_human,
            declare_human_auto_walk,
            declare_policy_dir,
            declare_venv,
            declare_launch_sensors,
            declare_cyclonedds_iface,
            set_rmw,
            set_cyclonedds,
            set_cyclonedds_iface,
            isaac_sim_process,
        ]
        + sensor_entities
    )
