# OM1-sim

Gazebo simulation package for the Unitree Go2 robot, designed to work with [OM1](https://github.com/OpenMind/OM1).

This repo provides a minimal simulation environment with lidar-based obstacle avoidance — no SLAM, Nav2, or mapping stack required.

## Architecture

```
OM1 (Python/LLM) <--> Zenoh <--> zenoh-bridge-ros2dds <--> ROS2 (OM1-sim) <--> Gazebo
```

### What's included

| Component | Purpose |
|-----------|---------|
| `go2_sim` | Main simulation package: launch file, topic remapping, sport API bridge, battery mock |
| `om_path` | Lidar-based path feasibility node — publishes safe movement directions to `/om/paths` |
| `go2_description` | Robot URDF, meshes, and Gazebo worlds |
| `champ` / `champ_base` | Quadruped locomotion controller (CHAMP framework) |
| `champ_msgs` | CHAMP custom message definitions |
| `unitree_api` / `unitree_go` | Unitree message definitions |
| `om_api` | OpenMind message definitions (Paths, etc.) |

### What's NOT included (handled by OM1-ros2-sdk for real robots)

- SLAM (`slam_toolbox`)
- Nav2 navigation stack
- Frontier exploration
- Orchestrator / REST API
- Localization (AMCL, scan matching)
- D435 depth camera processing
- Watchdog

## Prerequisites

- ROS 2 Humble
- Gazebo (Harmonic or Fortress)
- [OM1](https://github.com/OpenMind/OM1) for the AI agent
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
# Install ROS 2 dependencies
sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz-bridge \
  ros-humble-gz-ros2-control ros-humble-ros2-control ros-humble-ros2-controllers \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-xacro ros-humble-joy ros-humble-teleop-twist-joy \
  ros-humble-rmw-cyclonedds-cpp

# Install zenoh bridge
echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee /etc/apt/sources.list.d/zenoh.list
sudo apt update && sudo apt install zenoh-bridge-ros2dds

# Clone and build
git clone https://github.com/OpenMind/OM1-sim.git
cd OM1-sim
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Running

### 1. Start the Zenoh bridge

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
zenoh-bridge-ros2dds -c ./zenoh/zenoh_bridge_config.json5
```

### 2. Launch the Go2 simulation (in a separate terminal)

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch go2_sim go2_launch.py
```

### 3. Start OM1 (in a separate terminal)

```bash
# In the OM1 repo
cd ../OM1
uv run src/run.py go2_sim_autonomy
```

## Configuration

The corresponding OM1 config is `OM1/config/go2_sim_autonomy.json5`. It provides:

- **Autonomy mode**: Move around with lidar-based obstacle avoidance
- **Conversation mode**: Chat with the robot
- **Guard mode**: Autonomous patrol with status reports

## Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim_time` | `true` | Use Gazebo clock |
| `rviz` | `true` | Launch RViz |
| `world` | `home_world.sdf` | Gazebo world file |
| `robot_name` | `go2` | Robot name in Gazebo |
| `world_init_x/y/z` | `0/0/0.375` | Initial robot position |
| `publish_map_tf` | `true` | Publish static map→odom TF |

## Data Flow

```
Gazebo /scan (LaserScan)
    --> om_path node --> /om/paths
        --> Zenoh bridge --> om/paths (Zenoh topic)
            --> OM1 SimplePathsProvider
                --> LLM decides direction
                    --> SportClient --> /api/sport/request
                        --> go2_sport_node --> /cmd_vel --> Gazebo
```

## License

MIT
