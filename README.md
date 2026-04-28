# OM1-sim

Simulation packages for Unitree robots, designed to work with [OM1](https://github.com/OpenMind/OM1).

This repo provides two simulation environments:

| Simulator | Directory | Robots | Description |
|-----------|-----------|--------|-------------|
| **Gazebo** | `gazebo_sim/` | Go2 | Lightweight sim with lidar-based obstacle avoidance |
| **Isaac Sim** | `isaac_sim/` | Go2, G1 | High-fidelity NVIDIA sim with cameras, LiDAR, IMU |

## Architecture

```
OM1 (Python/LLM) <--> Zenoh <--> zenoh-bridge-ros2dds <--> ROS2 <--> Simulator
```

---

## Gazebo Simulation

### What's included

| Component | Purpose |
|-----------|---------|
| `gazebo_sim/go2_sim` | Launch file, topic remapping, sport API bridge, battery mock |
| `gazebo_sim/om_path` | Lidar-based path feasibility — publishes to `/om/paths` |
| `gazebo_sim/go2_description` | Robot URDF, meshes, and Gazebo worlds |
| `gazebo_sim/champ` / `champ_base` | Quadruped locomotion controller (CHAMP framework) |
| `gazebo_sim/champ_msgs` | CHAMP custom message definitions |
| `unitree_api` / `unitree_go` | Unitree message definitions (shared) |
| `om_api` | OpenMind message definitions (shared) |

### Prerequisites

- ROS 2 Humble
- Gazebo (Harmonic or Fortress)

### Setup (Native)

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

# Build
source /opt/ros/humble/setup.bash
cd OM1-sim
colcon build --symlink-install
source install/setup.bash
```

### Running Gazebo

**Terminal 1 — Simulation:**
```bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch go2_sim go2_launch.py
```

**Terminal 2 — Zenoh bridge:**
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
zenoh-bridge-ros2dds -c zenoh/zenoh_bridge_config.json5
```

**Terminal 3 — OM1:**
```bash
cd ../OM1
uv run src/run.py go2_sim_autonomy
```

### Gazebo Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim_time` | `true` | Use Gazebo clock |
| `rviz` | `true` | Launch RViz |
| `world` | `home_world.sdf` | Gazebo world file |
| `robot_name` | `go2` | Robot name in Gazebo |
| `world_init_x/y/z` | `0/0/0.375` | Initial robot position |
| `publish_map_tf` | `true` | Publish static map→odom TF |

---

## Isaac Sim Simulation

### Features

- **Isaac Sim**: Realistic physics simulation of Unitree Go2 and G1
- **Cameras**: RealSense depth + RGB, Go2 front camera
- **LiDAR**: Unitree L1 and Velodyne VLP-16 (2D scan)
- **IMU**: Simulated IMU sensor
- **ROS 2 Integration**: Full topic publishing (images, point clouds, odometry, TF, joint states)

### Prerequisites

- NVIDIA GPU with RTX support
- ROS 2 Humble
- CycloneDDS (`sudo apt install ros-humble-rmw-cyclonedds-cpp`)

### Setup

**Step 1: Build the ROS 2 workspace** (if not already done for Gazebo):

```bash
source /opt/ros/humble/setup.bash
cd OM1-sim
rosdep update
rosdep install --from-paths . --ignore-src -r -y
colcon build
```

**Step 2: Install Isaac Sim** (in a separate Python 3.11 venv):

```bash
cd OM1-sim/isaac_sim

uv venv --python 3.11 --seed env_isaacsim
source env_isaacsim/bin/activate

# Install IsaacSim 5.1
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install CUDA-enabled PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Verify installation
isaacsim
```

### Setup OM1-ros2-sdk (required for sensor/path nodes)

The Isaac Sim launch file automatically starts sensor nodes (om_path, obstacle detection, traversability) from [OM1-ros2-sdk](https://github.com/OpenMind/OM1-ros2-sdk). Build it as a sibling directory:

```bash
cd ~/Documents/GitHub
git clone https://github.com/OpenMind/OM1-ros2-sdk.git
cd OM1-ros2-sdk
source /opt/ros/humble/setup.bash
colcon build --packages-up-to go2_sdk --packages-ignore lto-test
```

### Running Isaac Sim

**Terminal 1 — Simulation:**

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Go2 (default) — launches Isaac Sim + bridge nodes + sensor nodes
ros2 launch isaac_sim isaac_sim_launch.py

# G1 humanoid
ros2 launch isaac_sim isaac_sim_launch.py robot_type:=g1

# Without sensor nodes (if OM1-ros2-sdk is not built)
ros2 launch isaac_sim isaac_sim_launch.py launch_sensors:=false
```

**Terminal 2 — Zenoh bridge:**

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
zenoh-bridge-ros2dds -c zenoh/zenoh_bridge_config.json5
```

**Terminal 3 — OM1:**

```bash
cd ../OM1
uv run src/run.py go2_sim_autonomy
```

### Isaac Sim Keyboard Controls

| Key | Action |
|-----|--------|
| `↑` / `Numpad 8` | Move forward |
| `↓` / `Numpad 2` | Move backward |
| `←` / `Numpad 4` | Strafe left |
| `→` / `Numpad 6` | Strafe right |
| `N` / `Numpad 7` | Rotate left |
| `M` / `Numpad 9` | Rotate right |

### Isaac Sim Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `robot_type` | `go2` | `go2` or `g1` |
| `policy_dir` | auto | Path to policy directory |
| `launch_sensors` | `true` | Launch sensor nodes from OM1-ros2-sdk |

### Isaac Sim CLI Options (run.py)

These are passed directly if running `run.py` manually (advanced usage):

| Argument | Default | Description |
|----------|---------|-------------|
| `--robot_type` | `go2` | `go2` or `g1` |
| `--policy_dir` | auto | Path to policy directory |

---

## Repository Structure

```
OM1-sim/
├── gazebo_sim/           # Gazebo simulation packages
│   ├── go2_sim/          # Launch, sport API bridge, battery mock
│   ├── go2_description/  # URDF, meshes, worlds
│   ├── om_path/          # Lidar-based path feasibility
│   ├── champ/            # Quadruped locomotion controller
│   ├── champ_base/       # CHAMP base package
│   └── champ_msgs/       # CHAMP message definitions
├── isaac_sim/            # Isaac Sim simulation
│   ├── launch/           # ROS2 launch file (isaac_sim_launch.py)
│   ├── run.py            # Main simulation script
│   ├── utils.py          # ROS2 OmniGraph utilities
│   ├── assets/           # Robot USD models (Go2, G1)
│   └── checkpoints/      # Pre-trained locomotion policies
├── om_api/               # OpenMind message definitions (shared)
├── unitree_api/          # Unitree API messages (shared)
├── unitree_go/           # Unitree Go messages (shared)
├── cyclonedds/           # CycloneDDS config
├── zenoh/                # Zenoh bridge config
└── README.md
```

## License

MIT
