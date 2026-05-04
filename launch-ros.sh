#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/OM1-sim/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

exec ros2 launch /home/ubuntu/OM1-sim/isaac_sim/launch/isaac_sim_launch_support_nodes.py
