#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/OM1-sim/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# lowstate_node is documented in the launch comment as "auto-spawned by run.py",
# but in practice the new run.py does not start it. Bring it up here so the
# /lowstate, /lf/lowstate, and battery topics are available.
nohup python3 /home/ubuntu/OM1-sim/isaac_sim/lowstate_node.py \
    > /home/ubuntu/OM1-sim/logs/lowstate.out 2>&1 &
echo "lowstate_node PID: $!"

exec ros2 launch /home/ubuntu/OM1-sim/isaac_sim/launch/isaac_sim_launch.py
