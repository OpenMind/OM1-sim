#!/usr/bin/env bash
set -e

# pick DCV display (matches launch-isaac-sim.sh logic)
unset DISPLAY XAUTHORITY
for d in :0 :1; do
    case "$d" in
        :0) xauth=/run/user/1000/gdm/Xauthority ;;
        :1) xauth=/run/user/1000/dcv/1.xauth ;;
    esac
    if [ -f "$xauth" ] && DISPLAY="$d" XAUTHORITY="$xauth" xset q >/dev/null 2>&1; then
        export DISPLAY="$d"
        export XAUTHORITY="$xauth"
        break
    fi
done

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/1000}

VENV=/home/ubuntu/isaac-sim-venv
source "$VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES

# bundled jazzy lib path
ROS2_LIB=""
for c in \
    "$VENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib" \
    "$VENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib" \
    "$VENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib" \
    "$VENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib" ; do
    [ -d "$c" ] && ROS2_LIB="$c" && break
done
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/ubuntu/cyclonedds.xml
export ROS_DOMAIN_ID=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ROS2_LIB}"

cd ~/OM1-sim/isaac_sim
exec python run.py --robot_type go2
