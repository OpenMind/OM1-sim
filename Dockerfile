FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-gz-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-effort-controllers \
    ros-humble-xacro \
    ros-humble-urdf \
    ros-humble-tf2-ros \
    ros-humble-joy \
    ros-humble-teleop-twist-joy \
    ros-humble-rviz2 \
    ros-humble-depth-image-proc \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    ros-humble-rosidl-generator-dds-idl \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install zenoh-bridge-ros2dds
RUN echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" > /etc/apt/sources.list.d/zenoh.list \
    && apt-get update \
    && apt-get install -y zenoh-bridge-ros2dds \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /app/om1_sim
COPY . /app/om1_sim/

# Install rosdep dependencies
RUN rosdep update --rosdistro humble || true
RUN . /opt/ros/humble/setup.sh && \
    rosdep install --from-paths . --ignore-src -r -y || true

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Source setup in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /app/om1_sim/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc

# Entrypoint
COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /app/om1_sim/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
exec "$@"
EOF
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "go2_sim", "go2_launch.py"]
