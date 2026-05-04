#!/usr/bin/env python3
"""
Mock unitree_go/LowState publisher for Isaac Sim.

Publishes /lowstate and /lf/lowstate at 100Hz with a battery drain/charge
sim. Reads /joint_states and remaps the 12 leg DOFs into Unitree motor order.

Toggle charging at runtime:
    ros2 param set /go2_lowstate_node is_charging true

"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import JointState
from unitree_go.msg import BmsState, IMUState, LowState, MotorState

# Unitree LowState motor_state order (12 active DOFs, indices 0-11):
#   FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#   RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
# Each entry lists candidate joint names for that motor — Isaac Sim uses
# Unitree-style names; the alternates cover other URDF conventions in case
# this node is reused outside Isaac Sim.
UNITREE_MOTOR_NAME_CANDIDATES = [
    ("FR_hip_joint",   "rf_hip_joint"),
    ("FR_thigh_joint", "rf_upper_leg_joint"),
    ("FR_calf_joint",  "rf_lower_leg_joint"),
    ("FL_hip_joint",   "lf_hip_joint"),
    ("FL_thigh_joint", "lf_upper_leg_joint"),
    ("FL_calf_joint",  "lf_lower_leg_joint"),
    ("RR_hip_joint",   "rh_hip_joint"),
    ("RR_thigh_joint", "rh_upper_leg_joint"),
    ("RR_calf_joint",  "rh_lower_leg_joint"),
    ("RL_hip_joint",   "lh_hip_joint"),
    ("RL_thigh_joint", "lh_upper_leg_joint"),
    ("RL_calf_joint",  "lh_lower_leg_joint"),
]


class Go2LowstateNode(Node):
    def __init__(self):
        super().__init__("go2_lowstate_node")

        self.declare_parameter(
            "is_charging",
            False,
            ParameterDescriptor(
                description="Whether the robot is currently charging",
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        self.declare_parameter("soc", 100.0)
        self.declare_parameter("drain_rate", 0.001)
        self.declare_parameter("idle_drain_rate", 0.0005)
        self.declare_parameter("charge_rate", 0.05)

        self.soc = self.get_parameter("soc").value
        self.drain_rate = self.get_parameter("drain_rate").value
        self.idle_drain_rate = self.get_parameter("idle_drain_rate").value
        self.charge_rate = self.get_parameter("charge_rate").value

        self.is_moving = False
        self.last_cmd_vel_time = 0.0
        self.cmd_vel_timeout = 0.5

        self._tick_counter = 0
        self._latest_joint_state: JointState | None = None
        # Cache: index into JointState.position/velocity for each Unitree motor,
        # or -1 if no matching joint name was found in the latest message.
        self._unitree_to_jointstate_idx = [-1] * 12
        self._joint_name_cache_key: tuple = ()

        self._lowstate_pub = self.create_publisher(LowState, "/lowstate", 10)
        self._lf_lowstate_pub = self.create_publisher(LowState, "/lf/lowstate", 10)

        self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)

        self.add_on_set_parameters_callback(self._on_param_change)

        self._timer = self.create_timer(0.01, self._publish_lowstate)

        self.get_logger().info(
            "Lowstate publisher started (100Hz to /lowstate, /lf/lowstate; "
            "reading /joint_states for motor positions)"
        )

    def _on_param_change(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == "is_charging":
                status = "STARTED" if param.value else "STOPPED"
                self.get_logger().info(f"Charging {status}")
            elif param.name == "soc":
                self.soc = param.value
        return SetParametersResult(successful=True)

    def _cmd_vel_callback(self, msg):
        linear_speed = (msg.linear.x**2 + msg.linear.y**2) ** 0.5
        angular_speed = abs(msg.angular.z)
        self.is_moving = linear_speed > 0.01 or angular_speed > 0.01
        self.last_cmd_vel_time = time.time()

    def _joint_state_callback(self, msg: JointState):
        self._latest_joint_state = msg
        # Rebuild the index map only when the joint name list changes.
        cache_key = tuple(msg.name)
        if cache_key != self._joint_name_cache_key:
            self._joint_name_cache_key = cache_key
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            new_map = []
            for candidates in UNITREE_MOTOR_NAME_CANDIDATES:
                idx = -1
                for cand in candidates:
                    if cand in name_to_idx:
                        idx = name_to_idx[cand]
                        break
                new_map.append(idx)
            self._unitree_to_jointstate_idx = new_map
            missing = [
                UNITREE_MOTOR_NAME_CANDIDATES[i][0]
                for i, idx in enumerate(new_map)
                if idx < 0
            ]
            if missing:
                self.get_logger().warn(
                    f"Joint name mapping missing for: {missing}. Got names: {msg.name}"
                )
            else:
                self.get_logger().info(
                    "Mapped 12/12 leg joints from /joint_states to Unitree order"
                )

    def _get_motor_qdq(self):
        """Return (positions, velocities) lists of length 12 in Unitree order."""
        positions = [0.0] * 12
        velocities = [0.0] * 12
        msg = self._latest_joint_state
        if msg is None:
            return positions, velocities
        n_pos = len(msg.position)
        n_vel = len(msg.velocity)
        for unitree_idx, js_idx in enumerate(self._unitree_to_jointstate_idx):
            if js_idx < 0:
                continue
            if js_idx < n_pos:
                positions[unitree_idx] = float(msg.position[js_idx])
            if js_idx < n_vel:
                velocities[unitree_idx] = float(msg.velocity[js_idx])
        return positions, velocities

    def _publish_lowstate(self):
        now = time.time()

        if now - self.last_cmd_vel_time > self.cmd_vel_timeout:
            self.is_moving = False

        is_charging = self.get_parameter("is_charging").value

        if is_charging:
            self.soc = min(100.0, self.soc + self.charge_rate)
            current_ma = 3000
        elif self.is_moving:
            self.soc = max(0.0, self.soc - self.drain_rate)
            current_ma = -5000
        else:
            self.soc = max(0.0, self.soc - self.idle_drain_rate)
            current_ma = -500

        msg = LowState()
        msg.head = [254, 239]
        msg.level_flag = 0
        msg.frame_reserve = 0
        msg.sn = [0, 0]
        msg.version = [0, 0]
        msg.bandwidth = 0

        msg.imu_state = IMUState()
        msg.imu_state.quaternion = [0.0, 0.0, 0.0, 1.0]
        msg.imu_state.gyroscope = [0.0, 0.0, 0.0]
        msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
        msg.imu_state.rpy = [0.0, 0.0, 0.0]
        msg.imu_state.temperature = 25

        positions, velocities = self._get_motor_qdq()
        motor_states = []
        for i in range(20):
            motor = MotorState()
            if i < 12:
                motor.mode = 1
                motor.q = positions[i]
                motor.dq = velocities[i]
                motor.ddq = 0.0
                motor.tau_est = 0.025 + (i % 3) * 0.01
                motor.q_raw = 0.0
                motor.dq_raw = 0.0
                motor.ddq_raw = 0.0
                motor.temperature = 26 + (i % 6)
                motor.lost = 0
                motor.reserve = [0, 588]
            else:
                motor.mode = 0
                motor.reserve = [0, 0]
            motor_states.append(motor)
        msg.motor_state = motor_states

        msg.bms_state = BmsState()
        msg.bms_state.version_high = 1
        msg.bms_state.version_low = 18
        msg.bms_state.status = 8
        msg.bms_state.soc = int(self.soc)
        msg.bms_state.current = current_ma
        msg.bms_state.cycle = 5
        msg.bms_state.bq_ntc = [25, 23]
        msg.bms_state.mcu_ntc = [29, 28]
        msg.bms_state.cell_vol = [
            3654, 3663, 3664, 3663, 3662, 3662, 3663, 3653,
            0, 0, 0, 0, 0, 0, 0,
        ]

        msg.foot_force = [21, 22, 22, 22]
        msg.foot_force_est = [0, 0, 0, 0]
        msg.tick = self._tick_counter
        self._tick_counter += 1
        msg.wireless_remote = [0] * 40
        msg.power_v = 24.0 + (self.soc / 100.0) * 9.6
        msg.power_a = current_ma / 1000.0
        msg.bit_flag = 36
        msg.adc_reel = 0.003
        msg.temperature_ntc1 = 43
        msg.temperature_ntc2 = 40
        msg.fan_frequency = [0, 0, 0, 0]
        msg.reserve = 0
        msg.crc = 0

        self._lowstate_pub.publish(msg)
        self._lf_lowstate_pub.publish(msg)

        if self._tick_counter % 500 == 0:
            status = "CHARGING" if is_charging else (
                "MOVING" if self.is_moving else "IDLE"
            )
            joints_status = "joints OK" if self._latest_joint_state else "no joints yet"
            self.get_logger().info(
                f"[{status}] SoC: {int(self.soc)}%, {joints_status}"
            )


def main():
    rclpy.init()
    node = Go2LowstateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
