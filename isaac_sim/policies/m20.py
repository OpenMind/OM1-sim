"""DeepRobotics M20 velocity-tracking locomotion policy.

The M20 is a *wheeled* quadruped: 12 leg joints (position-controlled) plus 4
wheel joints (velocity-controlled). The policy therefore differs from the pure
position-control Go2 policy in two ways:

* Action (16): the first 12 outputs are leg joint-position deltas, the last 4
  are wheel joint-velocity targets. Legs and wheels are actuated with different
  drive gains (legs kp=80/kd=2, wheels kp=0/kd=0.6), which the base
  ``PolicyController`` reads straight from ``env.yaml`` and applies per joint.
* Observation (57): base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
  + joint_pos_rel(16, wheel entries zeroed) + joint_vel(16) + last_action(16),
  gathered in the training joint order (legs [FL,FR,HL,HR] then wheels).

The USD articulation DOF order is not assumed; joints are matched by name at
initialize time via ``self.robot.dof_names``.
"""

from typing import Optional

import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.robot.policy.examples.controllers.config_loader import get_observations

# Canonical policy joint order (must match the training env.yaml: obs use
# preserve_order=True over these names, actions list them in this order).
LEG_JOINTS = [
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
]
WHEEL_JOINTS = ["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"]

# Per-leg-joint action scale (env.yaml: hipx 0.125, hipy/knee 0.25).
LEG_ACTION_SCALE = np.array(
    [0.125 if "hipx" in n else 0.25 for n in LEG_JOINTS], dtype=np.float32
)
WHEEL_ACTION_SCALE = 5.0  # env.yaml joint_vel action scale


class M20VelocityPolicy(PolicyController):
    """DeepRobotics M20 wheeled quadruped running a velocity-tracking policy."""

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        env_path: str,
        root_path: Optional[str] = None,
        name: str = "m20",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        self.load_policy(policy_path, env_path)

        # Observation scales come straight from env.yaml so the deployed obs
        # matches training term-for-term.
        obs_cfg = get_observations(self.policy_env_params) or {}

        def _scale(term: str, default: float) -> float:
            val = obs_cfg.get(term, {}).get("scale", default)
            return float(val) if isinstance(val, (int, float)) else default

        self._ang_vel_scale = _scale("base_ang_vel", 0.25)
        self._gravity_scale = _scale("projected_gravity", 1.0)
        self._cmd_scale = _scale("velocity_commands", 1.0)
        self._joint_pos_scale = _scale("joint_pos", 1.0)
        self._joint_vel_scale = _scale("joint_vel", 0.05)
        self._action_scale_obs = _scale("actions", 1.0)

        self._num_legs = len(LEG_JOINTS)
        self._num_wheels = len(WHEEL_JOINTS)
        self._num_actions = self._num_legs + self._num_wheels

        # DOF-order index maps, resolved in initialize() once dof_names exist.
        self._leg_ids: Optional[np.ndarray] = None
        self._wheel_ids: Optional[np.ndarray] = None
        self._policy_ids: Optional[np.ndarray] = None

        self._previous_action = np.zeros(self._num_actions, dtype=np.float32)
        self._leg_pos_target: Optional[np.ndarray] = None
        self._wheel_vel_target: Optional[np.ndarray] = None
        self._policy_counter = 0

        # Scripted turn-in-place ("pivot mode"), opt-in via the robot config's
        # pivot_assist field or --pivot_assist.
        self.pivot_assist = False
        self._pivot_wheel_gain = 11.5  # wheel rad/s per commanded rad/s of yaw
        self._pivot_wheel_kp = 6.0  # wheel rad/s per rad/s of yaw-rate error
        self._pivot_wheel_cap = 16.0  # highest wheel target validated
        self._pivot_rate_cap = 0.6  # rad/s of yaw actually attempted
        self._pivot_ramp_s = 0.6  # spin-up time to full differential
        self._pivot_ramp = 0.0
        # Anchor hold: a capped common-mode wheel term (body-x drive) pulls the
        # base back toward the pivot-entry point to limit positional wander.
        self._pivot_anchor: Optional[np.ndarray] = None
        self._pivot_pos_kp = 25.0  # wheel rad/s per metre of anchor error
        self._pivot_vel_kd = 8.0  # wheel rad/s per m/s of body-x speed
        self._pivot_common_cap = 4.0
        self._pivot_leg_action = np.zeros(self._num_legs, dtype=np.float32)

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize the articulation, resolve joint ordering, set defaults."""
        # control_mode="position": legs track position targets (kp=80); wheels
        # have kp=0 so their velocity targets are tracked purely by damping.
        super().initialize(physics_sim_view=physics_sim_view, control_mode="position")

        dof_names = list(self.robot.dof_names)
        missing = [n for n in LEG_JOINTS + WHEEL_JOINTS if n not in dof_names]
        if missing:
            raise ValueError(f"M20 USD is missing expected joints: {missing}")

        self._leg_ids = np.array([dof_names.index(n) for n in LEG_JOINTS], dtype=int)
        self._wheel_ids = np.array(
            [dof_names.index(n) for n in WHEEL_JOINTS], dtype=int
        )
        # Policy joint order for observations: legs then wheels.
        self._policy_ids = np.concatenate([self._leg_ids, self._wheel_ids])

        default_pos = np.asarray(self.default_pos, dtype=np.float32)
        # Start the robot at its default joint configuration.
        self.robot.set_joint_positions(default_pos)

        self._previous_action = np.zeros(self._num_actions, dtype=np.float32)
        self._leg_pos_target = default_pos[self._leg_ids].copy()
        self._wheel_vel_target = np.zeros(self._num_wheels, dtype=np.float32)
        self._policy_counter = 0

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        default_pos = np.asarray(self.default_pos, dtype=np.float32)

        # joint_pos_rel in policy order, with wheel entries zeroed (training used
        # mdp.joint_pos_rel_without_wheel).
        joint_pos_rel = (joint_pos - default_pos)[self._policy_ids].astype(np.float32)
        joint_pos_rel[self._num_legs :] = 0.0
        # joint_vel in policy order (all 16 joints).
        joint_vel_rel = joint_vel[self._policy_ids].astype(np.float32)

        obs = np.concatenate(
            [
                ang_vel_b * self._ang_vel_scale,
                gravity_b * self._gravity_scale,
                command * self._cmd_scale,
                joint_pos_rel * self._joint_pos_scale,
                joint_vel_rel * self._joint_vel_scale,
                self._previous_action * self._action_scale_obs,
            ],
            axis=0,
        ).astype(np.float32)
        return obs

    def forward(self, dt: float, command: np.ndarray) -> None:
        """Run one policy step, applying leg position + wheel velocity targets."""
        if (
            self.pivot_assist
            and abs(command[2]) > 0.2
            and abs(command[0]) < 0.05
            and abs(command[1]) < 0.05
        ):
            self._forward_pivot(dt, command)
            return
        self._pivot_anchor = None  # next pivot re-anchors at its entry pose
        self._pivot_ramp = 0.0
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            action = np.array(self._compute_action(obs), dtype=np.float32)
            self._previous_action = action.copy()

            default_pos = np.asarray(self.default_pos, dtype=np.float32)
            leg_actions = action[: self._num_legs]
            wheel_actions = action[self._num_legs :]
            # Leg targets: default offset + per-joint scaled position delta.
            self._leg_pos_target = (
                default_pos[self._leg_ids] + LEG_ACTION_SCALE * leg_actions
            )
            # Wheel targets: velocity = scale * action (default velocity is 0).
            self._wheel_vel_target = WHEEL_ACTION_SCALE * wheel_actions

        # Build full-DOF targets: positions drive the legs, velocities the wheels.
        pos_cmd = np.asarray(self.default_pos, dtype=np.float32).copy()
        pos_cmd[self._leg_ids] = self._leg_pos_target
        vel_cmd = np.zeros(len(pos_cmd), dtype=np.float32)
        vel_cmd[self._wheel_ids] = self._wheel_vel_target

        self.robot.apply_action(
            ArticulationAction(joint_positions=pos_cmd, joint_velocities=vel_cmd)
        )
        self._policy_counter += 1

    def _forward_pivot(self, dt: float, command: np.ndarray) -> None:
        """Scripted skid-steer pivot: policy-balanced legs, opposed wheel targets."""
        # Measured body yaw rate for the feedback term.
        ang_vel_I = self.robot.get_angular_velocity()
        pos_I, q_IB = self.robot.get_world_pose()
        R_IB = quat_to_rot_matrix(q_IB)
        wz_actual = float(R_IB.transpose()[2] @ ang_vel_I)

        # Capped and ramped target rate: gentle skid forces, no step transient.
        self._pivot_ramp = min(1.0, self._pivot_ramp + dt / self._pivot_ramp_s)
        wz_cmd = (
            float(np.clip(command[2], -self._pivot_rate_cap, self._pivot_rate_cap))
            * self._pivot_ramp
        )

        if self._policy_counter % self._decimation == 0:
            pivot_obs_cmd = np.array([0.0, 0.0, wz_cmd], dtype=np.float32)
            obs = self._compute_observation(pivot_obs_cmd)
            action = np.array(self._compute_action(obs), dtype=np.float32)
            default_pos_l = np.asarray(self.default_pos, dtype=np.float32)
            self._leg_pos_target = (
                default_pos_l[self._leg_ids]
                + LEG_ACTION_SCALE * action[: self._num_legs]
            )
            self._pivot_leg_action = action[: self._num_legs].copy()

        w = float(
            np.clip(
                self._pivot_wheel_gain * wz_cmd
                + self._pivot_wheel_kp * (wz_cmd - wz_actual),
                -self._pivot_wheel_cap,
                self._pivot_wheel_cap,
            )
        )

        # Anchor hold: common-mode wheel velocity toward the pivot-entry point.
        if self._pivot_anchor is None:
            self._pivot_anchor = np.array(pos_I[:2], dtype=np.float64)
        err_w = self._pivot_anchor - np.asarray(pos_I[:2], dtype=np.float64)
        body_x_w = R_IB[:2, 0]  # body x-axis in world (xy)
        err_x = float(err_w @ body_x_w)
        lin_vel_I = self.robot.get_linear_velocity()
        vx_body = float(R_IB.transpose()[0] @ lin_vel_I)
        # Positive joint velocity rolls the wheels BACKWARD (-x), so drive
        # the common mode with the negated body-x correction.
        common = float(
            np.clip(
                -(self._pivot_pos_kp * err_x - self._pivot_vel_kd * vx_body),
                -self._pivot_common_cap,
                self._pivot_common_cap,
            )
        )

        default_pos = np.asarray(self.default_pos, dtype=np.float32)
        self._wheel_vel_target = np.clip(
            np.array(
                [w + common, -w + common, w + common, -w + common],
                dtype=np.float32,
            ),
            -self._pivot_wheel_cap,
            self._pivot_wheel_cap,
        )
        # Keep the policy's obs self-consistent: last_action carries its own
        # leg action plus the scripted wheel targets in action units.
        self._previous_action = np.concatenate(
            [self._pivot_leg_action, self._wheel_vel_target / WHEEL_ACTION_SCALE]
        ).astype(np.float32)

        pos_cmd = default_pos.copy()
        pos_cmd[self._leg_ids] = self._leg_pos_target
        vel_cmd = np.zeros(len(pos_cmd), dtype=np.float32)
        vel_cmd[self._wheel_ids] = self._wheel_vel_target
        self.robot.apply_action(
            ArticulationAction(joint_positions=pos_cmd, joint_velocities=vel_cmd)
        )
        self._policy_counter += 1
