"""Unitree Go2 velocity-tracking locomotion policy."""

from typing import Optional

import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.robot.policy.examples.controllers.config_loader import (
    get_action,
    get_observations,
)

from ._common import _expand_param


class Go2VelocityPolicy(PolicyController):
    """The Unitree Go2 running a velocity tracking locomotion policy."""

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        env_path: str,
        root_path: Optional[str] = None,
        name: str = "go2",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        self.load_policy(policy_path, env_path)

        self._obs_order = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ]

        obs_cfg = get_observations(self.policy_env_params) or {}
        self._obs_scales = {}
        for name in self._obs_order:
            scale = obs_cfg.get(name, {}).get("scale")
            if scale is None:
                self._obs_scales[name] = 1.0
            elif isinstance(scale, (int, float)):
                self._obs_scales[name] = float(scale)
            else:
                self._obs_scales[name] = np.array(scale, dtype=np.float32)

        action_terms = get_action(self.policy_env_params) or {}
        self._action_cfg = next(iter(action_terms.values()), {})

        self._action_scale = None
        self._action_offset = None
        self._previous_action = None
        self._policy_counter = 0

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize the robot controller with physics simulation view and control mode."""
        super().initialize(physics_sim_view=physics_sim_view, control_mode="position")
        dof_count = len(self.default_pos)
        self._action_scale = _expand_param(
            self._action_cfg.get("scale"), dof_count, default=1.0
        )
        if self._action_cfg.get("use_default_offset", False):
            self._action_offset = np.array(self.default_pos, dtype=np.float32)
        else:
            self._action_offset = _expand_param(
                self._action_cfg.get("offset"), dof_count, default=0.0
            )

        self._previous_action = np.zeros(dof_count, dtype=np.float32)
        self.action = np.zeros(dof_count, dtype=np.float32)

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        joint_pos_rel = current_joint_pos - self.default_pos

        obs = np.concatenate(
            [
                ang_vel_b * self._obs_scales["base_ang_vel"],
                gravity_b * self._obs_scales["projected_gravity"],
                command * self._obs_scales["velocity_commands"],
                joint_pos_rel * self._obs_scales["joint_pos_rel"],
                current_joint_vel * self._obs_scales["joint_vel_rel"],
                self._previous_action * self._obs_scales["last_action"],
            ],
            axis=0,
        ).astype(np.float32)
        return obs

    def forward(self, dt: float, command: np.ndarray) -> None:
        """Execute one forward step of the policy with the given command."""
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = np.array(self._compute_action(obs), dtype=np.float32)
            self._previous_action = self.action.copy()

        target_pos = self._action_offset + (self._action_scale * self.action)
        action = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action)
        self._policy_counter += 1
