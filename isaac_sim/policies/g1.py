"""Unitree G1 humanoid velocity-tracking locomotion policy with observation history."""

import logging
import os
from typing import Optional

import carb
import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController

from ._common import _expand_param, _load_yaml

logger = logging.getLogger(__name__)


class G1VelocityPolicy(PolicyController):
    """
    The Unitree G1 humanoid running a velocity tracking locomotion policy with observation history.
    """

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        env_path: str,
        deploy_path: Optional[str] = None,
        root_path: Optional[str] = None,
        name: str = "g1",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        history_length: int = 5,
    ) -> None:
        import torch
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import define_prim, get_prim_at_path

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        if root_path is None:
            self.robot = SingleArticulation(
                prim_path=prim_path,
                name=name,
                position=position,
                orientation=orientation,
            )
        else:
            self.robot = SingleArticulation(
                prim_path=root_path,
                name=name,
                position=position,
                orientation=orientation,
            )

        self._deploy_cfg = {}
        if deploy_path and os.path.isfile(deploy_path):
            self._deploy_cfg = _load_yaml(deploy_path)
            logger.info("[G1] Loaded deploy config from %s", deploy_path)
        else:
            raise FileNotFoundError(f"deploy.yaml required for G1: {deploy_path}")

        import io

        import omni

        file_content = omni.client.read_file(policy_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.policy = torch.jit.load(file)
        logger.info("[G1] Loaded policy from %s", policy_path)

        self._decimation = int(
            self._deploy_cfg.get("step_dt", 0.02) / 0.005
        )  # Assume sim dt is 0.005
        if "decimation" in self._deploy_cfg:
            self._decimation = self._deploy_cfg["decimation"]

        self._history_length = history_length

        # Joint reordering: joint_ids_map[sim_idx] = sdk_idx
        self._joint_ids_map = self._deploy_cfg.get("joint_ids_map")
        if self._joint_ids_map:
            logger.info(
                "[G1] Joint reordering enabled: %d joints", len(self._joint_ids_map)
            )

        deploy_obs_cfg = self._deploy_cfg.get("observations", {})
        self._obs_scales = {}
        obs_names = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ]
        for obs_name in obs_names:
            scale = deploy_obs_cfg.get(obs_name, {}).get("scale")
            if scale is None:
                self._obs_scales[obs_name] = 1.0
            elif isinstance(scale, (int, float)):
                self._obs_scales[obs_name] = float(scale)
            else:
                self._obs_scales[obs_name] = np.array(scale, dtype=np.float32)

        self._action_cfg = self._deploy_cfg.get("actions", {}).get(
            "JointPositionAction", {}
        )

        self._action_scale = None
        self._action_offset = None
        self._previous_action = None
        self._policy_counter = 0

        self._default_pos_sim = np.array(
            self._deploy_cfg.get("default_joint_pos", []), dtype=np.float32
        )
        self._stiffness_sdk = np.array(
            self._deploy_cfg.get("stiffness", []), dtype=np.float32
        )
        self._damping_sdk = np.array(
            self._deploy_cfg.get("damping", []), dtype=np.float32
        )

        self.default_pos = None
        self.default_vel = None

        # Per-term observation history buffers
        self._obs_term_histories = None
        self._obs_term_names = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ]

    def _sdk_to_sim(self, sdk_array: np.ndarray) -> np.ndarray:
        """
        Convert SDK-order array to simulation order.
        """
        if self._joint_ids_map is None or len(sdk_array) != len(self._joint_ids_map):
            return sdk_array
        sim_array = np.zeros_like(sdk_array)
        for sim_idx, sdk_idx in enumerate(self._joint_ids_map):
            sim_array[sim_idx] = sdk_array[sdk_idx]
        return sim_array

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_tensor).detach().view(-1).numpy()
        return action

    def post_reset(self) -> None:
        """Reset robot state after an episode."""
        self.robot.post_reset()

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize robot articulation and physics simulation."""
        from omni.physx import get_physx_simulation_interface

        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes("force")

        get_physx_simulation_interface().flush_changes()

        self.robot.get_articulation_controller().switch_control_mode("position")

        dof_count = len(self.robot.dof_names)
        logger.info("[G1] Articulation has %d DOFs", dof_count)
        logger.info("[G1] Joint names: %s", self.robot.dof_names)

        if len(self._default_pos_sim) != dof_count:
            raise ValueError(
                f"deploy.yaml default_joint_pos has {len(self._default_pos_sim)} values, expected {dof_count}"
            )

        self.default_pos = self._default_pos_sim.copy()
        self.default_vel = np.zeros(dof_count, dtype=np.float32)

        if len(self._stiffness_sdk) == dof_count:
            stiffness_sim = self._sdk_to_sim(self._stiffness_sdk)
            damping_sim = (
                self._sdk_to_sim(self._damping_sdk)
                if len(self._damping_sdk) == dof_count
                else None
            )
            self.robot._articulation_view.set_gains(stiffness_sim, damping_sim)
            logger.info("[G1] Applied stiffness/damping from deploy.yaml")

        self.robot.set_joint_positions(self.default_pos)
        self.robot.set_joint_velocities(self.default_vel)
        logger.info("[G1] Set initial joint positions")

        self._action_scale = _expand_param(
            self._action_cfg.get("scale"), dof_count, default=0.25
        )
        offset_val = self._action_cfg.get("offset")
        if (
            offset_val is not None
            and isinstance(offset_val, (list, np.ndarray))
            and len(offset_val) == dof_count
        ):
            self._action_offset = np.array(offset_val, dtype=np.float32)
        else:
            self._action_offset = self.default_pos.copy()

        self._previous_action = np.zeros(dof_count, dtype=np.float32)
        self.action = np.zeros(dof_count, dtype=np.float32)

        # Initialize PER-TERM observation history buffers
        term_sizes = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_commands": 3,
            "joint_pos_rel": dof_count,
            "joint_vel_rel": dof_count,
            "last_action": dof_count,
        }
        self._obs_term_histories = {}
        for term_name in self._obs_term_names:
            size = term_sizes[term_name]
            self._obs_term_histories[term_name] = [
                np.zeros(size, dtype=np.float32) for _ in range(self._history_length)
            ]

        total_obs_size = sum(
            term_sizes[name] * self._history_length for name in self._obs_term_names
        )
        logger.info(
            "[G1] Initialization complete - %d DOFs, obs size: %d",
            dof_count,
            total_obs_size,
        )

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        """Compute observation with per-term history."""
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        joint_pos_rel = current_joint_pos - self.default_pos

        current_terms = {
            "base_ang_vel": (ang_vel_b * self._obs_scales["base_ang_vel"]).astype(
                np.float32
            ),
            "projected_gravity": (
                gravity_b * self._obs_scales["projected_gravity"]
            ).astype(np.float32),
            "velocity_commands": (
                command * self._obs_scales["velocity_commands"]
            ).astype(np.float32),
            "joint_pos_rel": (joint_pos_rel * self._obs_scales["joint_pos_rel"]).astype(
                np.float32
            ),
            "joint_vel_rel": (
                current_joint_vel * self._obs_scales["joint_vel_rel"]
            ).astype(np.float32),
            "last_action": (
                self._previous_action * self._obs_scales["last_action"]
            ).astype(np.float32),
        }

        for term_name in self._obs_term_names:
            self._obs_term_histories[term_name].pop(0)
            self._obs_term_histories[term_name].append(current_terms[term_name])

        obs_parts = []
        for term_name in self._obs_term_names:
            term_history = np.concatenate(self._obs_term_histories[term_name], axis=0)
            obs_parts.append(term_history)

        return np.concatenate(obs_parts, axis=0)

    def forward(self, dt: float, command: np.ndarray) -> None:
        """Step policy forward and apply actions to robot."""
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = np.array(self._compute_action(obs), dtype=np.float32)
            self._previous_action = self.action.copy()

        target_pos = self._action_offset + (self._action_scale * self.action)
        action = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action)
        self._policy_counter += 1
