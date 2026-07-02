"""LimX TRON1 WheelFoot velocity-tracking policy with an observation-history encoder."""

import logging
from typing import Optional, Tuple

import carb
import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction

from ._common import _load_yaml

logger = logging.getLogger(__name__)


class Tron1VelocityPolicy:
    """
    LimX TRON1 WheelFoot running a velocity tracking locomotion policy with
    observation history encoder.

    Architecture: encoder(obs_history) -> latent, then policy([latent, obs, cmd]) -> actions.
    8 DOF: 6 leg joints (position control) + 2 wheel joints (velocity control).
    """

    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        encoder_path: str,
        deploy_path: str,
        root_path: Optional[str] = None,
        name: str = "tron1",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        history_length: int = 10,
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

        # Load deploy config
        self._deploy_cfg = _load_yaml(deploy_path)
        if not self._deploy_cfg:
            raise FileNotFoundError(f"deploy.yaml required for TRON1: {deploy_path}")
        logger.info("[TRON1] Loaded deploy config from %s", deploy_path)

        # Load policy (actor MLP) and encoder as JIT models
        import io

        import omni

        file_content = omni.client.read_file(policy_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.policy = torch.jit.load(file)
        logger.info("[TRON1] Loaded policy from %s", policy_path)

        file_content = omni.client.read_file(encoder_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.encoder_model = torch.jit.load(file)
        logger.info("[TRON1] Loaded encoder from %s", encoder_path)

        self._decimation = self._deploy_cfg.get("decimation", 4)
        self._history_length = self._deploy_cfg.get("history_length", history_length)
        self._num_leg_joints = self._deploy_cfg.get("num_leg_joints", 6)
        self._num_wheel_joints = self._deploy_cfg.get("num_wheel_joints", 2)

        # Observation scales from deploy config
        deploy_obs_cfg = self._deploy_cfg.get("observations", {})
        self._obs_scales = {}
        for obs_name in [
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ]:
            scale = deploy_obs_cfg.get(obs_name, {}).get("scale")
            if scale is None:
                self._obs_scales[obs_name] = 1.0
            elif isinstance(scale, (int, float)):
                self._obs_scales[obs_name] = float(scale)
            else:
                self._obs_scales[obs_name] = np.array(scale, dtype=np.float32)

        # Action config
        leg_action_cfg = self._deploy_cfg.get("actions", {}).get("leg", {})
        wheel_action_cfg = self._deploy_cfg.get("actions", {}).get("wheel", {})
        self._leg_action_scale = leg_action_cfg.get("scale", 0.25)
        self._wheel_action_scale = wheel_action_cfg.get("scale", 1.0)

        # Default joint positions and PD gains
        self._default_pos_cfg = np.array(
            self._deploy_cfg.get("default_joint_pos", [0.0] * 8), dtype=np.float32
        )
        self._stiffness = np.array(
            self._deploy_cfg.get("stiffness", []), dtype=np.float32
        )
        self._damping = np.array(self._deploy_cfg.get("damping", []), dtype=np.float32)

        self.default_pos = None
        self.default_vel = None
        self._previous_action = None
        self._policy_counter = 0
        self._obs_history = None

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs_tensor).detach().view(-1).numpy()
        return action

    def _encode(self, obs_history: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            hist_tensor = torch.from_numpy(obs_history).view(1, -1).float()
            latent = self.encoder_model(hist_tensor).detach().view(-1).numpy()
        return latent

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
        logger.info("[TRON1] Articulation has %d DOFs", dof_count)
        logger.info("[TRON1] Joint names: %s", self.robot.dof_names)

        if len(self._default_pos_cfg) != dof_count:
            raise ValueError(
                f"deploy.yaml default_joint_pos has {len(self._default_pos_cfg)} "
                f"values, expected {dof_count}"
            )

        self.default_pos = self._default_pos_cfg.copy()
        self.default_vel = np.zeros(dof_count, dtype=np.float32)

        # Apply PD gains
        if len(self._stiffness) == dof_count and len(self._damping) == dof_count:
            self.robot._articulation_view.set_gains(self._stiffness, self._damping)
            logger.info("[TRON1] Applied stiffness/damping from deploy.yaml")

        self.robot.set_joint_positions(self.default_pos)
        self.robot.set_joint_velocities(self.default_vel)

        self._previous_action = np.zeros(dof_count, dtype=np.float32)
        self.action = np.zeros(dof_count, dtype=np.float32)

        # Initialize observation history buffer
        obs_dim = (
            3  # base_ang_vel
            + 3  # projected_gravity
            + self._num_leg_joints  # joint_pos_rel (legs only)
            + dof_count  # joint_vel (all)
            + dof_count  # last_action (all)
        )
        self._obs_history = [
            np.zeros(obs_dim, dtype=np.float32) for _ in range(self._history_length)
        ]

        logger.info(
            "[TRON1] Init complete - %d DOFs, obs_dim=%d, history=%d, "
            "encoder_input=%d",
            dof_count,
            obs_dim,
            self._history_length,
            obs_dim * self._history_length,
        )

    def _compute_observation(
        self, command: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute current obs and flattened history."""
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()

        # Leg joint positions only (first num_leg_joints)
        leg_pos_rel = (
            current_joint_pos[: self._num_leg_joints]
            - self.default_pos[: self._num_leg_joints]
        )

        obs = np.concatenate(
            [
                ang_vel_b * self._obs_scales["base_ang_vel"],
                gravity_b * self._obs_scales["projected_gravity"],
                leg_pos_rel * self._obs_scales["joint_pos_rel"],
                current_joint_vel * self._obs_scales["joint_vel_rel"],
                self._previous_action * self._obs_scales["last_action"],
            ],
            axis=0,
        ).astype(np.float32)

        # Update history
        self._obs_history.pop(0)
        self._obs_history.append(obs.copy())

        # Flatten history: [obs_t-9, obs_t-8, ..., obs_t]
        obs_history_flat = np.concatenate(self._obs_history, axis=0).astype(np.float32)

        return obs, obs_history_flat

    def forward(self, dt: float, command: np.ndarray) -> None:
        """Step policy forward: encode history, run actor, apply mixed actions."""
        if self._policy_counter % self._decimation == 0:
            obs, obs_history_flat = self._compute_observation(command)

            # Encoder: history -> latent
            latent = self._encode(obs_history_flat)

            # Policy input: [latent, obs, command]
            policy_input = np.concatenate([latent, obs, command], axis=0).astype(
                np.float32
            )
            self.action = np.array(self._compute_action(policy_input), dtype=np.float32)
            self._previous_action = self.action.copy()

        # Apply actions: legs get position targets, wheels get velocity targets
        n_leg = self._num_leg_joints
        leg_actions = self.action[:n_leg]
        wheel_actions = self.action[n_leg:]

        target_pos = self.default_pos.copy()
        target_pos[:n_leg] = (
            self.default_pos[:n_leg] + self._leg_action_scale * leg_actions
        )

        target_vel = np.zeros_like(self.default_pos)
        target_vel[n_leg:] = self._wheel_action_scale * wheel_actions

        action = ArticulationAction(
            joint_positions=target_pos, joint_velocities=target_vel
        )
        self.robot.apply_action(action)
        self._policy_counter += 1
