"""Per-robot velocity-tracking policies and a factory to build the right one.

Importing this package pulls in Isaac Sim modules, so it must be imported only
after ``SimulationApp(...)`` has been created.
"""

import os

from ._common import _expand_param, _load_yaml  # re-exported for run.py
from .g1 import G1VelocityPolicy
from .go2 import Go2VelocityPolicy
from .tron1 import Tron1VelocityPolicy

__all__ = [
    "Go2VelocityPolicy",
    "G1VelocityPolicy",
    "Tron1VelocityPolicy",
    "build_policy",
    "_load_yaml",
    "_expand_param",
]


def build_policy(
    robot_cfg,
    robot_root,
    usd_path,
    policy_path,
    env_path,
    deploy_path,
    policy_dir,
    init_pos,
):
    """Instantiate the velocity policy matching robot_cfg.type."""
    if robot_cfg.type == "tron1":
        encoder_path = os.path.join(policy_dir, "exported", "encoder.pt")
        return Tron1VelocityPolicy(
            prim_path=robot_root,
            name="Tron1",
            usd_path=usd_path,
            position=init_pos,
            policy_path=policy_path,
            encoder_path=encoder_path,
            deploy_path=deploy_path,
            history_length=robot_cfg.history_length,
        )
    if robot_cfg.type == "g1":
        return G1VelocityPolicy(
            prim_path=robot_root,
            name="G1",
            usd_path=usd_path,
            position=init_pos,
            policy_path=policy_path,
            env_path=env_path,
            deploy_path=deploy_path,
            history_length=robot_cfg.history_length,
        )
    return Go2VelocityPolicy(
        prim_path=robot_root,
        name="Go2",
        usd_path=usd_path,
        position=init_pos,
        policy_path=policy_path,
        env_path=env_path,
    )
