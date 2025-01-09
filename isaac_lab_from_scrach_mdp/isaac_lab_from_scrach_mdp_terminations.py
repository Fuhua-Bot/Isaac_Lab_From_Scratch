from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

import isaac_lab_from_scrach_mdp.isaac_lab_from_scrach_mdp_common as isaac_lab_from_scrach_mdp_common


def over_joint_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), angle_limit: float = 1.0
) -> torch.Tensor:
    """
    Args:
        env (ManagerBasedRLEnv): 环境对象。
        asset_cfg (SceneEntityCfg): 场景配置。
        angle_limit (float): 角度限制（弧度），默认为1.0。

    Returns:
        torch.Tensor: 布尔张量，表示每个行是否有超限关节。
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    current_joint_pos = asset.data.joint_pos[:, 0:4] - asset.data.default_joint_pos[:, 0:4]

    over_limit = torch.any(torch.abs(current_joint_pos) > angle_limit, dim=1)
    # print(over_limit)
    return over_limit.to(device=current_joint_pos.device)
