from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def quaternion_to_euler_torch(quaternions: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Convert quaternions to Euler angles (in radians) using GPU.

    Args:
        quaternions (torch.Tensor): Tensor of shape (N, 4), where each row is [w, x, y, z].
        convention (str): Rotation order, default is "XYZ".

    Returns:
        torch.Tensor: Tensor of shape (N, 3), where each row is [roll, pitch, yaw].
    """
    assert convention == "XYZ", "Only XYZ convention is supported in this implementation."
    assert quaternions.shape[-1] == 4, "Input tensor must have shape (N, 4)."

    # Ensure the tensor is on the GPU
    quaternions = quaternions.to("cuda") if not quaternions.is_cuda else quaternions

    # Extract components of the quaternion
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute roll (X-axis rotation)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Compute pitch (Y-axis rotation)
    sin_pitch = 2 * (w * y - z * x)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)  # Clamp to avoid invalid arcsin values
    pitch = torch.asin(sin_pitch)

    # Compute yaw (Z-axis rotation)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Combine results into a tensor
    euler_angles = torch.stack([roll, pitch, yaw], dim=1)

    return euler_angles
