from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from isaac_lab_from_scrach_mdp.isaac_lab_from_scrach_mdp_common import quaternion_to_euler_torch


# Base attitude
def base_attitude_err(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    euler_angles = quaternion_to_euler_torch(asset.data.root_quat_w)
    target_angles = torch.tensor([0.0, 0.0, 1.57], device="cuda:0")  # (3,)
    diff = euler_angles - target_angles.unsqueeze(0).repeat(euler_angles.size(0), 1)
    # base_attitude_err_l2_reward = torch.sum(torch.square(diff), dim=1)
    base_attitude_err_reward = torch.sum(torch.abs(diff), dim=1)
    # print("base_attitude_err_l2_reward", base_attitude_err_l2_reward)
    return base_attitude_err_reward


# Base height
def base_height_err_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, -1]  # shape: (N,)
    target_height = torch.tensor(0.25, device="cuda:0")
    height_diff = current_height - target_height  # shape: (N,)
    base_height_err_l2_reward = torch.square(height_diff)
    # print("base_height_err_l2_reward", base_height_err_l2_reward)
    return base_height_err_l2_reward


# joints
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torques_l2_reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    # print("joint_torques_l2_reward", joint_torques_l2_reward)
    return joint_torques_l2_reward


def drive_joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    drive_joint_acc_l2_reward = torch.sum(torch.square(asset.data.joint_acc[:, 0:4]), dim=1)
    # print("drive_joint_acc_l2_reward", drive_joint_acc_l2_reward)
    return drive_joint_acc_l2_reward


def wheel_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_acc_l2_reward = torch.sum(torch.square(asset.data.joint_acc[:, 8:]), dim=1)
    # print("wheel_acc_l2_reward", wheel_acc_l2_reward)
    return wheel_acc_l2_reward


def wheel_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel_reward = torch.sum(torch.abs(asset.data.joint_vel[:, 8:]), dim=1)
    # print("wheel_vel_reward", wheel_vel_reward)
    return wheel_vel_reward


def drive_joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(asset.data.joint_pos[:, 0:4] - asset.data.soft_joint_pos_limits[:, 0:4, 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos[:, 0:4] - asset.data.soft_joint_pos_limits[:, 0:4, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# smooth
def base_lin_vel_err_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_lin_vel = asset.data.root_lin_vel_w  # shape: (N,)
    target_lin_vel = torch.tensor([0.0, 0.0, 0.0], device="cuda:0")
    lin_vel_diff = current_lin_vel - target_lin_vel.unsqueeze(0).repeat(current_lin_vel.size(0), 1)
    base_lin_vel_err_l2_reward = torch.sum(torch.square(lin_vel_diff), dim=1)
    # print("base_lin_vel_err_l2_reward", base_lin_vel_err_l2_reward)
    return base_lin_vel_err_l2_reward


pre_action = None


def action_smooth(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    action = env.action_manager.action
    global pre_action

    if pre_action is None:
        pre_action = action.clone()
        action_smooth_reward = torch.tensor(0.0)  # 第一次调用时返回0
    else:
        # 先计算 reward，使用上一时刻的 pre_action
        action_smooth_reward = torch.sum(torch.square(action - pre_action))
        # print(pre_action, action, action_smooth_reward)
        # 然后更新 pre_action 为当前 action，供下一次调用使用
        pre_action = action.clone()

    return action_smooth_reward


def base_lin_acc_err_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_lin_acc = asset.data.body_acc_w[:, 0, :3]  # shape: (N,)
    target_lin_acc = torch.tensor([0.0, 0.0, 0.0], device="cuda:0")
    lin_acc_diff = current_lin_acc - target_lin_acc.unsqueeze(0).repeat(current_lin_acc.size(0), 1)
    base_lin_acc_err_l2_reward = torch.sum(torch.square(lin_acc_diff), dim=1)
    # print("base_lin_acc_err_l2_reward", base_lin_acc_err_l2_reward)
    return base_lin_acc_err_l2_reward


def feet_ground_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, dt: float) -> torch.Tensor:
    """Reward long contact time with the ground.

    This function rewards the agent for keeping its feet in contact with the ground for a long duration.
    The reward is computed based on the current contact time of the feet, summed across all specified body parts.

    Args:
        env: The reinforcement learning environment.
        sensor_cfg: The sensor configuration specifying which bodies to track.
        dt: The time window to consider for detecting contact.

    Returns:
        A tensor containing the ground contact reward for each agent in the batch.
    """
    # extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # compute contact information
    contact_time = contact_sensor.compute_first_contact(dt)[:, sensor_cfg.body_ids]
    # print(contact_time)
    # sum contact times across all specified bodies (e.g., feet)
    feet_ground_time_reward = torch.sum(contact_time, dim=1)
    # print("feet_ground_time_reward", feet_ground_time_reward)
    return feet_ground_time_reward
