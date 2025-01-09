# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
# from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from isaac_lab_from_scrach_robot import BIP_WL_CFG
from isaac_lab_from_scrach_mdp import isaac_lab_from_scrach_mdp_common
from isaac_lab_from_scrach_mdp import isaac_lab_from_scrach_mdp_reward
from isaac_lab_from_scrach_mdp import isaac_lab_from_scrach_mdp_observation
from isaac_lab_from_scrach_mdp import isaac_lab_from_scrach_mdp_termination

# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # robots
    robot: ArticulationCfg = BIP_WL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_air_time=True,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    null = mdp.NullCommandCfg()

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["LeftWheel", "RightWheel"], scale=10.0)

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "LeftPassiveDrive",
            "LeftMotorDrive",
            "RightMotorDrive",
            "RightPassiveDrive",
        ],
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        drive_joint_pose = ObsTerm(
            func=isaac_lab_from_scrach_mdp_observation.drive_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        drive_joint_vel = ObsTerm(func=isaac_lab_from_scrach_mdp_observation.drive_joint_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        wheel_vel = ObsTerm(func=isaac_lab_from_scrach_mdp_observation.wheel_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.enable_corruption = True
        #     self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "LeftWheel",
                    "RightWheel",
                    "LeftPassiveDrive",
                    "LeftMotorDrive",
                    "RightMotorDrive",
                    "RightPassiveDrive",
                ],
            ),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # reset_all = EventTerm(func=mdp.reset_scene_to_default,
    #     mode="reset",
    #     params={
    #     },)

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Positive Reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)  # range(0-1)

    # Encourage wheel on the ground
    feet_ground_time = RewTerm(
        func=isaac_lab_from_scrach_mdp_reward.feet_ground_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["wheel_left", "wheel_right"]),
            "dt": 0.5,
        },
    )
    # Encourage use wheel for locomotion
    wheel_dof_vel = RewTerm(func=isaac_lab_from_scrach_mdp_reward.wheel_vel, weight=2.5e-4)

    # Negative Reward
    base_attitude = RewTerm(func=isaac_lab_from_scrach_mdp_reward.base_attitude_err, weight=-0.5)
    base_height = RewTerm(func=isaac_lab_from_scrach_mdp_reward.base_height_err_l2, weight=-0.5)
    base_lin_vel = RewTerm(func=isaac_lab_from_scrach_mdp_reward.base_lin_vel_err_l2, weight=-0.1)
    base_lin_acc = RewTerm(func=isaac_lab_from_scrach_mdp_reward.base_lin_acc_err_l2, weight=-1e-3)

    # # # Actuation smooth penalty
    action_smooth = RewTerm(func=isaac_lab_from_scrach_mdp_reward.action_smooth, weight=-1e-5)
    dof_torques_l2 = RewTerm(func=isaac_lab_from_scrach_mdp_reward.joint_torques_l2, weight=-1e-4)

    drive_dof_acc_l2 = RewTerm(func=isaac_lab_from_scrach_mdp_reward.drive_joint_acc_l2, weight=-1e-5)
    wheel_dof_acc_l2 = RewTerm(func=isaac_lab_from_scrach_mdp_reward.wheel_acc_l2, weight=-1e-6)

    # Penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "chassis_base",
                    "passive_link_left",
                    "motor_link_left",
                    "drive_link_right_motor",
                    "drive_link_left_passive",
                    "drive_link_right_passive",
                    "drive_link_left_motor",
                    "motor_link_right",
                    "passive_link_right",
                ],
            ),
            "threshold": 1.0,
        },
    )

    # # -- task
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # wheel_spinning_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.5)

    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-1.0)
    # (3) Primary task: keep pole upright
    # base_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "chassis_base",
                    "passive_link_left",
                    "motor_link_left",
                    "drive_link_right_motor",
                    "drive_link_left_passive",
                    "drive_link_right_passive",
                    "drive_link_left_motor",
                    "motor_link_right",
                    "passive_link_right",
                ],
            ),
            "threshold": 1.0,
        },
    )
    # over_joint_limit = DoneTerm(func=isaac_lab_from_scrach_mdp_termination.over_joint_limit, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class BIP_WL_EnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing = True
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False
