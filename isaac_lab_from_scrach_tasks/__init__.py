# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym
import isaac_lab_from_scrach_agents
from .isaac_lab_from_scrach_env import BIP_WL_EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="bip-wl-minimal",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": isaac_lab_from_scrach_env.BIP_WL_EnvCfg,
        "rsl_rl_cfg_entry_point": f"{isaac_lab_from_scrach_agents.__name__}.rsl_rl_ppo_cfg:BIPWLPPORunnerCfg",
        "sb3_cfg_entry_point": f"{isaac_lab_from_scrach_agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
