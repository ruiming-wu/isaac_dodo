# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaac_dodo.assets.robots.dodo import DODO_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class DodoVelocityTrackingEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 8
    observation_space = 35
    # 8 joint positions
    # 8 joint velocities
    # 8 joint efforts
    # 3 base linear velocities
    # 3 base angular velocities
    # roll, pitch, yaw
    # linear velocity command (1 dim)
    # angular velocity command (1 dim)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = DODO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5
    ]

    # task
    cmd_lin_range = [-1.0, 1.0]  # m/s
    cmd_ang_range = [-1.0, 1.0]  # rad/s

    termination_height = 0.3  # m
    termination_roll = 1.0  # rad
    termination_pitch = 1.0  # rad

    # reward scales
    reward_lin_vel_w = 2.0
    reward_ang_vel_w = 2.0
    reward_orientation_w = 0.5
    reward_torque_reg_w = 0.01
    reward_action_rate_w = 0.05
    reward_alive_w = 0.2
    reward_failure_penalty = -100.0

    # shaping sigmas
    lin_vel_sigma = 0.5
    ang_vel_sigma = 0.5
    orientation_sigma = 0.5
    action_rate_sigma = 0.2
