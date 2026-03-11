# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward function implementations for Dodo biped locomotion.

All threshold and parameter values are centralized in dodo_manage_cfg_constants.py
for easy tuning and reproducibility.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaac_dodo.tasks.manager_based.dodo_manage.mdp.observations as obs
from isaac_dodo.tasks.manager_based.dodo_manage.dodo_manage_cfg_constants import (
    FORCE_THRESHOLDS, REWARD_CONFIG
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    
    Encourages the robot to keep its body aligned with the gravity vector.
    
    Args:
        env: The environment
        threshold: Projection threshold for upright bonus
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding when in contact with ground.

    Penalizes horizontal movement of feet during contact phase, encouraging stable stance.
    
    Args:
        env: The environment
        sensor_cfg: Configuration for the contact sensor
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Penalty tensor of shape (num_envs,)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > FORCE_THRESHOLDS["slide"]
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def action_rate_l2(env) -> torch.Tensor:
    """L2 penalty on action rate: ||a_t - a_{t-1}||^2.
    
    Encourages smooth, continuous control without abrupt action changes.
    
    Args:
        env: The environment
        
    Returns:
        Penalty tensor of shape (num_envs,)
    """
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    rate = torch.sum((a - a_prev) ** 2, dim=-1)
    return rate


def single_support_reward(env, sensor_cfg: SceneEntityCfg, force_threshold: float = None) -> torch.Tensor:
    """Reward when exactly one foot is in contact.
    
    Encourages single-leg stance phase during walking gait.
    
    Args:
        env: The environment
        sensor_cfg: Configuration for the contact sensor
        force_threshold: Minimum force to detect contact (defaults to stance threshold)
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if force_threshold is None:
        force_threshold = FORCE_THRESHOLDS["stance"]
        
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w
    force_mag = torch.linalg.norm(forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    in_contact = force_mag > force_threshold

    left = in_contact[:, 0]
    right = in_contact[:, 1]

    single = torch.logical_xor(left, right)
    double_support = torch.logical_and(left, right)
    flight = torch.logical_not(torch.logical_or(left, right))

    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (torch.norm(cmd, dim=1) > 0.1).float()

    # Keep this as one consolidated gait term:
    # + reward single-support phase,
    # - penalize persistent double-support (stiff two-leg glide),
    # - slight penalty on full-flight for stability.
    reward = 0.8 * single.float() - 1.2 * double_support.float() - 0.1 * flight.float()
    return reward * moving

class alternate_footstep_reward(ManagerTermBase):
    """Reward alternating foot contacts: L then R then L ..."""

    def __init__(self, env, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_contact = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)  # [N,2]
        self.last_step_is_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int8)  # 1=left, -1=right, 0=none

    def reset(self, env_ids: torch.Tensor):
        self.prev_contact[env_ids] = False
        self.last_step_is_left[env_ids] = 0

    def __call__(self, env, sensor_cfg: SceneEntityCfg, force_threshold: float = 15.0, command_name: str = "base_velocity"):
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]          # [N,2,3]
        in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold      # [N,2] bool

        # detect rising edge: was False, now True  (foot touchdown)
        touchdown = (~self.prev_contact) & in_contact                           # [N,2]
        self.prev_contact = in_contact

        left_td = touchdown[:, 0]
        right_td = touchdown[:, 1]

        # ignore when both touch down simultaneously
        valid = left_td ^ right_td  # exactly one foot touchdown

        # which foot touched down this step
        step_is_left = torch.where(left_td, torch.ones_like(self.last_step_is_left), -torch.ones_like(self.last_step_is_left))

        # reward if alternates vs last step
        alternated = valid & (self.last_step_is_left != 0) & (step_is_left != self.last_step_is_left)
        repeated  = valid & (self.last_step_is_left != 0) & (step_is_left == self.last_step_is_left)

        # update memory when valid touchdown
        self.last_step_is_left = torch.where(valid, step_is_left, self.last_step_is_left)

        # gate by command speed (only when should be walking)
        cmd = env.command_manager.get_command(command_name)[:, :2]
        moving = torch.norm(cmd, dim=1) > 0.1

        rew_sparse = alternated.float() - 0.5 * repeated.float()

        # Dense phase shaping to avoid sparse-touchdown learning stall.
        single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
        double_support = torch.logical_and(in_contact[:, 0], in_contact[:, 1]).float()
        rew_dense = 0.12 * single_support - 0.08 * double_support

        return (rew_sparse + rew_dense) * moving.float()

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def feet_lateral_separation_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_sep: float = 0.14,
    std: float = 0.05,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Encourage reasonable lateral distance between feet during stance.
    
    Uses Gaussian reward around target separation distance.
    Only active when at least one foot is in contact.
    
    Args:
        env: The environment
        asset_cfg: Configuration for the robot asset
        sensor_cfg: Configuration for the contact sensor
        target_sep: Target foot separation (meters)
        std: Standard deviation for Gaussian reward
        force_threshold: Minimum force to detect contact
        
    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold
    stance_any = torch.any(in_contact, dim=-1)

    sep = torch.abs(feet_pos[:, 0, 1] - feet_pos[:, 1, 1])
    r = torch.exp(-((sep - target_sep) ** 2) / (std ** 2))

    return r * stance_any.float()

def yaw_rate_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize angular velocity around z-axis (yaw).
    
    Encourages forward-facing locomotion without spinning.
    
    Args:
        env: The environment
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2] ** 2

def lin_vel_y_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral (sideways) velocity.
    
    Encourages forward motion without lateral drift.
    
    Args:
        env: The environment
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 1] ** 2


def lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize vertical velocity to suppress pogo-style hopping."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2] ** 2


def feet_clearance_reward(
    env, asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    force_threshold: float = 5.0,
):
    robot = env.scene[asset_cfg.name]

    feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_z = feet_pos_w[..., 2]  # [num_envs, 2]

    # infer contact from forces
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w
    force_mag = torch.linalg.norm(forces_w[:, sensor_cfg.body_ids, :], dim=-1)  # [num_envs, 2]
    in_contact = force_mag > force_threshold

    # Only reward clearance when foot is NOT in contact (swing foot)
    swing = ~in_contact

    # hinge reward up to target_height
    z_clamped = torch.clamp(foot_z, 0.0, target_height) / target_height

    # mask: stance foot gets 0 reward
    reward = torch.sum(z_clamped * swing.float(), dim=-1)
    return reward

def knee_flexion_target_exp(
    env,
    asset_cfg: SceneEntityCfg,
    knee_target: float = 0.8,   # 目标屈膝角度（你先从 0.6~1.0 试）
    std: float = 0.3,
):
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, num_selected]
    err = q - knee_target
    # 每个env对选中的膝关节做平均
    return torch.exp(-torch.mean(err * err, dim=-1) / (std * std))

def swing_knee_flexion_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    knee_cfg: SceneEntityCfg,
    knee_target: float = 0.9,
    std: float = 0.3,
    force_threshold: float = 10.0,
):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]   # [N,2,3]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold  # [N,2]
    swing = ~in_contact  # [N,2]

    asset: Articulation = env.scene[knee_cfg.name]
    qk = asset.data.joint_pos[:, knee_cfg.joint_ids]  # [N,2] 选中左右膝

    # 只在 swing 的那条腿上算奖励
    err = qk - knee_target
    r = torch.exp(-(err * err) / (std * std))  # [N,2]
    r = torch.sum(r * swing.float(), dim=-1)   # [N]

    return r

def hip_swing_amplitude_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target: float = 0.30,
    max_amp: float = 0.70,
    force_threshold: float = 10.0,
):
    asset: Articulation = env.scene[asset_cfg.name]

    # hip angles [N,2]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if q.dim() != 2 or q.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    if qd.dim() != 2 or qd.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    amp = torch.clamp(torch.abs(q), max=max_amp)
    amp_r = torch.mean(torch.clamp(amp / target, max=1.0), dim=-1)  # [N]

    vel_mag = torch.mean(torch.abs(qd), dim=-1)
    vel_gate = torch.clamp(vel_mag / 0.5, max=1.0)

    r = amp_r * vel_gate

    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = torch.norm(cmd, dim=1) > 0.1

    return r * moving.float()


def hip_antiphase_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    std: float = 0.2,
    force_threshold: float = 10.0,
):
    """Reward opposite-phase hip motion to encourage alternating gait.

    The reward is highest when the selected left/right hip pair has similar
    magnitude with opposite sign, i.e. q_left + q_right ~= 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    if q.dim() != 2 or q.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if qd.dim() != 2 or qd.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    phase_error = q[:, 0] + q[:, 1]
    antiphase_r = torch.exp(-(phase_error * phase_error) / (std * std))

    # Velocity anti-phase: encourage opposite hip angular velocities.
    # If both hips move in the same direction, this term drops.
    vel_prod = qd[:, 0] * qd[:, 1]
    vel_antiphase = torch.sigmoid(-6.0 * vel_prod)
    vel_mag = 0.5 * (torch.abs(qd[:, 0]) + torch.abs(qd[:, 1]))
    vel_gate = torch.clamp(vel_mag / 0.6, max=1.0)

    # Amplitude gate: when both hips are near zero (default posture), reward is 0.
    # This prevents the degenerate solution where the robot stands still and gets
    # max antiphase reward because q_L + q_R = 0 + 0 = 0.
    amplitude = torch.mean(torch.abs(q), dim=-1)
    amp_gate = torch.clamp(amplitude / 0.15, max=1.0)  # ramps from 0 at hip=0 to 1.0 at hip>=0.15 rad

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold
    single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()

    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > 0.1).float()

    # Mostly gate to single-support but keep a tiny dense signal for exploration.
    phase_gate = 0.1 + 0.9 * single_support
    return antiphase_r * vel_antiphase * amp_gate * vel_gate * moving * phase_gate


def hip_velocity_antiphase_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    force_threshold: float = 10.0,
):
    """Dense reward for opposite hip angular velocities during gait.

    This term directly rewards qd_left * qd_right < 0 while suppressing
    static near-zero motion and double-support exploitation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if qd.dim() != 2 or qd.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold
    single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()

    vel_prod = qd[:, 0] * qd[:, 1]
    anti = torch.sigmoid(-8.0 * vel_prod)
    vel_mag = 0.5 * (torch.abs(qd[:, 0]) + torch.abs(qd[:, 1]))
    vel_gate = torch.clamp(vel_mag / 0.8, max=1.0)

    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > 0.1).float()

    phase_gate = 0.05 + 0.95 * single_support
    return anti * vel_gate * moving * phase_gate


def symmetry_amplitude_reward(
    env,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
):
    """Encourage left/right leg joint amplitudes to stay balanced.

    Uses absolute joint amplitudes so opposite motion directions are allowed,
    while discouraging one-sided gait dominance.
    """
    asset: Articulation = env.scene[left_cfg.name]

    q = asset.data.joint_pos
    q_left = q[:, left_cfg.joint_ids]
    q_right = q[:, right_cfg.joint_ids]

    if q_left.dim() != 2 or q_right.dim() != 2:
        return torch.zeros(env.num_envs, device=env.device)

    n = min(q_left.size(1), q_right.size(1))
    if n == 0:
        return torch.zeros(env.num_envs, device=env.device)

    # Compare absolute amplitudes for each left/right pair.
    amp_diff = torch.abs(torch.abs(q_left[:, :n]) - torch.abs(q_right[:, :n]))
    penalty = torch.mean(amp_diff, dim=-1)

    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > 0.1).float()

    return -penalty * moving
