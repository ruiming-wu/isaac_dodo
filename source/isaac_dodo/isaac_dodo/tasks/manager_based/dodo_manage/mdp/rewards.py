# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaac_dodo.tasks.manager_based.dodo_manage.mdp.observations as obs
from isaac_dodo.tasks.manager_based.dodo_manage.dodo_manage_cfg_constants import (
    FORCE_THRESHOLDS,
    REWARD_CONFIG,
    SCENE_CONFIG,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def pitch_stability_bonus(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    return torch.exp(-(pitch * pitch) / (std * std))


def pitch_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 1] ** 2


def stance_stability_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    pitch_std: float = 0.22,
    height_std: float = 0.05,
    force_threshold: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    pitch_reward = torch.exp(-(pitch * pitch) / (pitch_std * pitch_std))
    base_height = asset.data.root_pos_w[:, 2]
    height_error = base_height - 0.60
    height_reward = torch.exp(-(height_error * height_error) / (height_std * height_std))
    forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold
    stance_any = torch.any(in_contact, dim=-1).float()
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return pitch_reward * height_reward * stance_any * moving


def torso_height_target_reward(
    env,
    target_height: float = 0.52,
    std: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    height_error = base_height - target_height
    reward = torch.exp(-(height_error * height_error) / (std * std))
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return reward * moving


def phase_reference_reward(
    env,
    hip_cfg: SceneEntityCfg,
    knee_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    phase_period: float = 0.72,
    hip_amplitude: float = 0.38,
    hip_std: float = 0.22,
    knee_stance: float = -0.22,
    knee_swing_amp: float = 0.48,
    knee_std: float = 0.20,
) -> torch.Tensor:
    asset: Articulation = env.scene[hip_cfg.name]
    sim_step = SCENE_CONFIG["sim_dt"] * SCENE_CONFIG["decimation"]
    time_s = env.episode_length_buf.float() * sim_step
    # This is an explicit sinusoidal gait prior: RL is free to deviate, but gets rewarded
    # for staying near a simple alternating hip/knee pattern.
    phase = (2.0 * torch.pi * time_s) / phase_period
    desired_hips = torch.stack(
        (hip_amplitude * torch.sin(phase), hip_amplitude * torch.sin(phase + torch.pi)), dim=-1
    )
    left_swing = torch.clamp(torch.sin(phase), min=0.0)
    right_swing = torch.clamp(torch.sin(phase + torch.pi), min=0.0)
    desired_knees = torch.stack(
        (knee_stance - knee_swing_amp * left_swing, knee_stance - knee_swing_amp * right_swing), dim=-1
    )
    hips = asset.data.joint_pos[:, hip_cfg.joint_ids]
    knees = asset.data.joint_pos[:, knee_cfg.joint_ids]
    hip_error = torch.mean(torch.square(hips - desired_hips), dim=-1)
    knee_error = torch.mean(torch.square(knees - desired_knees), dim=-1)
    hip_reward = torch.exp(-hip_error / (hip_std * hip_std))
    knee_reward = torch.exp(-knee_error / (knee_std * knee_std))
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return hip_reward * knee_reward * moving


def hip_phase_reference_reward(
    env,
    hip_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    phase_period: float = 0.72,
    hip_amplitude: float = 0.45,
    std: float = 0.14,
) -> torch.Tensor:
    asset: Articulation = env.scene[hip_cfg.name]
    sim_step = SCENE_CONFIG["sim_dt"] * SCENE_CONFIG["decimation"]
    time_s = env.episode_length_buf.float() * sim_step
    # A lighter version of the phase prior that only constrains the hip pair.
    phase = (2.0 * torch.pi * time_s) / phase_period
    desired_hips = torch.stack(
        (hip_amplitude * torch.sin(phase), hip_amplitude * torch.sin(phase + torch.pi)), dim=-1
    )
    hips = asset.data.joint_pos[:, hip_cfg.joint_ids]
    hip_error = torch.mean(torch.square(hips - desired_hips), dim=-1)
    reward = torch.exp(-hip_error / (std * std))
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return reward * moving


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > FORCE_THRESHOLDS["slide"]
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def action_rate_l2(env) -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)


def single_support_reward(env, sensor_cfg: SceneEntityCfg, force_threshold: float = None) -> torch.Tensor:
    if force_threshold is None:
        force_threshold = FORCE_THRESHOLDS["stance"]
    sensor = env.scene.sensors[sensor_cfg.name]
    force_mag = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    in_contact = force_mag > force_threshold
    left = in_contact[:, 0]
    right = in_contact[:, 1]
    single = torch.logical_xor(left, right)
    double_support = torch.logical_and(left, right)
    flight = torch.logical_not(torch.logical_or(left, right))
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    reward = 0.5 * single.float() - 0.15 * double_support.float() - 0.01 * flight.float()
    return reward * moving


class alternate_footstep_reward(ManagerTermBase):
    def __init__(self, env, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_contact = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
        self.last_step_is_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int8)

    def reset(self, env_ids: torch.Tensor):
        self.prev_contact[env_ids] = False
        self.last_step_is_left[env_ids] = 0

    def __call__(self, env, sensor_cfg: SceneEntityCfg, force_threshold: float = 15.0, command_name: str = "base_velocity"):
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold
        touchdown = (~self.prev_contact) & in_contact
        self.prev_contact = in_contact
        left_td = touchdown[:, 0]
        right_td = touchdown[:, 1]
        valid = left_td ^ right_td
        step_is_left = torch.where(left_td, torch.ones_like(self.last_step_is_left), -torch.ones_like(self.last_step_is_left))
        alternated = valid & (self.last_step_is_left != 0) & (step_is_left != self.last_step_is_left)
        repeated = valid & (self.last_step_is_left != 0) & (step_is_left == self.last_step_is_left)
        self.last_step_is_left = torch.where(valid, step_is_left, self.last_step_is_left)
        cmd = env.command_manager.get_command(command_name)[:, :2]
        moving = torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]
        rew_sparse = alternated.float() - 0.5 * repeated.float()
        single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
        double_support = torch.logical_and(in_contact[:, 0], in_contact[:, 1]).float()
        rew_dense = 0.05 * single_support - 0.01 * double_support
        return (rew_sparse + rew_dense) * moving.float()


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def swing_foot_forward_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target: float = 0.09,
    std: float = 0.06,
    force_threshold: float = 6.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w[:, :3].unsqueeze(1)
    yaw_q = yaw_quat(asset.data.root_quat_w)
    yaw_q_rep = yaw_q.unsqueeze(1).expand(-1, feet_pos_w.shape[1], -1).reshape(-1, 4)
    feet_rel = (feet_pos_w - base_pos_w).reshape(-1, 3)
    feet_rel_b = quat_apply_inverse(yaw_q_rep, feet_rel).reshape(feet_pos_w.shape[0], feet_pos_w.shape[1], 3)
    # Measure swing-foot placement in a yaw-aligned body frame so "forward" follows the robot,
    # not the world x-axis.
    forward = feet_rel_b[..., 0]
    in_contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > force_threshold
    swing = ~in_contact
    error = (forward - target) ** 2
    reward = torch.exp(-error / (std * std)) * swing.float()
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return torch.sum(reward, dim=-1) * moving


def yaw_rate_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2] ** 2


def lin_vel_y_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 1] ** 2


def lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2] ** 2


def knee_flexion_target_exp(env, asset_cfg: SceneEntityCfg, knee_target: float = 0.8, std: float = 0.3):
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    err = q - knee_target
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
    in_contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > force_threshold
    swing = ~in_contact
    asset: Articulation = env.scene[knee_cfg.name]
    qk = asset.data.joint_pos[:, knee_cfg.joint_ids]
    err = qk - knee_target
    r = torch.exp(-(err * err) / (std * std))
    return torch.sum(r * swing.float(), dim=-1)


def knee_symmetry_reward(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity", std: float = 0.18) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff = torch.abs(torch.abs(q[:, 0]) - torch.abs(q[:, 1]))
    reward = torch.exp(-(diff * diff) / (std * std))
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return reward * moving


def leg_phase_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    knee_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    knee_delta_target: float = 0.22,
    std: float = 0.16,
    force_threshold: float = 6.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > force_threshold
    knees = env.scene[knee_cfg.name].data.joint_pos[:, knee_cfg.joint_ids]
    hips = env.scene[hip_cfg.name].data.joint_pos[:, hip_cfg.joint_ids]
    # When one leg swings and the other supports, reward a knee bend difference together with
    # opposite-signed hip motion.
    left_swing = (~in_contact[:, 0]) & in_contact[:, 1]
    right_swing = (~in_contact[:, 1]) & in_contact[:, 0]
    knee_delta = torch.where(left_swing, knees[:, 0] - knees[:, 1], torch.where(right_swing, knees[:, 1] - knees[:, 0], torch.zeros_like(knees[:, 0])))
    hip_error = hips[:, 0] + hips[:, 1]
    reward = torch.exp(-((knee_delta - knee_delta_target) ** 2) / (std * std)) * torch.exp(-(hip_error * hip_error) / (std * std))
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return reward * moving


def hip_antiphase_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    std: float = 0.2,
    force_threshold: float = 10.0,
):
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if q.dim() != 2 or q.size(1) < 2 or qd.dim() != 2 or qd.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    phase_error = q[:, 0] + q[:, 1]
    antiphase_r = torch.exp(-(phase_error * phase_error) / (std * std))
    vel_prod = qd[:, 0] * qd[:, 1]
    vel_antiphase = torch.sigmoid(-6.0 * vel_prod)
    vel_mag = 0.5 * (torch.abs(qd[:, 0]) + torch.abs(qd[:, 1]))
    vel_gate = torch.clamp(vel_mag / 0.6, max=1.0)
    amplitude = torch.mean(torch.abs(q), dim=-1)
    amp_gate = torch.clamp(amplitude / 0.15, max=1.0)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > force_threshold
    single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    phase_gate = 0.1 + 0.9 * single_support
    return antiphase_r * vel_antiphase * amp_gate * vel_gate * moving * phase_gate


def hip_velocity_antiphase_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    force_threshold: float = 10.0,
):
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if qd.dim() != 2 or qd.size(1) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > force_threshold
    single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
    vel_prod = qd[:, 0] * qd[:, 1]
    anti = torch.sigmoid(-8.0 * vel_prod)
    vel_mag = 0.5 * (torch.abs(qd[:, 0]) + torch.abs(qd[:, 1]))
    vel_gate = torch.clamp(vel_mag / 0.8, max=1.0)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    phase_gate = 0.05 + 0.95 * single_support
    return anti * vel_gate * moving * phase_gate


def symmetry_amplitude_reward(env, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg, command_name: str = "base_velocity"):
    asset: Articulation = env.scene[left_cfg.name]
    q = asset.data.joint_pos
    q_left = q[:, left_cfg.joint_ids]
    q_right = q[:, right_cfg.joint_ids]
    if q_left.dim() != 2 or q_right.dim() != 2:
        return torch.zeros(env.num_envs, device=env.device)
    n = min(q_left.size(1), q_right.size(1))
    if n == 0:
        return torch.zeros(env.num_envs, device=env.device)
    amp_diff = torch.abs(torch.abs(q_left[:, :n]) - torch.abs(q_right[:, :n]))
    penalty = torch.mean(amp_diff, dim=-1)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return -penalty * moving
