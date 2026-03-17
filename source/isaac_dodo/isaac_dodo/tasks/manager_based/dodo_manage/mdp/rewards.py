# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
============================================================
DODO Manage 奖励函数模块
============================================================

本模块包含所有用于Dodo机器人行走训练的奖励函数。

## 奖励体系结构（三层次）

### 第一层：基础稳定性（权重: 5-10）
  - upright / roll_stability / pitch_stability: 身体姿态
  - torso_height_target: 躯干高度维持
  - stance_stability: 站立相位期间的稳定性
  
### 第二层：速度跟踪（权重: 1-2）
  - track_lin_vel: 前进速度匹配
  - roll_rate / pitch_rate: 平滑运动（避免抖动）
  
### 第三层：步态结构（权重: 0.5-2.8）← 通过Curriculum逐步提升
  - single_support: 单腿支撑（避免双支撑）
  - alternate_steps: 左右脚交替（核心步态）
  - swing_knee_contrast: 膝盖差异化控制
  - swing_clearance_balance: 两腿摆动高度平衡

## 奖励流程

1. 在early epochs: 强调稳定性和速度跟踪
   → 机器人学会基本平衡不摔跤
   
2. 在curriculum触发点（iter 180/260/360）: 逐步加强步态权重
   → 机器人学会从滑动到真正的步行（单腿支撑+交替）

3. 最终结果: 同时满足三层要求
   → 稳定不摔 + 跟随速度命令 + 清晰的左右交替走路

## 关键差异: 这个Dodo配置 vs 标准四足

- 8个关节（每条腿2个自由度）而不是标准的髋-膝-踝-脚结构
- 使用"行走"而不是"跑步"（速度0.1-0.28 m/s vs 通常0.8-1.5 m/s）
- 强调膝盖协调（swing_knee_contrast权重0.8）因为膝盖关节很关键

"""

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


def _moving_mask(env, command_name: str = "base_velocity") -> torch.Tensor:
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    return (torch.norm(cmd_xy, dim=1) > REWARD_CONFIG["gait_reward_gate_speed"]).float()


def _contact_force_mag(sensor: ContactSensor, body_ids) -> torch.Tensor:
    return torch.linalg.norm(sensor.data.net_forces_w[:, body_ids, :], dim=-1)


def _in_contact_mask(sensor: ContactSensor, body_ids, force_threshold: float) -> torch.Tensor:
    return _contact_force_mag(sensor, body_ids) > force_threshold


def _env_zeros(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _gait_phase(env, phase_period: float) -> torch.Tensor:
    sim_step = SCENE_CONFIG["sim_dt"] * SCENE_CONFIG["decimation"]
    time_s = env.episode_length_buf.float() * sim_step
    return (2.0 * torch.pi * time_s) / phase_period


# Reward a sufficiently upright torso using projected gravity/up direction.
def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


# Smoothly reward small torso pitch angles around zero.
def pitch_stability_bonus(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    return torch.exp(-(pitch * pitch) / (std * std))


def pitch_guard_l2_penalty(
    env: ManagerBasedRLEnv, threshold: float = 0.22, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    excess = torch.clamp(torch.abs(pitch) - threshold, min=0.0)
    return excess * excess


def height_floor_penalty(
    env: ManagerBasedRLEnv, target_height: float = 0.50, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Linear penalty proportional to how far torso has dropped below target height.
    Unlike Gaussian, this gives gradient signal at ALL heights below the target."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    deficit = torch.clamp(target_height - base_height, min=0.0)
    return deficit


# Smoothly reward small torso roll angles around zero.
def roll_stability_bonus(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    return torch.exp(-(roll * roll) / (std * std))


# Smoothly reward small absolute yaw angles around world-forward heading.
def yaw_stability_bonus(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.exp(-(yaw * yaw) / (std * std))


def tilt_xy_l2_penalty(
    env: ManagerBasedRLEnv,
    lateral_scale: float = 1.6,
    forward_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal components of projected gravity in body frame.

    This directly suppresses diagonal lean (e.g. right-forward body tilt)
    that can survive with only roll/pitch gaussian bonuses.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Project world gravity into body frame; horizontal components encode body tilt.
    g_proj = math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.GRAVITY_VEC_W)
    gx = g_proj[:, 0]
    gy = g_proj[:, 1]
    return lateral_scale * gx * gx + forward_scale * gy * gy


# Penalize fast pitching motion in the robot body frame.
def pitch_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 1] ** 2


# Penalize fast rolling motion in the robot body frame.
def roll_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 0] ** 2


# During locomotion, encourage the stance phase to keep the torso level and at a usable height.
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
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    stance_any = torch.any(in_contact, dim=-1).float()
    moving = _moving_mask(env)
    return pitch_reward * height_reward * stance_any * moving


# Keep the torso near a nominal walking height instead of crouching or pogoing too high.
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
    moving = _moving_mask(env)
    return reward * moving


# Follow an explicit sinusoidal reference for both hips and knees.
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
    # This is an explicit sinusoidal gait prior: RL is free to deviate, but gets rewarded
    # for staying near a simple alternating hip/knee pattern.
    phase = _gait_phase(env, phase_period)
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
    moving = _moving_mask(env, command_name)
    return hip_reward * knee_reward * moving


# A lighter phase prior that only constrains left/right hip alternation.
def hip_phase_reference_reward(
    env,
    hip_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    phase_period: float = 0.72,
    hip_amplitude: float = 0.45,
    std: float = 0.14,
) -> torch.Tensor:
    asset: Articulation = env.scene[hip_cfg.name]
    # A lighter version of the phase prior that only constrains the hip pair.
    phase = _gait_phase(env, phase_period)
    desired_hips = torch.stack(
        (hip_amplitude * torch.sin(phase), hip_amplitude * torch.sin(phase + torch.pi)), dim=-1
    )
    hips = asset.data.joint_pos[:, hip_cfg.joint_ids]
    hip_error = torch.mean(torch.square(hips - desired_hips), dim=-1)
    reward = torch.exp(-hip_error / (std * std))
    moving = _moving_mask(env, command_name)
    return reward * moving


def hip_swing_amplitude_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_amplitude: float = 0.22,
    std: float = 0.10,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    hips = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if hips.dim() != 2 or hips.size(1) < 2:
        return _env_zeros(env)

    # Use left/right hip separation as a proxy for instantaneous swing opening.
    # This measures whether the hips are actually spreading into an alternating gait,
    # instead of just sitting at a constant non-zero absolute angle.
    hip_opening = 0.5 * torch.abs(hips[:, 0] - hips[:, 1])
    amp_error = hip_opening - target_amplitude
    reward = torch.exp(-(amp_error * amp_error) / (std * std))

    # Keep some pressure against the trivial same-direction bias solution.
    phase_bias = torch.exp(-torch.square(hips[:, 0] + hips[:, 1]) / (std * std))
    moving = _moving_mask(env, command_name)
    return reward * phase_bias * moving


# Penalize foot sliding while a foot is in contact.
def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > FORCE_THRESHOLDS["slide"]
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


# Penalize large changes between consecutive actions.
def action_rate_l2(env) -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)


def hip_knee_motion_target_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_speed: float = 2.2,
    std: float = 0.9,
) -> torch.Tensor:
    """Reward moderate hip+knee joint speed to force visible leg swing.
    This avoids the local optimum where only distal joints jitter."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    speed = torch.mean(torch.abs(joint_vel), dim=-1)
    reward = torch.exp(-torch.square(speed - target_speed) / (std * std))
    moving = _moving_mask(env, command_name)
    return reward * moving


def ankle_shake_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize excessive ankle joint speed to suppress foot-end vibration propulsion."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.mean(torch.square(joint_vel), dim=-1)
    moving = _moving_mask(env, command_name)
    return penalty * moving


def ankle_pose_lock_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target: float = 0.0,
    std: float = 0.25,
) -> torch.Tensor:
    """Penalize ankle posture drifting far from neutral to avoid toe-driven propulsion."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = joint_pos - target
    penalty = torch.mean(torch.square(error), dim=-1) / (std * std)
    moving = _moving_mask(env, command_name)
    return penalty * moving


# ============ 步态核心奖励函数 ============
# Prefer exactly one foot in contact during walking, while softly discouraging double support.
def single_support_reward(env, sensor_cfg: SceneEntityCfg, force_threshold: float = None) -> torch.Tensor:
    """
    单腿支撑奖励：只有一条腿接地时获得奖励。
    
    作用: 推动交替步态的形成，避免双腿同时接地或飞行。
    
    规则:
    - 正好一条腿接地 (XOR): +0.7分
    - 两条腿都接地 (双支撑): -0.30分  
    - 没有腿接地 (飞行): -0.20分
    - 只在前进运动中计算
    
    参数:
        force_threshold: 判定接地的法向力阈值 (默认来自FORCE_THRESHOLDS["stance"])
    """
    if force_threshold is None:
        force_threshold = FORCE_THRESHOLDS["stance"]
    sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    left = in_contact[:, 0]
    right = in_contact[:, 1]
    single = torch.logical_xor(left, right)
    double_support = torch.logical_and(left, right)
    flight = torch.logical_not(torch.logical_or(left, right))
    moving = _moving_mask(env)
    reward = 0.7 * single.float() - 0.30 * double_support.float() - 0.20 * flight.float()
    return reward * moving


# ============ 步态交替控制函数 ============
# Reward alternating left/right touchdowns and add a light dense contact-timing bonus.
class alternate_footstep_reward(ManagerTermBase):
    """
    交替步态奖励：核心步态形成函数，奖励左右脚的严格交替。
    
    作用: 这是强制形成清晰交替步态的主要机制。
    
    有状态跟踪:
    - prev_contact: 上一时步各脚接触状态
    - last_step_is_left: 上一次触地是左脚(+1)还是右脚(-1)
    
    奖励规则 (只在运动中计算):
    - 发生交替触地 (left后right或反之): +1.0分 <- 核心激励
    - 发生重复触地 (同脚连续): -0.5分
    
    密集奖励 (辅助):
    - 单腿支撑期间: +0.05分
    - 双腿支撑期间: -0.01分
    
    关键参数:
        force_threshold: 判定触地的最小法向力 (default=15.0 N)
    """
    def __init__(self, env, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.last_step_is_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.int8)

    def reset(self, env_ids: torch.Tensor):
        self.last_step_is_left[env_ids] = 0

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        force_threshold: float = 15.0,
        command_name: str = "base_velocity",
        min_air_time: float = 0.08,
    ):
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
        step_dt = SCENE_CONFIG["sim_dt"] * SCENE_CONFIG["decimation"]
        touchdown = sensor.compute_first_contact(step_dt) & (sensor.data.last_air_time > min_air_time)
        left_td = touchdown[:, 0]
        right_td = touchdown[:, 1]
        valid = left_td ^ right_td
        step_is_left = torch.where(left_td, torch.ones_like(self.last_step_is_left), -torch.ones_like(self.last_step_is_left))
        alternated = valid & (self.last_step_is_left != 0) & (step_is_left != self.last_step_is_left)
        repeated = valid & (self.last_step_is_left != 0) & (step_is_left == self.last_step_is_left)
        self.last_step_is_left = torch.where(valid, step_is_left, self.last_step_is_left)
        moving = _moving_mask(env, command_name)
        rew_sparse = alternated.float() - 0.5 * repeated.float()
        single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
        double_support = torch.logical_and(in_contact[:, 0], in_contact[:, 1]).float()
        rew_dense = 0.05 * single_support - 0.01 * double_support
        return (rew_sparse + rew_dense) * moving


# Reward matching commanded planar velocity in a yaw-aligned body frame.
def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    vel_xy = vel_yaw[:, :2]

    lin_vel_error = torch.sum(torch.square(cmd_xy - vel_xy), dim=1)
    tracking_reward = torch.exp(-lin_vel_error / std**2)

    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    act_speed = torch.linalg.norm(vel_xy, dim=1)
    # Require meaningful forward motion when a locomotion command is active.
    # This suppresses the local optimum of standing still and collecting posture rewards.
    min_required_speed = torch.clamp(0.5 * cmd_speed, min=0.06)
    speed_gate = torch.clamp(act_speed / (min_required_speed + 1.0e-6), max=1.0)
    moving_cmd = (cmd_speed > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return tracking_reward * (moving_cmd * speed_gate + (1.0 - moving_cmd))


# Reward matching commanded planar velocity directly in world frame.
# Using world-frame tracking discourages policies from earning reward while constantly yawing.
def track_lin_vel_xy_world_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_xy = asset.data.root_lin_vel_w[:, :2]
    lin_vel_error = torch.sum(
        torch.square(cmd_xy - vel_xy), dim=1
    )
    tracking_reward = torch.exp(-lin_vel_error / std**2)

    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    act_speed = torch.linalg.norm(vel_xy, dim=1)
    min_required_speed = torch.clamp(0.5 * cmd_speed, min=0.06)
    speed_gate = torch.clamp(act_speed / (min_required_speed + 1.0e-6), max=1.0)
    moving_cmd = (cmd_speed > REWARD_CONFIG["gait_reward_gate_speed"]).float()
    return tracking_reward * (moving_cmd * speed_gate + (1.0 - moving_cmd))


# Reward matching the commanded yaw rate in world frame.
def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def no_progress_penalty(
    env,
    command_name: str = "base_velocity",
    speed_ratio: float = 0.60,
    min_speed: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_xy = asset.data.root_lin_vel_w[:, :2]

    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
    act_speed = torch.linalg.norm(vel_xy, dim=1)
    moving_cmd = (cmd_speed > REWARD_CONFIG["gait_reward_gate_speed"]).float()

    required_speed = torch.clamp(speed_ratio * cmd_speed, min=min_speed)
    shortfall = torch.clamp(required_speed - act_speed, min=0.0) / (required_speed + 1.0e-6)
    return shortfall * moving_cmd


# Reward the swing foot landing ahead of the base in the robot's forward frame.
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
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    swing = ~in_contact
    error = (forward - target) ** 2
    reward = torch.exp(-error / (std * std)) * swing.float()
    moving = _moving_mask(env, command_name)
    return torch.sum(reward, dim=-1) * moving


# During single support, reward the swing foot being clearly higher than the stance foot.
def swing_clearance_balance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_delta: float = 0.055,
    std: float = 0.03,
    force_threshold: float = 6.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_z = feet_pos_w[..., 2]

    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    left_swing = (~in_contact[:, 0]) & in_contact[:, 1]
    right_swing = (~in_contact[:, 1]) & in_contact[:, 0]
    single_support = left_swing | right_swing

    # Positive means right foot is higher than left foot.
    right_minus_left = feet_z[:, 1] - feet_z[:, 0]
    desired_delta = torch.where(
        right_swing,
        torch.full_like(right_minus_left, target_delta),
        torch.where(left_swing, torch.full_like(right_minus_left, -target_delta), torch.zeros_like(right_minus_left)),
    )
    err = right_minus_left - desired_delta
    reward = torch.exp(-(err * err) / (std * std)) * single_support.float()

    moving = _moving_mask(env, command_name)
    return reward * moving


def stance_load_balance_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    std: float = 0.18,
    force_threshold: float = 6.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = _contact_force_mag(sensor, sensor_cfg.body_ids)
    in_contact = force_mag > force_threshold
    double_support = torch.logical_and(in_contact[:, 0], in_contact[:, 1])

    # Compare stance load symmetry; persistent one-sided loading correlates with torso lean.
    left_load = force_mag[:, 0]
    right_load = force_mag[:, 1]
    load_diff = torch.abs(left_load - right_load) / (left_load + right_load + 1.0e-6)
    reward = torch.exp(-(load_diff * load_diff) / (std * std)) * double_support.float()

    moving = _moving_mask(env, command_name)
    return reward * moving


def right_load_bias_l2_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    force_threshold: float = 6.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = _contact_force_mag(sensor, sensor_cfg.body_ids)
    in_contact = force_mag > force_threshold
    double_support = torch.logical_and(in_contact[:, 0], in_contact[:, 1]).float()

    left_load = force_mag[:, 0]
    right_load = force_mag[:, 1]
    # Only penalize when right side carries more than left side.
    right_bias = torch.clamp(right_load - left_load, min=0.0) / (left_load + right_load + 1.0e-6)

    moving = _moving_mask(env, command_name)
    return (right_bias * right_bias) * double_support * moving


def right_swing_clearance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_delta: float = 0.065,
    std: float = 0.028,
    force_threshold: float = 6.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    right_swing = (~in_contact[:, 1]) & in_contact[:, 0]
    right_minus_left = feet_z[:, 1] - feet_z[:, 0]
    err = right_minus_left - target_delta
    reward = torch.exp(-(err * err) / (std * std)) * right_swing.float()

    moving = _moving_mask(env, command_name)
    return reward * moving


# Penalize unnecessary yaw spinning.
def yaw_rate_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2] ** 2


# Penalize sideways drift.
def lin_vel_y_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 1] ** 2


# Penalize vertical bouncing.
def lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2] ** 2


# Encourage both knees to stay near a nominal flexion target.
def knee_flexion_target_exp(env, asset_cfg: SceneEntityCfg, knee_target: float = 0.8, std: float = 0.3):
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    err = q - knee_target
    return torch.exp(-torch.mean(err * err, dim=-1) / (std * std))


# During swing, encourage the knee on that side to bend toward the swing target.
def swing_knee_flexion_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    knee_cfg: SceneEntityCfg,
    knee_target: float = 0.9,
    std: float = 0.3,
    force_threshold: float = 10.0,
):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    swing = ~in_contact
    asset: Articulation = env.scene[knee_cfg.name]
    qk = asset.data.joint_pos[:, knee_cfg.joint_ids]
    err = qk - knee_target
    r = torch.exp(-(err * err) / (std * std))
    return torch.sum(r * swing.float(), dim=-1)


# During single-support phase, reward the swing knee bending more than the stance knee.
def swing_knee_contrast_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    knee_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_delta: float = 0.20,
    std: float = 0.10,
    non_alternating_penalty: float = 0.0,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    left_swing = (~in_contact[:, 0]) & in_contact[:, 1]
    right_swing = (~in_contact[:, 1]) & in_contact[:, 0]
    alternating = left_swing | right_swing

    knees = env.scene[knee_cfg.name].data.joint_pos[:, knee_cfg.joint_ids]
    # Knee bend direction is negative in this model; positive delta means swing knee bends more.
    delta = torch.where(
        left_swing,
        knees[:, 1] - knees[:, 0],
        torch.where(right_swing, knees[:, 0] - knees[:, 1], torch.zeros_like(knees[:, 0])),
    )
    reward = torch.exp(-torch.square(delta - target_delta) / (std * std))
    phase_term = torch.where(alternating, reward, -non_alternating_penalty * torch.ones_like(reward))
    moving = _moving_mask(env, command_name)
    return phase_term * moving


# Encourage similar left/right knee bend magnitudes during locomotion.
def knee_symmetry_reward(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity", std: float = 0.18) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff = torch.abs(torch.abs(q[:, 0]) - torch.abs(q[:, 1]))
    reward = torch.exp(-(diff * diff) / (std * std))
    moving = _moving_mask(env, command_name)
    return reward * moving


# Reward a coordinated phase pattern: one knee bends more while the hips move oppositely.
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
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    knees = env.scene[knee_cfg.name].data.joint_pos[:, knee_cfg.joint_ids]
    hips = env.scene[hip_cfg.name].data.joint_pos[:, hip_cfg.joint_ids]
    # When one leg swings and the other supports, reward a knee bend difference together with
    # opposite-signed hip motion.
    left_swing = (~in_contact[:, 0]) & in_contact[:, 1]
    right_swing = (~in_contact[:, 1]) & in_contact[:, 0]
    knee_delta = torch.where(left_swing, knees[:, 0] - knees[:, 1], torch.where(right_swing, knees[:, 1] - knees[:, 0], torch.zeros_like(knees[:, 0])))
    hip_error = hips[:, 0] + hips[:, 1]
    reward = torch.exp(-((knee_delta - knee_delta_target) ** 2) / (std * std)) * torch.exp(-(hip_error * hip_error) / (std * std))
    moving = _moving_mask(env, command_name)
    return reward * moving


# Reward left/right hips being out of phase with sufficient amplitude, especially in single support.
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
        return _env_zeros(env)
    phase_error = q[:, 0] + q[:, 1]
    antiphase_r = torch.exp(-(phase_error * phase_error) / (std * std))
    vel_prod = qd[:, 0] * qd[:, 1]
    vel_antiphase = torch.sigmoid(-6.0 * vel_prod)
    vel_mag = 0.5 * (torch.abs(qd[:, 0]) + torch.abs(qd[:, 1]))
    vel_gate = torch.clamp(vel_mag / 0.6, max=1.0)
    amplitude = torch.mean(torch.abs(q), dim=-1)
    amp_gate = torch.clamp(amplitude / 0.15, max=1.0)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
    moving = _moving_mask(env, command_name)
    phase_gate = 0.1 + 0.9 * single_support
    return antiphase_r * vel_antiphase * amp_gate * vel_gate * moving * phase_gate


# Reward opposite-signed hip velocities so the legs keep alternating rather than moving together.
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
        return _env_zeros(env)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = _in_contact_mask(sensor, sensor_cfg.body_ids, force_threshold)
    single_support = torch.logical_xor(in_contact[:, 0], in_contact[:, 1]).float()
    vel_prod = qd[:, 0] * qd[:, 1]
    anti = torch.sigmoid(-8.0 * vel_prod)
    vel_mag = 0.5 * (torch.abs(qd[:, 0]) + torch.abs(qd[:, 1]))
    vel_gate = torch.clamp(vel_mag / 0.8, max=1.0)
    moving = _moving_mask(env, command_name)
    phase_gate = 0.05 + 0.95 * single_support
    return anti * vel_gate * moving * phase_gate


def symmetry_amplitude_reward(env, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg, command_name: str = "base_velocity"):
    asset: Articulation = env.scene[left_cfg.name]
    q = asset.data.joint_pos
    q_left = q[:, left_cfg.joint_ids]
    q_right = q[:, right_cfg.joint_ids]
    if q_left.dim() != 2 or q_right.dim() != 2:
        return _env_zeros(env)
    n = min(q_left.size(1), q_right.size(1))
    if n == 0:
        return _env_zeros(env)
    amp_diff = torch.abs(torch.abs(q_left[:, :n]) - torch.abs(q_right[:, :n]))
    penalty = torch.mean(amp_diff, dim=-1)
    moving = _moving_mask(env, command_name)
    return -penalty * moving


def bilateral_leg_participation_reward(
    env,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_activity: float = 1.0,
    activity_std: float = 0.4,
) -> torch.Tensor:
    asset: Articulation = env.scene[left_cfg.name]
    qd = asset.data.joint_vel
    qd_left = qd[:, left_cfg.joint_ids]
    qd_right = qd[:, right_cfg.joint_ids]
    if qd_left.dim() != 2 or qd_right.dim() != 2:
        return _env_zeros(env)

    left_activity = torch.mean(torch.abs(qd_left), dim=-1)
    right_activity = torch.mean(torch.abs(qd_right), dim=-1)
    total_activity = 0.5 * (left_activity + right_activity)

    # Both legs should stay active, and one leg should not dominate the other.
    balance = torch.minimum(left_activity, right_activity) / (torch.maximum(left_activity, right_activity) + 1.0e-6)
    activity_error = total_activity - target_activity
    activity_gate = torch.exp(-(activity_error * activity_error) / (activity_std * activity_std))

    moving = _moving_mask(env, command_name)
    return balance * activity_gate * moving
