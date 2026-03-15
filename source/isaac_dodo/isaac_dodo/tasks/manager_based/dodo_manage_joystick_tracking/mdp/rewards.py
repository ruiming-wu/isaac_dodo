# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaac_dodo.tasks.manager_based.dodo_manage_joystick_tracking.mdp.observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _planar_body_velocities(asset: Articulation) -> torch.Tensor:
    """将世界系线速度投影到机器人偏航坐标系。"""
    heading_quat_w = yaw_quat(asset.data.root_quat_w)
    return quat_apply_inverse(heading_quat_w, asset.data.root_lin_vel_w)


def _body_ang_velocities(asset: Articulation) -> torch.Tensor:
    """将世界系角速度转到机体系。"""
    return quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)


def _moving_command_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    moving_lin = torch.abs(command[:, 0]) > 0.03
    moving_yaw = torch.abs(command[:, 2]) > 0.10
    return moving_lin | moving_yaw


def _curriculum_progress(
    env: ManagerBasedRLEnv,
    start_step: int = 2_000,
    end_step: int = 30_000,
) -> float:
    """根据训练步数返回 [0, 1] 的课程学习进度。"""
    step_count = float(getattr(env, "common_step_counter", 0))
    if step_count <= start_step:
        return 0.0
    if step_count >= end_step:
        return 1.0
    return (step_count - start_step) / max(end_step - start_step, 1)


def _scheduled_tolerance(
    env: ManagerBasedRLEnv,
    initial_std: float,
    final_std: float,
    start_step: int = 2_000,
    end_step: int = 30_000,
) -> float:
    progress = _curriculum_progress(env, start_step, end_step)
    return initial_std + progress * (final_std - initial_std)


def _scheduled_bonus_scale(
    env: ManagerBasedRLEnv,
    initial_scale: float,
    final_scale: float,
    start_step: int = 2_000,
    end_step: int = 30_000,
) -> float:
    progress = _curriculum_progress(env, start_step, end_step)
    return initial_scale + progress * (final_scale - initial_scale)

def upright_reward(
    env: ManagerBasedRLEnv, 
    std: float=0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励保持直立姿态。"""
    roll, pitch, yaw = obs.base_roll_pitch_yaw(env, asset_cfg).unbind(dim=-1)
    orientation_error = torch.square(roll) + torch.square(pitch)
    return torch.exp(-orientation_error / std**2) # maximum reward is 1.0

def height_reward(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    std: float=0.1, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励接近目标高度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    height_error = base_height - target_height
    return torch.exp(-torch.square(height_error) / std**2) # maximum reward is 1.0

def linear_velocity_tracking_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float=0.1, 
    final_std: float | None = None,
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 2_000,
    curriculum_end_step: int = 30_000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励基座偏航坐标系下的平面线速度跟踪。"""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = _planar_body_velocities(asset)[:, :2]
    command = env.command_manager.get_command(command_name)[:, :2]
    vel_error = torch.sum(torch.square(command - vel), dim=1)
    current_std = _scheduled_tolerance(env, std, final_std if final_std is not None else std, curriculum_start_step, curriculum_end_step)
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    return current_scale * torch.exp(-vel_error / current_std**2)

def angular_velocity_tracking_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float=0.1, 
    final_std: float | None = None,
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 2_000,
    curriculum_end_step: int = 30_000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机体系偏航角速度跟踪。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = _body_ang_velocities(asset)[:, 2]
    command = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error = torch.square(command - ang_vel)
    current_std = _scheduled_tolerance(env, std, final_std if final_std is not None else std, curriculum_start_step, curriculum_end_step)
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    return current_scale * torch.exp(-ang_vel_error / current_std**2)

def feet_air_time_reward(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    command_name: str = "base_velocity",
    std: float = 0.12,
    final_std: float | None = None,
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 2_000,
    curriculum_end_step: int = 30_000,
) -> torch.Tensor:
    """奖励更接近人形步态的摆脚空中时间。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [num_envs, 2]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # [num_envs, 2]
    in_contact = contact_time > 0.0  # [num_envs, 2]

    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    swing_air_time = torch.where(in_contact, torch.zeros_like(air_time), air_time)
    swing_air_time = torch.max(swing_air_time, dim=1)[0]
    swing_air_time = torch.clamp(swing_air_time, max=0.6)

    command = env.command_manager.get_command(command_name)
    target_air_time = 0.18 + 0.75 * torch.abs(command[:, 0]) + 0.08 * torch.abs(command[:, 2])
    target_air_time = torch.clamp(target_air_time, min=0.18, max=0.42)
    moving_mask = _moving_command_mask(env, command_name).float()

    current_std = _scheduled_tolerance(env, std, final_std if final_std is not None else std, curriculum_start_step, curriculum_end_step)
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    reward = torch.exp(-torch.square(swing_air_time - target_air_time) / current_std**2)
    return current_scale * reward * single_stance.float() * moving_mask

def feet_swing_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.15,
    std: float = 0.1,
    final_std: float | None = None,
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 2_000,
    curriculum_end_step: int = 30_000,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """摆动阶段脚保持目标高度。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    # 获取摆动状态
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_swing = (air_time > 0.05).float()  # [num_envs, 2]
    
    # 获取脚离地面的高度（z坐标，假设地面z=0）
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [num_envs, 2]
    
    # 奖励接近目标高度的摆动脚
    height_error = foot_height - target_height
    current_std = _scheduled_tolerance(env, std, final_std if final_std is not None else std, curriculum_start_step, curriculum_end_step)
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    reward = in_swing * torch.exp(-height_error**2 / current_std**2)
    
    return current_scale * torch.mean(reward, dim=1)


def feet_clearance_reward(
    env: ManagerBasedRLEnv,
    min_height: float = 0.08,
    std: float = 0.03,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_link_4"),
) -> torch.Tensor:
    """奖励摆动脚有足够离地间隙，减少擦地滑行。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_swing = (air_time > 0.03).float()
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance_gap = torch.clamp(min_height - foot_height, min=0.0)
    reward = in_swing * torch.exp(-clearance_gap**2 / std**2)
    return torch.mean(reward, dim=1)


def gait_single_stance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    command_name: str = "base_velocity",
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 2_000,
    curriculum_end_step: int = 30_000,
) -> torch.Tensor:
    """鼓励行走时出现清晰的左右交替支撑。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = (torch.sum(in_contact.int(), dim=1) == 1).float()
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    return current_scale * single_stance * _moving_command_mask(env, command_name).float()


def gait_phase_symmetry_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    command_name: str = "base_velocity",
    std: float = 0.16,
    final_std: float | None = None,
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 2_000,
    curriculum_end_step: int = 30_000,
) -> torch.Tensor:
    """鼓励左右脚周期时间更接近，减少明显跛行。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    cycle_time = torch.clamp(air_time + contact_time, max=1.0)
    left_cycle = cycle_time[:, 0]
    right_cycle = cycle_time[:, 1]
    current_std = _scheduled_tolerance(env, std, final_std if final_std is not None else std, curriculum_start_step, curriculum_end_step)
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    reward = torch.exp(-torch.square(left_cycle - right_cycle) / current_std**2)
    return current_scale * reward * _moving_command_mask(env, command_name).float()


def gait_step_period_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    command_name: str = "base_velocity",
    target_step_period: float = 0.5,
    std: float = 0.18,
    final_std: float | None = None,
    bonus_scale: float = 1.0,
    final_bonus_scale: float | None = None,
    curriculum_start_step: int = 4_000,
    curriculum_end_step: int = 40_000,
) -> torch.Tensor:
    """鼓励左右换脚时间接近目标步长。

    这里把“单脚支撑持续时间”近似看作一步的时长，并鼓励其接近 0.5s。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    stance_duration = torch.where(in_contact, contact_time, torch.zeros_like(contact_time))
    stance_duration = torch.max(stance_duration, dim=1)[0]
    stance_duration = torch.clamp(stance_duration, max=1.0)

    current_std = _scheduled_tolerance(
        env,
        std,
        final_std if final_std is not None else std,
        curriculum_start_step,
        curriculum_end_step,
    )
    current_scale = _scheduled_bonus_scale(
        env,
        bonus_scale,
        final_bonus_scale if final_bonus_scale is not None else bonus_scale,
        curriculum_start_step,
        curriculum_end_step,
    )
    reward = torch.exp(-torch.square(stance_duration - target_step_period) / current_std**2)
    return current_scale * reward * single_stance.float() * _moving_command_mask(env, command_name).float()

def stance_high_velocity_penalty(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """支撑阶段脚应该低速。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    # 获取脚接触状态：contact_time > 0 表示着地
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = (contact_time > 0.0).float()  # [num_envs, 2]
    
    # 获取脚的速度（link_4 的速度）
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # [num_envs, 2, 2]
    foot_speed = torch.norm(foot_vel, dim=-1)  # [num_envs, 2]
    
    penalty = in_contact * (1.0 - torch.exp(-foot_speed**2 / std**2))
    
    return torch.mean(penalty, dim=1)  # 平均两只脚, maximum penalty is 1.0

def swing_high_force_penalty(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4")
) -> torch.Tensor:
    """摆动阶段脚应该无接触力。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取摆动状态：air_time > 0 表示离地
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_swing = (air_time > 0.0).float()  # [num_envs, 2]
    
    # 获取脚的接触力
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [num_envs, 2, 3]
    force_magnitude = torch.norm(contact_forces, dim=-1)  # [num_envs, 2]
    
    penalty = in_swing * (1.0 - torch.exp(-force_magnitude**2 / std**2))
    
    return torch.mean(penalty, dim=1) # average over both feet, maximum penalty is 1.0

def feet_slide_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚足部滑动。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    penalty = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return penalty # higher penalty for more sliding, no upper limit

def joint_acc_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚关节加速度过大。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(joint_acc), dim=1)
    return penalty

def joint_vel_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚动作变化过大。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(joint_vel), dim=1)
    return penalty

def joint_tau_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚关节力矩过大。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(joint_tau), dim=1)
    return penalty

def energy_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚能量消耗过大。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power = joint_tau * joint_vel  # instantaneous power per joint
    energy = torch.sum(torch.abs(power), dim=1)  # total energy consumption
    return energy

def linear_vel_z_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚垂直速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    v_z = asset.data.root_lin_vel_w[:, 2]
    return torch.square(v_z)

def angular_vel_xy_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚XY平面角速度（roll/pitch）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_ang_vel_w[:, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=1)

def joint_in_range_reward(
    env: ManagerBasedRLEnv, 
    ranges,
    joint_ids=None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励关节位置在指定范围内。"""
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve joint indices
    ids = joint_ids if joint_ids is not None else asset_cfg.joint_ids
    if ids is None:
        ids = slice(None)

    joint_pos = asset.data.joint_pos[:, ids]

    # normalize ranges to per-joint tensor
    if isinstance(ranges[0], (list, tuple)):
        ranges_tensor = torch.tensor(ranges, device=env.device, dtype=joint_pos.dtype)
    else:
        ranges_tensor = torch.tensor([ranges] * joint_pos.shape[1], device=env.device, dtype=joint_pos.dtype)

    lower_in_range = joint_pos >= ranges_tensor[None, :, 0]
    upper_in_range = joint_pos <= ranges_tensor[None, :, 1]

    in_range = lower_in_range & upper_in_range
    return torch.all(in_range, dim=1).float()




# def hip_soft_penalty(
#     env: ManagerBasedRLEnv, 
#     soft_limit: tuple[float, float], 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """惩罚髋关节位置越过软限制。"""
#     asset: Articulation = env.scene[asset_cfg.name]
#     hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
#     left_violation = -(hip_pos[:, 0] - soft_limit[0]).clip(max=0.0)
#     right_violation = -(hip_pos[:, 1] - soft_limit[1]).clip(max=0.0)
#     violations = torch.stack([left_violation, right_violation], dim=1)
#     return torch.sum(violations, dim=1)

# def knee_usage_reward(
#     env: ManagerBasedRLEnv, 
#     min_bend_angle: float = 0.2,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """奖励膝关节弯曲（摆动腿膝盖比支撑腿更弯曲）。"""
#     asset = env.scene[asset_cfg.name]
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     joint_pos = asset.data.joint_pos
#     joint_names = asset.joint_names

#     # 可靠获取左右脚索引
#     foot_names = [name.lower() for name in contact_sensor.body_names]
#     left_foot_idx = None
#     right_foot_idx = None
#     for i, name in enumerate(foot_names):
#         if "left" in name:
#             left_foot_idx = i
#         if "right" in name:
#             right_foot_idx = i
    
#     # 查找膝关节索引
#     left_knee_idx = None
#     right_knee_idx = None
#     for i, name in enumerate(joint_names):
#         if "left_joint_3" in name:
#             left_knee_idx = i
#         elif "right_joint_3" in name:
#             right_knee_idx = i
    
#     if left_knee_idx is None or right_knee_idx is None or left_foot_idx is None or right_foot_idx is None:
#         return torch.zeros(env.num_envs, device=env.device)
    
#     left_knee_pos = joint_pos[:, left_knee_idx]
#     right_knee_pos = joint_pos[:, right_knee_idx]
    
#     # 检测哪只脚在空中
#     contact_time = contact_sensor.data.current_contact_time[:, [left_foot_idx, right_foot_idx]]
#     in_contact = contact_time > 0.0
#     left_in_contact = in_contact[:, 0]
#     right_in_contact = in_contact[:, 1]
    
#     single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
#     # 计算膝盖角度差奖励
#     left_swing_reward = torch.clamp(right_knee_pos - left_knee_pos - min_bend_angle, min=0.0, max=0.3) / 0.3
#     right_swing_reward = torch.clamp(left_knee_pos - right_knee_pos - min_bend_angle, min=0.0, max=0.3) / 0.3
    
#     knee_reward = torch.where(
#         ~left_in_contact & right_in_contact,
#         left_swing_reward,
#         torch.where(
#             left_in_contact & ~right_in_contact,
#             right_swing_reward,
#             torch.zeros_like(left_swing_reward)
#         )
#     )
    
#     knee_reward = knee_reward * single_stance.float()
#     return knee_reward
