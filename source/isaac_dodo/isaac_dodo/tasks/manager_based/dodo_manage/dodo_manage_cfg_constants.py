# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
============================================================
DODO Manage 中央常量配置
============================================================

本模块集中管理所有Dodo训练参数：

## 配置级别
1. SCENE_CONFIG: 物理模拟参数（时步、环境数等）
2. COMMAND_RANGES: 速度命令范围（由Curriculum修改）
3. REWARD_WEIGHTS: 各奖励项的系数（关键参数！）
4. REWARD_PARAMS: 具体奖励函数的参数（如目标高度、标准差等）
5. TERMINATION_CONFIG: 环境终止条件

## 快速参考：关键参数调整

### 如果机器人在"滑冰"（地面拖动）:
  ↑ 增加 feet_slide (-0.35)        # 更强烈地惩罚接触滑动
  ↓ 减少 ankle_pose_lock (-0.15)   # 解除对脚踝的锁定

### 如果机器人在"摔跤"（不稳定）:
  ↑ 增加 stance_stability (0.80)   # 在站立相强化稳定性
  ↑ 增加 pitch_stability (1.20)    # 强化前后平衡
  ↑ 增加 upright (0.78)            # 增加直立奖励

### 如果步态"不清晰"（不交替）:
  ↑ 增加 alternate_steps (2.00)    # 更强烈地奖励交替
  ↑ 增加 single_support (1.60)     # 更强烈地奖励单脚支撑
  ↑ 增加 swing_knee_contrast (0.80) # 膝盖差异更明显

## 三阶段Curriculum触发点

Stage 1 (iter 120):  command_speed (0.10, 0.22) m/s
Stage 2 (iter 180):  single_support weight 2.20
Stage 3 (iter 260):  alternate_steps weight 2.80
Stage 4 (iter 360):  swing_knee_contrast weight 1.40
Stage 5 (iter 420):  command_speed (0.12, 0.28) m/s ← 评测用速度

"""

from __future__ import annotations

SCENE_CONFIG = {
    "num_envs": 4096,
    "env_spacing": 5.0,
    "terrain_friction": 0.8,
    "sim_dt": 1 / 120.0,
    "decimation": 2,
    "episode_length_s": 16.0,
}

CONTROL_CONFIG = {
    "action_scale": 0.65,
    "use_default_offset": True,
}

OBSERVATION_SCALES = {
    "base_ang_vel": 0.25,
    "joint_vel_rel": 0.1,
    "feet_body_forces": 0.01,
}

COMMAND_RANGES = {
    "lin_vel_x": (0.12, 0.24),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (0.0, 0.0),
    "heading": (0.0, 0.0),
    "resampling_time": (4.0, 6.0),
}

RESET_RANGES = {
    "base_pos_x": (-0.5, 0.5),
    "base_pos_y": (-0.5, 0.5),
    "base_pos_z": (0.4, 0.6),
    "base_lin_vel_x": (-0.10, 0.10),
    "base_lin_vel_y": (-0.06, 0.06),
    "base_lin_vel_z": (-0.06, 0.06),
    "joint_pos_offset": (-0.03, 0.03),
    "joint_vel_offset": (-0.05, 0.05),
}

FORCE_THRESHOLDS = {
    "stance": 5.0,
    "swing": 3.0,
    "slide": 8.0,
}

TERMINATION_CONFIG = {
    "min_torso_height": 0.3,
    "max_tilt_angle": 0.60,
}

# Shared reward defaults used by multiple reward terms.
REWARD_CONTACT_FORCE_THRESHOLD = 6.0
REWARD_GAIT_GATE_SPEED = 0.06

REWARD_WEIGHTS = {
    "=== SURVIVAL & BASIC STABILITY ===": "...",
    "termination": -10.0,            # 摔倒或非法终止惩罚
    "action_l2": -0.004,             # 动作幅度正则化
    "action_rate": -0.025,           # 平滑动作变化
    "upright": 0.45,                 # 躯干竖直奖励（降低站桩吸引）
    "roll_stability": 0.55,          # 左右平衡（降低站桩吸引）
    "feet_slide": -0.55,             # 脚部滑动惩罚

    "=== GAIT STRUCTURE ===": "...",
    "alternate_steps": 1.40,         # 左右交替迈步（避免原地点腿刷分）
    "single_support": 1.20,          # 单脚支撑
    "swing_foot_forward": 1.5,      # 摆腿向前

    "=== LOCOMOTION GOAL ===": "...",
    "track_lin_vel": 2.60,           # 前进速度追踪（更强）
    "no_progress": -2.60,            # 不前进惩罚（更强，抑制原地点腿）

    "=== KNEE DIRECTION CONSTRAINT ===": "...",
    "knee_flex": 0.35,               # 约束膝关节朝后弯曲（本模型负向为屈曲）

    "=== ALL OTHER REWARDS DISABLED ===": "...",
    "pitch_stability": 0.0,
    "pitch_guard": 0.0,
    "yaw_stability": 0.0,
    "tilt_xy": 0.0,
    "roll_rate": 0.0,
    "pitch_rate": 0.0,
    "stance_stability": 0.0,
    "torso_height_target": 0.0,
    "height_floor": 0.0,
    "track_ang_vel": 0.0,
    "yaw_rate": 0.0,
    "lin_vel_y": 0.0,
    "lin_vel_z": 0.0,
    "stance_load_balance": 0.0,
    "right_load_bias": 0.0,
    "swing_clearance_balance": 0.0,
    "right_swing_clearance": 0.0,
    "swing_knee": 0.0,
    "swing_knee_contrast": 0.0,
    "knee_symmetry": 0.0,
    "leg_phase": 0.0,
    "phase_reference": 0.0,
    "hip_phase_reference": 0.0,
    "hip_swing_amplitude": 0.0,
    "hip_antiphase": 0.0,
    "hip_vel_antiphase": 0.0,
    "symmetry_amp": 0.0,
    "bilateral_leg_participation": 0.0,
    "hip_knee_motion": 0.0,
    "ankle_shake": 0.0,
    "ankle_pose_lock": 0.0,
}

REWARD_PARAMS = {
    # Stability-related reward parameters.
    "upright_threshold": 0.45,
    "roll_std": 0.24,
    "pitch_std": 0.18,
    "pitch_guard_threshold": 0.22,
    "yaw_std": 0.18,
    "tilt_xy_lateral_scale": 1.8,
    "tilt_xy_forward_scale": 1.0,
    "stance_pitch_std": 0.12,
    "stance_height_std": 0.05,
    "torso_height_target": 0.50,
    "torso_height_std": 0.06,
    # Velocity-tracking parameters.
    "lin_vel_std": 0.08,
    "ang_vel_std": 0.5,
    "no_progress_speed_ratio": 0.85,
    "no_progress_min_speed": 0.12,
    # Foot placement / swing-foot shaping parameters.
    "swing_forward_target": 0.14,
    "swing_forward_std": 0.07,
    "swing_forward_force_threshold": REWARD_CONTACT_FORCE_THRESHOLD,
    "swing_clearance_target_delta": 0.085,
    "swing_clearance_std": 0.045,
    "swing_clearance_force_threshold": REWARD_CONTACT_FORCE_THRESHOLD,
    "right_swing_target_delta": 0.065,
    "right_swing_std": 0.028,
    "right_swing_force_threshold": REWARD_CONTACT_FORCE_THRESHOLD,
    "stance_load_balance_std": 0.22,
    "stance_load_balance_force_threshold": REWARD_CONTACT_FORCE_THRESHOLD,
    "gait_reward_gate_speed": 0.05,
    # Knee-target and left/right coordination parameters.
    "knee_target": -0.45,
    "knee_std": 0.18,
    "swing_knee_target": -0.52,
    "swing_knee_std": 0.18,
    "swing_knee_force_threshold": REWARD_CONTACT_FORCE_THRESHOLD,
    "swing_knee_contrast_delta": 0.28,
    "swing_knee_contrast_std": 0.22,
    "swing_knee_non_alternating_penalty": 0.10,
    "knee_symmetry_std": 0.10,
    "leg_phase_knee_delta": 0.14,
    "leg_phase_std": 0.24,
    # Explicit sinusoidal gait-reference parameters.
    "phase_period": 0.78,
    "phase_hip_amplitude": 0.44,
    "phase_hip_std": 0.26,
    "hip_phase_std": 0.16,
    "alternate_min_air_time": 0.08,
    "hip_swing_target_amplitude": 0.20,
    "hip_swing_std": 0.12,
    "bilateral_leg_activity_target": 1.0,
    "bilateral_leg_activity_std": 0.35,
    "phase_knee_stance": -0.24,
    "phase_knee_swing_amp": 0.62,
    "phase_knee_std": 0.16,
    # Hip-specific coordination parameters.
    "hip_antiphase_std": 0.24,
    "hip_knee_speed_target": 1.1,
    "hip_knee_speed_std": 1.1,
    "ankle_pos_target": 0.0,
    "ankle_pos_std": 0.35,
}

# Backward-compatible aggregate used by existing config wiring.
REWARD_CONFIG = {"weights": REWARD_WEIGHTS, **REWARD_PARAMS}

JOINT_CONFIG = {
    "hip_joints": ["hip_left", "hip_right"],
    "knee_joints": ["upper_leg_left", "upper_leg_right"],
    "hip_knee_joints": ["hip_left", "hip_right", "upper_leg_left", "upper_leg_right"],
    "ankle_joints": ["lower_leg_left", "lower_leg_right", "foot_left", "foot_right"],
    "feet_bodies": ["foot_left", "foot_right"],
    "left_leg_joints": ["hip_left", "upper_leg_left", "lower_leg_left", "foot_left"],
    "right_leg_joints": ["hip_right", "upper_leg_right", "lower_leg_right", "foot_right"],
}


def get_force_threshold(context: str) -> float:
    return FORCE_THRESHOLDS.get(context, 15.0)


def get_reward_weight(reward_name: str) -> float:
    return REWARD_WEIGHTS.get(reward_name, 0.0)
