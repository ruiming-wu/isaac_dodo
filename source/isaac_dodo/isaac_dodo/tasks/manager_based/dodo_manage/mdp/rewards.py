# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from isaaclab.utils.math import quat_apply_inverse, yaw_quat


import isaac_dodo.tasks.manager_based.dodo_manage.mdp.observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


def hip_pos_manual_limit(env: ManagerBasedRLEnv, softlimit: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # compute out of limits constraints
    left_violation = -(hip_pos[:, 0] - softlimit[0]).clip(max=0.0)
    right_violation = -(hip_pos[:, 1] - softlimit[1]).clip(max=0.0)
    violations = torch.stack([left_violation, right_violation], dim=1)
    return torch.sum(violations, dim=1)

def feet_air_time_positive_biped_snesor(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """鼓励双足机器人单脚支撑，另一脚摆动（需要配备传感器）"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_positive_biped(
        env, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    基于膝关节角度鼓励交替步态（单脚摆动）
    - 只有当一只脚摆动、另一只脚支撑时才给奖励
    - 双脚腾空或双脚支撑均无奖励
    """
    # 获取左右膝关节角度 [num_envs]
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    left_angle_hip = joint_pos[:, 0]
    right_angle_hip = joint_pos[:, 1]
    left_angle_knee = joint_pos[:, 2]
    right_angle_knee = joint_pos[:, 3]
    

    # 定义阶段阈值
    SWING_THRESHOLD_KNEE = 0.4  # 角度 > 此值 = 摆动（脚在空中）
    SWING_THRESHOLD_HIP = -0.2  # 角度 < 此值 = 摆动

    # 判断是否处于摆动阶段（脚在空中）
    left_swing_knee = (left_angle_knee > SWING_THRESHOLD_KNEE)   # [num_envs], bool
    right_swing_knee = (right_angle_knee > SWING_THRESHOLD_KNEE)
    left_swing_hip = (left_angle_hip < SWING_THRESHOLD_HIP)
    right_swing_hip = (right_angle_hip < SWING_THRESHOLD_HIP)

    # 计算每只脚的"有效摆动时间"（角度超出阈值的部分）
    left_air_time = torch.clamp(left_angle_knee, min=0.0)
    right_air_time = torch.clamp(right_angle_knee, min=0.0)

    # 只允许单腿摆动 (左摆 且 右不摆)  OR  (右摆 且 左不摆)
    # 检测单腿摆动状态
    left_single_swing = left_swing_knee & left_swing_hip & ~right_swing_knee & ~right_swing_hip
    right_single_swing = ~left_swing_knee & ~left_swing_hip & right_swing_knee & right_swing_hip
    is_single_swing = left_single_swing | right_single_swing

    # 获取命令速度并计算条件
    command_vel = env.command_manager.get_command(command_name)[:, :2]
    velocity_condition = torch.norm(command_vel, dim=1) > 0.1

    # 选择摆动脚的 air_time
    swing_air_time = torch.where(
        left_single_swing, 
        left_air_time, 
        torch.where(right_single_swing, right_air_time, torch.zeros_like(left_air_time))
    )

    # 计算基础奖励（强调两腿角度差异）
    base_reward = swing_air_time * 10 * torch.abs(left_angle_knee - right_angle_knee)

    # 检查膝关节角度差异是否显著，以避免轻微摆动。
    has_significant_knee_angle_diff = torch.abs(left_angle_knee - right_angle_knee) > 0.1

    # 检查走路姿势
    is_walk_pose = (left_angle_knee>0) & (left_angle_hip<0) & (right_angle_knee>0) & (right_angle_hip<0)

    # 根据条件应用奖励/惩罚
    reward = torch.where(
        is_single_swing & velocity_condition & has_significant_knee_angle_diff & is_walk_pose,
        base_reward,          # 单腿摆动+有速度命令 → 正奖励
        -base_reward   # 其他情况 → 惩罚
    )
    reward = torch.clamp(reward, max=threshold)
    return reward

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


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)
