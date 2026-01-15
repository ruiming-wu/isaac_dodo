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


# def move_to_target_bonus(
#     env: ManagerBasedRLEnv,
#     threshold: float,
#     target_pos: tuple[float, float, float],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward for moving to the target heading."""
#     heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
#     return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


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

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

# 

def action_rate_l2(env):
    """
    L2 penalty on action rate: ||a_t - a_{t-1}||^2
    """
    # current action: [num_envs, action_dim]
    a = env.action_manager.action

    # previous action
    a_prev = env.action_manager.prev_action

    # squared L2 norm per env
    rate = torch.sum((a - a_prev) ** 2, dim=-1)

    return rate

def single_support_reward(env, sensor_cfg: SceneEntityCfg, force_threshold: float = 5.0):
    """Reward when exactly one foot is in contact."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w  # 如果你用的是这个字段
    force_mag = torch.linalg.norm(forces_w[:, sensor_cfg.body_ids, :], dim=-1)  # [N,2]
    in_contact = force_mag > force_threshold

    left = in_contact[:, 0]
    right = in_contact[:, 1]

    single = torch.logical_xor(left, right)   # exactly one True
    return single.float()

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

        rew = alternated.float() - 0.5 * repeated.float()
        return rew * moving.float()

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

#keep feet in reasonable lateral separation 
def feet_lateral_separation_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_sep: float = 0.14,     # 目标脚间距（米）先试 0.12~0.18
    std: float = 0.05,            # 宽容度（越小越严格）
    force_threshold: float = 10.0 # stance 判断阈值
):
    """Encourage a reasonable lateral distance between feet during stance."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # feet positions: [N,2,3]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    # contact mask: [N,2]
    forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces_w, dim=-1) > force_threshold

    # only enforce when at least one foot is in contact (usually walking)
    stance_any = torch.any(in_contact, dim=-1)  # [N]

    # lateral separation in world Y (abs)
    sep = torch.abs(feet_pos[:, 0, 1] - feet_pos[:, 1, 1])  # [N]

    # Gaussian reward around target
    r = torch.exp(-((sep - target_sep) ** 2) / (std ** 2))

    return r * stance_any.float()

# 


# Unused reward terms (currently disabled in env cfg)
# Keep here for future experiments.
# 
# class progress_reward(ManagerTermBase):
#     """Reward for making progress towards the target."""

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # initialize the base class
#         super().__init__(cfg, env)
#         # create history buffer
#         self.potentials = torch.zeros(env.num_envs, device=env.device)
#         self.prev_potentials = torch.zeros_like(self.potentials)

#     def reset(self, env_ids: torch.Tensor):
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = self._env.scene["robot"]
#         # compute projection of current heading to desired heading vector
#         target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
#         to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
#         # reward terms
#         self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
#         self.prev_potentials[env_ids] = self.potentials[env_ids]

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         target_pos: tuple[float, float, float],
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         # compute vector to target
#         target_pos = torch.tensor(target_pos, device=env.device)
#         to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
#         to_target_pos[:, 2] = 0.0
#         # update history buffer and compute new potential
#         self.prev_potentials[:] = self.potentials[:]
#         self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

#         return self.potentials - self.prev_potentials

# def feet_air_time_positive_biped(
#         env, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """
#     基于膝关节角度鼓励交替步态（单脚摆动）
#     - 只有当一只脚摆动、另一只脚支撑时才给奖励
#     - 双脚腾空或双脚支撑均无奖励
#     """
#     # 获取左右膝关节角度 [num_envs]
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
#     left_angle_hip = joint_pos[:, 0]
#     right_angle_hip = joint_pos[:, 1]
#     left_angle_knee = joint_pos[:, 2]
#     right_angle_knee = joint_pos[:, 3]
    

#     # 定义阶段阈值
#     SWING_THRESHOLD_KNEE = 0.4  # 角度 > 此值 = 摆动（脚在空中）
#     SWING_THRESHOLD_HIP = -0.2  # 角度 < 此值 = 摆动

#     # 判断是否处于摆动阶段（脚在空中）
#     left_swing_knee = (left_angle_knee > SWING_THRESHOLD_KNEE)   # [num_envs], bool
#     right_swing_knee = (right_angle_knee > SWING_THRESHOLD_KNEE)
#     left_swing_hip = (left_angle_hip < SWING_THRESHOLD_HIP)
#     right_swing_hip = (right_angle_hip < SWING_THRESHOLD_HIP)

#     # 计算每只脚的"有效摆动时间"（角度超出阈值的部分）
#     left_air_time = torch.clamp(left_angle_knee, min=0.0)
#     right_air_time = torch.clamp(right_angle_knee, min=0.0)

#     # 只允许单腿摆动 (左摆 且 右不摆)  OR  (右摆 且 左不摆)
#     # 检测单腿摆动状态
#     left_single_swing = left_swing_knee & left_swing_hip & ~right_swing_knee & ~right_swing_hip
#     right_single_swing = ~left_swing_knee & ~left_swing_hip & right_swing_knee & right_swing_hip
#     is_single_swing = left_single_swing | right_single_swing

#     # 获取命令速度并计算条件
#     command_vel = env.command_manager.get_command(command_name)[:, :2]
#     velocity_condition = torch.norm(command_vel, dim=1) > 0.1

#     # 选择摆动脚的 air_time
#     swing_air_time = torch.where(
#         left_single_swing, 
#         left_air_time, 
#         torch.where(right_single_swing, right_air_time, torch.zeros_like(left_air_time))
#     )

#     # 计算基础奖励（强调两腿角度差异）
#     base_reward = swing_air_time * 10 * torch.abs(left_angle_knee - right_angle_knee)

#     # 检查膝关节角度差异是否显著，以避免轻微摆动。
#     has_significant_knee_angle_diff = torch.abs(left_angle_knee - right_angle_knee) > 0.1

#     # 检查走路姿势
#     is_walk_pose = (left_angle_knee>0) & (left_angle_hip<0) & (right_angle_knee>0) & (right_angle_hip<0)

#     # 根据条件应用奖励/惩罚
#     reward = torch.where(
#         is_single_swing & velocity_condition & has_significant_knee_angle_diff & is_walk_pose,
#         base_reward,          # 单腿摆动+有速度命令 → 正奖励
#         -base_reward   # 其他情况 → 惩罚
#     )
#     reward = torch.clamp(reward, max=threshold)
#     return reward


# def feet_flight_penalty(env, sensor_cfg: SceneEntityCfg, force_threshold: float = 5.0):
#     sensor = env.scene.sensors[sensor_cfg.name]
#     data = sensor.data
#     forces = data.net_forces_w   # 如果你用的是这个字段
#     force_mag = torch.linalg.norm(forces, dim=-1)
#     in_contact = force_mag > force_threshold

#     left = in_contact[:, 0]
#     right = in_contact[:, 1]

#     # 只惩罚双脚都不接触（飞起来）
#     flight = torch.logical_and(~left, ~right)
#     return flight.float()

# def feet_flat_penalty(
#     env,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     force_threshold: float = 5.0,
#     local_up_axis: torch.Tensor | None = None,
# ):
#     asset: Articulation = env.scene[asset_cfg.name]
#     sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

#     # contact mask from forces
#     forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]           # [N,2,3]
#     force_mag = torch.linalg.norm(forces_w, dim=-1)                          # [N,2]
#     in_contact = force_mag > force_threshold                                 # [N,2]

#     # foot quats in world
#     foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]           # [N,2,4] (wxyz)

#     # local up axis of foot (default +Z)
#     if local_up_axis is None:
#         local_up_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=foot_quat_w.dtype)
#     local_up_axis = local_up_axis.view(1, 3)                                 # [1,3]

#     # ---- flatten (N,2,4) -> (N*2,4) to satisfy math_utils.quat_apply ----
#     N = foot_quat_w.shape[0]
#     foot_quat_flat = foot_quat_w.reshape(-1, 4)                              # [N*2,4]
#     vec_flat = local_up_axis.repeat(foot_quat_flat.shape[0], 1)              # [N*2,3]

#     foot_up_flat = math_utils.quat_apply(foot_quat_flat, vec_flat)           # [N*2,3]
#     foot_up_w = foot_up_flat.reshape(N, 2, 3)                                # [N,2,3]

#     # alignment with world +Z
#     cos = foot_up_w[..., 2].clamp(-1.0, 1.0)                                 # [N,2]
#     tilt = (1.0 - cos)                                                       # 0 when flat

#     # only penalize in stance
#     penalty = torch.sum(tilt * in_contact.float(), dim=-1)                   # [N]
#     return penalty


# def feet_clearance_reward(
#     env, asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     target_height: float = 0.05,
#     force_threshold: float = 5.0,
# ):
#     robot = env.scene[asset_cfg.name]

#     # ===== DEBUG PRINT (only once) =====
#     if not hasattr(env, "_printed_foot_quat"):
#         foot_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N,2,4]
#         print("DEBUG foot body_ids:", asset_cfg.body_ids)
#         print("DEBUG LEFT foot quat:", foot_quat_w[0, 0])
#         print("DEBUG RIGHT foot quat:", foot_quat_w[0, 1])
#         print("DEBUG ROOT quat:", robot.data.root_quat_w[0])
#         env._printed_foot_quat = True

#     feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
#     foot_z = feet_pos_w[..., 2]  # [num_envs, 2]

#     # infer contact from forces
#     sensor = env.scene.sensors[sensor_cfg.name]
#     forces_w = sensor.data.net_forces_w
#     force_mag = torch.linalg.norm(forces_w[:, sensor_cfg.body_ids, :], dim=-1)  # [num_envs, 2]
#     in_contact = force_mag > force_threshold

#     # Only reward clearance when foot is NOT in contact (swing foot)
#     swing = ~in_contact

#     # hinge reward up to target_height
#     z_clamped = torch.clamp(foot_z, 0.0, target_height) / target_height

#     # mask: stance foot gets 0 reward
#     reward = torch.sum(z_clamped * swing.float(), dim=-1)

#     return reward

# class joint_pos_limits_penalty_ratio(ManagerTermBase):
#     """Penalty for violating joint position limits weighted by the gear ratio."""

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # add default argument
#         asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]

#         # resolve the gear ratio for each joint
#         self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
#         index_list, _, value_list = string_utils.resolve_matching_names_values(
#             cfg.params["gear_ratio"], asset.joint_names
#         )
#         self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
#         self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         threshold: float,
#         gear_ratio: dict[str, float],
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         # compute the penalty over normalized joints
#         joint_pos_scaled = math_utils.scale_transform(
#             asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
#         )
#         # scale the violation amount by the gear ratio
#         violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
#         violation_amount = violation_amount * self.gear_ratio_scaled

#         return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


# class power_consumption(ManagerTermBase):
#     """Penalty for the power consumed by the actions to the environment.

#     This is computed as commanded torque times the joint velocity.
#     """

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # add default argument
#         asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]

#         # resolve the gear ratio for each joint
#         self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
#         index_list, _, value_list = string_utils.resolve_matching_names_values(
#             cfg.params["gear_ratio"], asset.joint_names
#         )
#         self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
#         self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

#     def __call__(
#         self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
#     ) -> torch.Tensor:
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         # return power = torque * velocity (here actions: joint torques)
#         return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)
