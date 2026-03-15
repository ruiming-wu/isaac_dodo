# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_roll_pitch_yaw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """完整的欧拉角：Roll、Pitch、Yaw。
    
    这是最稳定和完整的 IMU 信息组合，三个轴的旋转角度都包括在内。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    
    # 返回 roll, pitch, yaw (不进行 atan2 归一化)
    return torch.cat((roll.unsqueeze(-1), pitch.unsqueeze(-1), yaw.unsqueeze(-1)), dim=-1)


def feet_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
) -> torch.Tensor:
    """观测双脚接触状态。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return (contact_time > 0.0).float()


def feet_air_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    clip_max: float = 0.8,
) -> torch.Tensor:
    """观测双脚当前离地持续时间。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.clamp(air_time, max=clip_max)


def feet_contact_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_link_4"),
    clip_max: float = 0.8,
) -> torch.Tensor:
    """观测双脚当前着地持续时间。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return torch.clamp(contact_time, max=clip_max)
