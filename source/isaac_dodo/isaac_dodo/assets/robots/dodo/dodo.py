# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Dodo Robot (dodo_daimao.usd model)

这个文件定义了Dodo机器人在ISaac Lab中的配置。

## 机器人结构

模型: dodo_daimao.usd
关节总数: 8个（4条腿，每条腿2个自由度）

关节名称模式（正则表达式）:
- hip_left / hip_right           → 髋关节（1个DOF）
- upper_leg_left / upper_leg_right  → 上腿（1个DOF）
- lower_leg_left / lower_leg_right  → 下腿（不可独立控制）
- foot_left / foot_right         → 脚部（被动，接地反馈）

## 初始状态

init_state.joint_pos 中的默认位置（单位: 弧度）:
- hip_.*: 0.10        → 髋关节轻微外展
- upper_leg_.*: -0.16 → 上腿向下弯曲
- lower_leg_.*: 0.24  → 下腿向上折叠（站立姿态）
- foot_.*: 0.00       → 脚部中立

这个初始姿态对应机器人站立并准备向前走的状态。

## 执行器配置

所有关节由ImplicitActuatorCfg驱动：
- 刚度 (stiffness): 32.0
- 阻尼 (damping): 3.0
- PD控制，接收关节位置目标

## 联系传感器

脚部传感器 (contact sensors):
- 法向力 (normal forces)
- 接触点位置
- 空中接触时间

这些数据用于步态检测和平衡控制。
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

_THIS_DIR = Path(__file__).resolve().parent
USD_PATH = _THIS_DIR / "dodo_daimao.usd"

DODO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True, 
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "hip_.*": 0.10,        # hip
            "upper_leg_.*": -0.16, # knee-equivalent
            "lower_leg_.*": 0.24,  # ankle-equivalent
            "foot_.*": 0.00,       # foot
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Use distinct actuator names so both groups are kept.
        "hip_upper": ImplicitActuatorCfg(
            joint_names_expr=["hip_.*", "upper_leg_.*"],
            stiffness=32.0,
            damping=3.0,
            armature=0.01,
            effort_limit_sim=27.0,
            velocity_limit_sim=5.5,
        ),
        "lower_foot": ImplicitActuatorCfg(
            joint_names_expr=["lower_leg_.*", "foot_.*"],
            stiffness=32.0,
            damping=3.0,
            armature=0.01,
            effort_limit_sim=9.0,
            velocity_limit_sim=5.5,
        ),
    },
)
