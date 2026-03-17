# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
============================================================
DODO Manage 环境配置模块
============================================================

本模块定义Dodo走路训练环境的完整配置：

## 配置类层次结构

DodoManageEnvCfg (主配置)
  ├── DodoManageSceneCfg      → 场景（地形、机器人、传感器）
  ├── CommandsCfg              → 速度命令生成器
  ├── ObservationsCfg          → 机器人观测（状态、传感器读数）
  ├── ActionsCfg               → 动作（关节目标位置）
  ├── RewardsCfg               → 奖励函数列表
  ├── TerminationsCfg          → 终止条件（摔倒、超时等）
  ├── EventCfg                 → 重置事件
  └── CurriculumCfg            → 课程学习阶段 ← 核心！

## CurriculumCfg 介绍

这个配置类实现了5阶段的课程学习：

阶段        迭代数    改变内容
────────────────────────────────────────
Stage0      0-120    基础学习（低速0.10-0.22）
Stage1      120-180  速度范围确定（0.10-0.22）✓
Stage2      180-260  启用single_support奖励 2.20
Stage3      260-360  启用alternate_steps奖励 2.80
Stage4      360+     启用swing_knee_contrast 1.40
Stage5      420+     扩大速度到评测范围(0.12-0.28)

作用机制:
- 通过_set_value_after_steps()函数根据迭代计数生效
- modify_term_cfg修改命令范围
- modify_reward_weight修改奖励权重

## 使用说明

1. 调试时: 可临时禁用某些curriculum项
   - 注释对应的CurrTerm即可
   
2. 更改curriculum时间表:
   - 修改"num_steps"的值（单位：迭代数）
   - 新权重在修改后不自动复原

3. 查看curriculum是否生效:
   - 查看训练日志中的"Curriculum/"开头的指标
   - 验证权重值是否在指定时刻改变

"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaac_dodo.tasks.manager_based.dodo_manage.mdp as mdp
from isaac_dodo.assets.robots.dodo import DODO_CFG
from isaac_dodo.tasks.manager_based.dodo_manage.dodo_manage_cfg_constants import (
    COMMAND_RANGES,
    CONTROL_CONFIG,
    JOINT_CONFIG,
    OBSERVATION_SCALES,
    RESET_RANGES,
    REWARD_PARAMS,
    SCENE_CONFIG,
    TERMINATION_CONFIG,
    get_force_threshold,
    get_reward_weight,
)


BASE_VELOCITY_CMD = "base_velocity"
FEET_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"])
ROBOT_FEET_CFG = SceneEntityCfg("robot", body_names=JOINT_CONFIG["feet_bodies"])
ROBOT_HIP_CFG = SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"])
ROBOT_KNEE_CFG = SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"])
ROBOT_HIP_KNEE_CFG = SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_knee_joints"])
ROBOT_ANKLE_CFG = SceneEntityCfg("robot", joint_names=JOINT_CONFIG["ankle_joints"])
ROBOT_LEFT_LEG_CFG = SceneEntityCfg("robot", joint_names=JOINT_CONFIG["left_leg_joints"])
ROBOT_RIGHT_LEG_CFG = SceneEntityCfg("robot", joint_names=JOINT_CONFIG["right_leg_joints"])

BASE_CMD_PARAM = {"command_name": BASE_VELOCITY_CMD}
STANCE_FORCE_THRESHOLD = get_force_threshold("stance")
STANCE_CONTACT_PARAM = {"sensor_cfg": FEET_SENSOR_CFG, "force_threshold": STANCE_FORCE_THRESHOLD}

FEET_SENSOR_PARAM = {"sensor_cfg": FEET_SENSOR_CFG}
FEET_ASSET_SENSOR_PARAM = {"asset_cfg": ROBOT_FEET_CFG, **FEET_SENSOR_PARAM}
KNEE_ASSET_PARAM = {"asset_cfg": ROBOT_KNEE_CFG}
HIP_ASSET_PARAM = {"asset_cfg": ROBOT_HIP_CFG}
LEG_PAIR_PARAM = {"left_cfg": ROBOT_LEFT_LEG_CFG, "right_cfg": ROBOT_RIGHT_LEG_CFG}


def _set_value_after_steps(env, env_ids, data, value, num_steps):
    """
    Curriculum辅助函数: 在指定的学习步数之后，修改环境参数。
    
    使用场景: 
    - 在训练早期保持参数低值（如简单的速度命令）
    - 在训练稳定后逐步提高难度
    
    参数:
        env: 机器学习环境
        env_ids: 环境ID列表
        data: 当前数据状态
        value: 要设置的新参数值（元组或标量）
        num_steps: 触发阈值（以学习迭代数计）
    
    返回:
        value - 如果已经超过num_steps迭代
        NO_CHANGE - 如果还未达到num_steps，保持原值
    """
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE


@configclass
class DodoManageSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    robot: ArticulationCfg = DODO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=COMMAND_RANGES["resampling_time"],
        # Keep a small fraction of standing commands so the policy does not forget balance,
        # but most environments still practice forward walking.
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=COMMAND_RANGES["lin_vel_x"],
            lin_vel_y=COMMAND_RANGES["lin_vel_y"],
            ang_vel_z=COMMAND_RANGES["ang_vel_z"],
            heading=COMMAND_RANGES["heading"],
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["hip_.*", "upper_leg_.*", "lower_leg_.*", "foot_.*"],
        scale=CONTROL_CONFIG["action_scale"],
        use_default_offset=CONTROL_CONFIG["use_default_offset"],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=OBSERVATION_SCALES["base_ang_vel"])
        commands = ObsTerm(func=mdp.generated_commands, params={**BASE_CMD_PARAM})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=OBSERVATION_SCALES["joint_vel_rel"])
        actions = ObsTerm(func=mdp.last_action)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=OBSERVATION_SCALES["base_ang_vel"])
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=OBSERVATION_SCALES["joint_vel_rel"])
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=OBSERVATION_SCALES["feet_body_forces"],
            params={"asset_cfg": ROBOT_FEET_CFG},
        )
        actions = ObsTerm(func=mdp.last_action)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": RESET_RANGES["base_pos_x"], "y": RESET_RANGES["base_pos_y"], "z": RESET_RANGES["base_pos_z"]},
            "velocity_range": {"x": RESET_RANGES["base_lin_vel_x"], "y": RESET_RANGES["base_lin_vel_y"], "z": RESET_RANGES["base_lin_vel_z"]},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": RESET_RANGES["joint_pos_offset"], "velocity_range": RESET_RANGES["joint_vel_offset"]},
    )


@configclass
class RewardsCfg:
    """Minimal reward set: survival + gait structure + speed tracking"""
    
    # Survival: Avoid falling
    termination = RewTerm(func=mdp.is_terminated, weight=get_reward_weight("termination"))
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=get_reward_weight("upright"), params={"threshold": REWARD_PARAMS["upright_threshold"]})
    roll_stability = RewTerm(func=mdp.roll_stability_bonus, weight=get_reward_weight("roll_stability"), params={"std": REWARD_PARAMS["roll_std"]})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=get_reward_weight("feet_slide"),
        params={**FEET_ASSET_SENSOR_PARAM},
    )
    
    # Smoothness: Smooth actions
    action_l2 = RewTerm(func=mdp.action_l2, weight=get_reward_weight("action_l2"))
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=get_reward_weight("action_rate"))
    
    # Gait structure: Core stepping pattern
    single_support = RewTerm(
        func=mdp.single_support_reward,
        weight=get_reward_weight("single_support"),
        params={**STANCE_CONTACT_PARAM},
    )
    alternate_steps = RewTerm(
        func=mdp.alternate_footstep_reward,
        weight=get_reward_weight("alternate_steps"),
        params={**STANCE_CONTACT_PARAM, **BASE_CMD_PARAM, "min_air_time": REWARD_PARAMS["alternate_min_air_time"]},
    )
    swing_foot_forward = RewTerm(
        func=mdp.swing_foot_forward_reward,
        weight=get_reward_weight("swing_foot_forward"),
        params={
            **FEET_ASSET_SENSOR_PARAM,
            **BASE_CMD_PARAM,
            "target": REWARD_PARAMS["swing_forward_target"],
            "std": REWARD_PARAMS["swing_forward_std"],
            "force_threshold": REWARD_PARAMS["swing_forward_force_threshold"],
        },
    )
    knee_flex = RewTerm(
        func=mdp.knee_flexion_target_exp,
        weight=get_reward_weight("knee_flex"),
        params={
            **KNEE_ASSET_PARAM,
            "knee_target": REWARD_PARAMS["knee_target"],
            "std": REWARD_PARAMS["knee_std"],
        },
    )
    
    # Locomotion goal: Follow speed command
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp,
        weight=get_reward_weight("track_lin_vel"),
        params={**BASE_CMD_PARAM, "std": REWARD_PARAMS["lin_vel_std"]},
    )
    no_progress = RewTerm(
        func=mdp.no_progress_penalty,
        weight=get_reward_weight("no_progress"),
        params={
            **BASE_CMD_PARAM,
            "speed_ratio": REWARD_PARAMS["no_progress_speed_ratio"],
            "min_speed": REWARD_PARAMS["no_progress_min_speed"],
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": TERMINATION_CONFIG["min_torso_height"]})
    roll_threshold = DoneTerm(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": TERMINATION_CONFIG["max_tilt_angle"]})
    pitch_threshold = DoneTerm(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": TERMINATION_CONFIG["max_tilt_angle"]})


@configclass
class CurriculumCfg:
    """No curriculum: Train with fixed parameters from iteration 0"""
    pass


@configclass
class DodoManageEnvCfg(ManagerBasedRLEnvCfg):
    scene: DodoManageSceneCfg = DodoManageSceneCfg(num_envs=SCENE_CONFIG["num_envs"], env_spacing=SCENE_CONFIG["env_spacing"])
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = SCENE_CONFIG["decimation"]
        self.episode_length_s = SCENE_CONFIG["episode_length_s"]
        self.sim.dt = SCENE_CONFIG["sim_dt"]
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = SCENE_CONFIG["terrain_friction"]
        self.sim.physics_material.dynamic_friction = SCENE_CONFIG["terrain_friction"]
        self.sim.physics_material.restitution = 0.0
