# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

##
# Scene definition
##

TARGET_POS = (1000.0, 0.0, 0.0)

TARGET_LIN_VEL = [-0.5, 0.0, 1.0]
TARGET_ANG_VEL = [-0.5, 0.0, 0.5]

@configclass
class DodoManageSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot : ArticulationCfg = DODO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02, # 2%的环境将获得零速度命令（站立不动）
        rel_heading_envs=1.0, # 100%的环境将使用朝向控制
        heading_command=True, # 朝向命令
        heading_control_stiffness=0.5, # 刚度系数，响应朝向变化的速度
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( # 定义了训练过程中机器人可能接收到的各种命令的取值范围，避免过拟合到特定的运动模式
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=2.5
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

    # isaaclab自带
        # base_height = ObsTerm(func=mdp.base_pos_z) # 观测机器人基座的高度（z坐标）
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # 观测机器人基座的线性速度(包含x、y、z三个方向)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25) # 基座的角速度(使用scale进行归一化缩放)        
        # 关节状态
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized) 
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)

        actions = ObsTerm(func=mdp.last_action)

    # 自己编写的
        # base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll) # 机器人的偏航角(yaw)和翻滚角(roll)
        # base_up_proj = ObsTerm(func=mdp.base_up_proj) # 机器人向上方向与世界坐标系z轴的投影关系，用于判断机器人是否保持直立姿态
        # base_heading_proj = ObsTerm( # 观测机器人朝向与目标方向的投影关系
        #     func=mdp.base_heading_proj, 
        #     params={"target_pos": (1000.0, 0.0,  0.0)}
        # )
        # base_angle_to_target = ObsTerm( # 观测机器人面向目标的角度差
        #     func=mdp.base_angle_to_target, 
        #     params={"target_pos": TARGET_POS} # 目标坐标位置
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Test config class for critic observation group"""
    # isaaclab自带
        base_height = ObsTerm(func=mdp.base_pos_z) # 观测机器人基座的高度（z坐标）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # 观测机器人基座的线性速度(包含x、y、z三个方向)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25) # 基座的角速度(使用scale进行归一化缩放)        
        # 关节状态
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized) 
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)

        # 接触力
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"])},
        )

        actions = ObsTerm(func=mdp.last_action)

    # 自己编写的
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll) # 机器人的偏航角(yaw)和翻滚角(roll)
        base_up_proj = ObsTerm(func=mdp.base_up_proj) # 机器人向上方向与世界坐标系z轴的投影关系，用于判断机器人是否保持直立姿态
        # base_heading_proj = ObsTerm( # 观测机器人朝向与目标方向的投影关系
        #     func=mdp.base_heading_proj, 
        #     params={"target_pos": TARGET_POS}
        # )
        # base_angle_to_target = ObsTerm( # 观测机器人面向目标的角度差
        #     func=mdp.base_angle_to_target, 
        #     params={"target_pos": TARGET_POS} # 目标坐标位置
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.4, 0.6)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

# isaaclab自带
    alive = RewTerm(func=mdp.is_alive, weight=2.0) # 存活奖励
    termination = RewTerm(func=mdp.is_terminated, weight=-10.0) # 结束惩罚
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01) # 惩罚过大的动作

    # 髋关节移位置惩罚
    hip_joint_move = RewTerm(func=mdp.hip_pos_manual_limit,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_joint_1", "right_joint_1"]),
            "softlimit": (0, 0),
        },
    )

# 自己编写的
    # # 前进进度奖励
    # progress = RewTerm(func=mdp.progress_reward, weight=5.0, params={"target_pos": TARGET_POS})
    # 直立姿态奖励 
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.5, params={"threshold": 0.45})
    # # 朝向目标奖励
    # move_to_target = RewTerm(func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": TARGET_POS})
    # 线速度跟踪
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    # 角速度跟踪
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.3}
    )
    # 能耗惩罚
    energy = RewTerm(
        func=mdp.power_consumption, weight=-0.005,
        params={"gear_ratio": {".*": 2.5}},
    )
    # 关节极限惩罚
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.25,
        params={"threshold": 0.98, "gear_ratio": {".*": 2.5}},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped_snesor,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "threshold": 2.5,
        },
    )
        
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_link_4"),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.3})
    roll_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
    )
    pitch_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
    )

    hip_threshold = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_joint_1", "right_joint_1"]), "bounds": [-0.15, 0.15]},
    )


@configclass
class DodoManageEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: DodoManageSceneCfg = DodoManageSceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0