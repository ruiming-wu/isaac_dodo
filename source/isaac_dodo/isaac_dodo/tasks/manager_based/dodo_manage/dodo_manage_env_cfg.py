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
from isaac_dodo.tasks.manager_based.dodo_manage.dodo_manage_cfg_constants import (
    SCENE_CONFIG, CONTROL_CONFIG, OBSERVATION_SCALES, COMMAND_RANGES, RESET_RANGES,
    FORCE_THRESHOLDS, TERMINATION_CONFIG, REWARD_CONFIG, JOINT_CONFIG,
    get_force_threshold, get_reward_weight
)

##
# Scene definition
##

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
        resampling_time_range=COMMAND_RANGES["resampling_time"],
        rel_standing_envs=0.35, # 先让一部分环境学会稳定站立，再过渡到行走
        rel_heading_envs=0.0, # 100%的环境将使用朝向控制
        heading_command=False, # 朝向命令
        heading_control_stiffness=0.5, # 刚度系数，响应朝向变化的速度
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( # 定义了训练过程中机器人可能接收到的各种命令的取值范围，避免过拟合到特定的运动模式
            lin_vel_x=COMMAND_RANGES["lin_vel_x"],
            lin_vel_y=COMMAND_RANGES["lin_vel_y"],
            ang_vel_z=COMMAND_RANGES["ang_vel_z"],
            heading=COMMAND_RANGES["heading"],
        ),
    )



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     scale=0.4,
    # )
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_joint_.*", "right_joint_.*"],
        scale=CONTROL_CONFIG["action_scale"],
        use_default_offset=CONTROL_CONFIG["use_default_offset"],
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

    # isaaclab自带
        base_height = ObsTerm(func=mdp.base_pos_z) # 观测机器人基座的高度（z坐标）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # 观测机器人基座的线性速度(包含x、y、z三个方向)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25) # 基座的角速度(使用scale进行归一化缩放)        
        
        
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节状态
        # joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized) 
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)

        actions = ObsTerm(func=mdp.last_action)

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

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
        params={"pose_range": {"x": RESET_RANGES["base_pos_x"], "y": RESET_RANGES["base_pos_y"], "z": RESET_RANGES["base_pos_z"]},
            "velocity_range": {"x": RESET_RANGES["base_lin_vel_x"], "y": RESET_RANGES["base_lin_vel_y"], "z": RESET_RANGES["base_lin_vel_z"]},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": RESET_RANGES["joint_pos_offset"],
            "velocity_range": RESET_RANGES["joint_vel_offset"],
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP. All parameters are centralized in dodo_manage_cfg_constants."""

    # Termination penalty
    termination = RewTerm(func=mdp.is_terminated, weight=get_reward_weight("termination"))
    
    # Basic penalties
    action_l2 = RewTerm(func=mdp.action_l2, weight=get_reward_weight("action_l2"))
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=get_reward_weight("action_rate"))
    
    # Stability rewards
    upright = RewTerm(
        func=mdp.upright_posture_bonus,
        weight=get_reward_weight("upright"),
        params={"threshold": REWARD_CONFIG["upright_threshold"]}
    )
    
    # Velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=get_reward_weight("track_lin_vel"),
        params={"command_name": "base_velocity", "std": REWARD_CONFIG["lin_vel_std"]}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=get_reward_weight("track_ang_vel"),
        params={"command_name": "base_velocity", "std": REWARD_CONFIG["ang_vel_std"]}
    )
    
    # Locomotion penalties
    yaw_rate = RewTerm(func=mdp.yaw_rate_l2, weight=get_reward_weight("yaw_rate"))
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2, weight=get_reward_weight("lin_vel_y"))
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=get_reward_weight("lin_vel_z"))
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=get_reward_weight("feet_slide"),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=JOINT_CONFIG["feet_bodies"]),
        },
    )
    
    # Gait rewards
    single_support = RewTerm(
        func=mdp.single_support_reward,
        weight=get_reward_weight("single_support"),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "force_threshold": get_force_threshold("stance"),
        },
    )
    
    alternate_steps = RewTerm(
        func=mdp.alternate_footstep_reward,
        weight=get_reward_weight("alternate_steps"),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "force_threshold": get_force_threshold("stance"),
            "command_name": "base_velocity",
        },
    )
    
    # Foot geometry rewards
    feet_sep = RewTerm(
        func=mdp.feet_lateral_separation_reward,
        weight=get_reward_weight("feet_sep"),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=JOINT_CONFIG["feet_bodies"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "target_sep": REWARD_CONFIG["lateral_sep_target"],
            "std": REWARD_CONFIG["lateral_sep_std"],
            "force_threshold": REWARD_CONFIG["lateral_sep_force_threshold"],
        },
    )
    
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_reward,
        weight=get_reward_weight("feet_clearance"),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=JOINT_CONFIG["feet_bodies"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "target_height": REWARD_CONFIG["clearance_target_height"],
            "force_threshold": REWARD_CONFIG["clearance_force_threshold"],
        },
    )
    
    # Joint flexion rewards
    knee_flex = RewTerm(
        func=mdp.knee_flexion_target_exp,
        weight=get_reward_weight("knee_flex"),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"]),
            "knee_target": REWARD_CONFIG["knee_target"],
            "std": REWARD_CONFIG["knee_std"],
        },
    )
    
    swing_knee = RewTerm(
        func=mdp.swing_knee_flexion_reward,
        weight=get_reward_weight("swing_knee"),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "knee_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"]),
            "knee_target": REWARD_CONFIG["swing_knee_target"],
            "std": REWARD_CONFIG["swing_knee_std"],
            "force_threshold": REWARD_CONFIG["swing_knee_force_threshold"],
        },
    )
    
    hip_swing = RewTerm(
        func=mdp.hip_swing_amplitude_reward,
        weight=get_reward_weight("hip_swing"),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "command_name": "base_velocity",
            "target": REWARD_CONFIG["hip_target_amplitude"],
            "max_amp": REWARD_CONFIG["hip_max_amplitude"],
            "force_threshold": REWARD_CONFIG["hip_force_threshold"],
        },
    )

    hip_antiphase = RewTerm(
        func=mdp.hip_antiphase_reward,
        weight=get_reward_weight("hip_antiphase"),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "command_name": "base_velocity",
            "std": REWARD_CONFIG.get("hip_antiphase_std", 0.25),
            "force_threshold": get_force_threshold("stance"),
        },
    )

    hip_vel_antiphase = RewTerm(
        func=mdp.hip_velocity_antiphase_reward,
        weight=get_reward_weight("hip_vel_antiphase"),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "command_name": "base_velocity",
            "force_threshold": get_force_threshold("stance"),
        },
    )

    symmetry_amp = RewTerm(
        func=mdp.symmetry_amplitude_reward,
        weight=get_reward_weight("symmetry_amp"),
        params={
            "left_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["left_leg_joints"]),
            "right_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["right_leg_joints"]),
            "command_name": "base_velocity",
        },
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": TERMINATION_CONFIG["min_torso_height"]}
    )
    
    # (3) Terminate if the robot tilts too much (roll/pitch)
    roll_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": TERMINATION_CONFIG["max_tilt_angle"]},
    )
    pitch_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": TERMINATION_CONFIG["max_tilt_angle"]},
    )


@configclass
class DodoManageEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Dodo biped walking environment.
    
    All magic numbers, thresholds, and parameters are centralized in dodo_manage_cfg_constants.py
    for easy tuning and reproducibility.
    """

    # Scene settings
    scene: DodoManageSceneCfg = DodoManageSceneCfg(
        num_envs=SCENE_CONFIG["num_envs"],
        env_spacing=SCENE_CONFIG["env_spacing"]
    )
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization. Applies centralized simulation settings."""
        # General settings
        self.decimation = SCENE_CONFIG["decimation"]
        self.episode_length_s = SCENE_CONFIG["episode_length_s"]
        
        # Simulation settings
        self.sim.dt = SCENE_CONFIG["sim_dt"]
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        
        # Default friction material
        self.sim.physics_material.static_friction = SCENE_CONFIG["terrain_friction"]
        self.sim.physics_material.dynamic_friction = SCENE_CONFIG["terrain_friction"]
        self.sim.physics_material.restitution = 0.0