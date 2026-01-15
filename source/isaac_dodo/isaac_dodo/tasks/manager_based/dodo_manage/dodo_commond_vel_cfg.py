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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from omni.isaac.orbit_assets.isaaclab import ISAACLAB_ASSETS_DATA_DIR

import isaac_dodo.tasks.manager_based.dodo_manage.mdp as mdp
from isaac_dodo.assets.robots.dodo import DODO_CFG

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
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,  # 0%站立环境，都要运动
        rel_heading_envs=0.0,   # 0%朝向控制，直接使用角速度
        heading_command=False,  # 不使用朝向命令
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(-0.5, 0.5),  
    ))

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        # isaaclab自带
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # 观测机器人基座的线性速度(包含x、y、z三个方向)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 基座的角速度(使用scale进行归一化缩放)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)) # 观测投影后的重力方向信息
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节状态
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # # 自己编写的
        # base_roll_pitch = ObsTerm(func=mdp.base_roll_pitch) # 机器人的翻滚角(roll) 和俯仰角(pitch)
        # base_up_proj = ObsTerm(func=mdp.base_up_proj) # 机器人向上方向与世界坐标系z轴的投影关系，用于判断机器人是否保持直立姿态
        # base_heading_proj = ObsTerm(  # 朝向与命令方向的投影关系
        #     func=mdp.base_heading_proj_to_command,
        #     params={"command_name": "base_velocity"}
        # )
        # base_angle_to_command = ObsTerm(  # 观测机器人与命令方向的角度差
        #     func=mdp.base_angle_to_command,
        #     params={"command_name": "base_velocity"}
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Test config class for critic observation group"""
        # isaaclab自带
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01)) # 观测机器人基座的高度（z坐标）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # 观测机器人基座的线性速度(包含x、y、z三个方向)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 基座的角速度(使用scale进行归一化缩放)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)) # 观测投影后的重力方向信息
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节状态
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # 接触力
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"])},
        )

        actions = ObsTerm(func=mdp.last_action)

        # 自己编写的
        base_roll_pitch_yaw = ObsTerm(func=mdp.base_roll_pitch_yaw) # 机器人的偏航角(yaw) 翻滚角(roll) 和俯仰角(pitch)
        base_up_proj = ObsTerm(func=mdp.base_up_proj) # 机器人向上方向与世界坐标系z轴的投影关系，用于判断机器人是否保持直立姿态
        # base_heading_proj = ObsTerm(  # 朝向与命令方向的投影关系
        #     func=mdp.base_heading_proj_to_command,
        #     params={"command_name": "base_velocity"}
        # )
        # base_angle_to_command = ObsTerm(  # 观测机器人与命令方向的角度差
        #     func=mdp.base_angle_to_command,
        #     params={"command_name": "base_velocity"}
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

    # startup events
    # 随机化机器人基座的质心位置偏移
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset events
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # isaaclab自带
    # alive = RewTerm(func=mdp.is_alive, weight=2.0) # 存活奖励
    termination = RewTerm(func=mdp.is_terminated, weight=-5.0) # 结束惩罚
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01) # 惩罚过大的动作

    # 自己编写的
    # 直立姿态奖励
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.25, params={"threshold": 0.45})

    # 线速度跟踪
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 角速度跟踪
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # 能耗惩罚
    energy = RewTerm(
        func=mdp.power_consumption, weight=-0.005,
        params={"gear_ratio": {".*": 2.5}},
    )
    
    # # 翻滚角(roll)惩罚
    # roll_penalty = RewTerm(
    #     func=mdp.roll_penalty, weight=-0.75,
    #     params={"std": 0.3},
    # )
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # 惩罚基座在x、y方向的角速度
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005) # 惩罚动作变化率
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint_.*"])}) # 惩罚关节力矩
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint_1",".*_joint_2", ".*_joint_3"])} ) # 惩罚关节加速度

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint_1"])},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "threshold": 0.3,
        },
    )
        
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
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
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    roll_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
    )
    pitch_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
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
        self.decimation = 4
        self.episode_length_s = 40.0
    
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
