# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
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

import isaac_dodo.tasks.manager_based.dodo_manage_joystick_tracking.mdp as mdp
from isaac_dodo.assets.robots.dodo import DODO_CFG

##
# Scene definition
##

COMMAND_LIN_VEL_X_RANGE = (-0.1, 0.2)
COMMAND_ANG_VEL_Z_RANGE = (-0.5, 0.5)

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
    base_velocity = mdp.CurriculumVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        initial_forward_range=(0.0, 0.05),
        final_forward_range=(0.0, COMMAND_LIN_VEL_X_RANGE[1]),
        initial_backward_range=(0.0, 0.0),
        final_backward_range=(COMMAND_LIN_VEL_X_RANGE[0], 0.0),
        initial_ang_vel_range=(-0.12, 0.12),
        final_ang_vel_range=COMMAND_ANG_VEL_Z_RANGE,
        initial_standing_ratio=0.25,
        final_standing_ratio=0.05,
        num_lin_bins=9,
        num_ang_bins=11,
        adaptive_sampling_start_progress=0.25,
        adaptive_sampling_temperature=2.5,
        min_difficulty_floor=0.08,
        env_success_ema_alpha=0.05,
        bin_mastery_ema_alpha=0.15,
        success_lin_vel_tolerance=0.08,
        success_ang_vel_tolerance=0.12,
        curriculum_start_step=2_000,
        backward_start_step=7_000,
        curriculum_end_step=20_000,
        debug_vis=False,
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
        scale=10.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)

        # 关节状态
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_tau = ObsTerm(func=mdp.joint_effort)

        # 命令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # IMU信息（完整的欧拉角）
        base_roll_pitch_yaw = ObsTerm(func=mdp.base_roll_pitch_yaw)  # Roll, Pitch, Yaw

        feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
        )
        feet_air_time = ObsTerm(
            func=mdp.feet_air_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
        )
        feet_contact_time = ObsTerm(
            func=mdp.feet_contact_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
        )
    
        # 历史
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Test config class for critic observation group"""
        # 基座状态信息
        base_height = ObsTerm(func=mdp.base_pos_z)  # 观测机器人基座的高度（z坐标）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 观测机器人基座的线性速度(包含x、y、z三个方向)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)  # 基座的角速度(使用scale进行归一化缩放)
        
        # 关节状态
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_tau = ObsTerm(func=mdp.joint_effort)

        # 接触力
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"])},
        )
        feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
        )
        feet_air_time = ObsTerm(
            func=mdp.feet_air_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
        )
        feet_contact_time = ObsTerm(
            func=mdp.feet_contact_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
        )

        # IMU信息
        base_roll_pitch_yaw = ObsTerm(func=mdp.base_roll_pitch_yaw)  # Roll, Pitch, Yaw 
        
        # 任务信息
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 目标速度命令
        
        # 历史
        actions = ObsTerm(func=mdp.last_action)

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
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (based on humanoid locomotion paper)."""

    # ========== Body Control ==========
    upright = RewTerm(
        func=mdp.upright_reward,
        weight=3.0,
        params={"std": 0.1},
    )

    base_height = RewTerm(
        func=mdp.height_reward,
        weight=3,
        params={"target_height": 0.40, "std": 0.2},
    )

    # ========== Locomotion & Gait Shaping ==========
    lin_vel_tracking = RewTerm(
        func=mdp.linear_velocity_tracking_reward,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.22,
            "final_std": 0.05,
            "bonus_scale": 0.6,
            "final_bonus_scale": 1.8,
            "curriculum_start_step": 2_000,
            "curriculum_end_step": 30_000,
        },
    )

    ang_vel_tracking = RewTerm(
        func=mdp.angular_velocity_tracking_reward,
        weight=3.5,
        params={
            "command_name": "base_velocity",
            "std": 0.28,
            "final_std": 0.08,
            "bonus_scale": 0.5,
            "final_bonus_scale": 1.6,
            "curriculum_start_step": 2_000,
            "curriculum_end_step": 30_000,
        },
    )

    stance_low_vel = RewTerm(
        func=mdp.stance_high_velocity_penalty,
        weight=-3.0,
        params={"std": 0.1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_link_4")},
    )

    swing_low_force = RewTerm(
        func=mdp.swing_high_force_penalty,
        weight=-2.0,
        params={"std": 35.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4")},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_reward,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "command_name": "base_velocity",
            "std": 0.20,
            "final_std": 0.07,
            "bonus_scale": 0.4,
            "final_bonus_scale": 1.8,
            "curriculum_start_step": 4_000,
            "curriculum_end_step": 35_000,
        },
    )

    gait_single_stance = RewTerm(
        func=mdp.gait_single_stance_reward,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "command_name": "base_velocity",
            "bonus_scale": 0.3,
            "final_bonus_scale": 1.5,
            "curriculum_start_step": 4_000,
            "curriculum_end_step": 35_000,
        },
    )

    gait_phase_symmetry = RewTerm(
        func=mdp.gait_phase_symmetry_reward,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "command_name": "base_velocity",
            "std": 0.26,
            "final_std": 0.10,
            "bonus_scale": 0.2,
            "final_bonus_scale": 1.5,
            "curriculum_start_step": 6_000,
            "curriculum_end_step": 40_000,
        },
    )

    gait_step_period = RewTerm(
        func=mdp.gait_step_period_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "command_name": "base_velocity",
            "target_step_period": 0.5,
            "std": 0.24,
            "final_std": 0.08,
            "bonus_scale": 0.2,
            "final_bonus_scale": 1.8,
            "curriculum_start_step": 8_000,
            "curriculum_end_step": 45_000,
        },
    )

    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height_reward,
        weight=5.0,
        params={
            "target_height": 0.14,
            "std": 0.16,
            "final_std": 0.05,
            "bonus_scale": 0.25,
            "final_bonus_scale": 1.8,
            "curriculum_start_step": 4_000,
            "curriculum_end_step": 35_000,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_link_4"),
        },
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance_reward,
        weight=4.0,
        params={
            "min_height": 0.09,
            "std": 0.025,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_link_4"),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide_penalty,
        weight=-6.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link_4"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_link_4")},
    )

    knee_in_range = RewTerm(
        func=mdp.joint_in_range_reward,
        weight=1.5,
        params={"ranges": (0.65, 0.75), "joint_ids": [2, 7], "asset_cfg": SceneEntityCfg("robot")},
    )

    hip_in_range = RewTerm(
        func=mdp.joint_in_range_reward,
        weight=1.5,
        params={"ranges": (-0.2, -0.15), "joint_ids": [0, 5], "asset_cfg": SceneEntityCfg("robot")},
    )

    # ========== Safety, Smoothness & Regularization ==========
    alive = RewTerm(func=mdp.is_alive, weight=1.5)
    termination = RewTerm(func=mdp.is_terminated, weight=-20.0)

    linear_vel_z = RewTerm(
        func=mdp.linear_vel_z_penalty,
        weight=-0.1,
    )

    angular_vel_xy = RewTerm(
        func=mdp.angular_vel_xy_penalty,
        weight=-0.2,
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_penalty,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_penalty,
        weight=-0.015,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_tau = RewTerm(
        func=mdp.joint_tau_penalty,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    energy = RewTerm(
        func=mdp.energy_penalty,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Episode length
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Robot fell (height below minimum)
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.25}
    )
    
    # (3) Orientation out of bounds
    roll_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.0},
    )
    
    pitch_threshold = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.0},
    )

    # # (4) Joint position out of bounds for hip
    # hip_limits = DoneTerm(
    #     func=mdp.joints_out_of_range,
    #     params={
    #         "ranges": (-0.3, 0.3),  # 扩大范围允许更多运动
    #         "joint_ids": [0, 5],  # left_joint_1, right_joint_1
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # # (5) Joint position out of bounds for knee
    # knee_limits = DoneTerm(
    #     func=mdp.joints_out_of_range,
    #     params={
    #         "ranges": (0.2, 1.2),
    #         "joint_ids": [2, 7],  # left_joint_3, right_joint_3
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )


@configclass
class DodoManageJoystickTrackingEnvCfg(ManagerBasedRLEnvCfg):
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
        self.decimation = 2  # 控制频率 120/2 = 60Hz
        self.episode_length_s = 20.0
    
        # simulation settings
        self.sim.dt = 1 / 120.0  # 8.33ms仿真步长
        self.sim.render_interval = self.decimation
    
        # 物理求解器优化 - 对双足机器人重要
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.solver_type = 1  # TGS求解器，更稳定
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.max_position_iteration_count = 255
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 255
    
        # 接触设置 - 改善足部接触
        self.sim.physx.default_buffer_size_multiplier = 5.0
    
        # 默认摩擦材料 - 确保足部抓地力
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0  # 无弹性，避免弹跳
        self.sim.physics_material.friction_combine_mode = "multiply"  # 摩擦力组合方式
        self.sim.physics_material.restitution_combine_mode = "multiply"
