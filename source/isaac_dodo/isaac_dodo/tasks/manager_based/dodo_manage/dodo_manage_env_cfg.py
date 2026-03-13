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

import isaac_dodo.tasks.manager_based.dodo_manage.mdp as mdp
from isaac_dodo.assets.robots.dodo import DODO_CFG
from isaac_dodo.tasks.manager_based.dodo_manage.dodo_manage_cfg_constants import (
    COMMAND_RANGES,
    CONTROL_CONFIG,
    JOINT_CONFIG,
    OBSERVATION_SCALES,
    RESET_RANGES,
    REWARD_CONFIG,
    SCENE_CONFIG,
    TERMINATION_CONFIG,
    get_force_threshold,
    get_reward_weight,
)


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
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
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
        joint_names=["left_joint_.*", "right_joint_.*"],
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
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names=JOINT_CONFIG["feet_bodies"])},
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
    termination = RewTerm(func=mdp.is_terminated, weight=get_reward_weight("termination"))
    action_l2 = RewTerm(func=mdp.action_l2, weight=get_reward_weight("action_l2"))
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=get_reward_weight("action_rate"))
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=get_reward_weight("upright"), params={"threshold": REWARD_CONFIG["upright_threshold"]})
    pitch_stability = RewTerm(func=mdp.pitch_stability_bonus, weight=get_reward_weight("pitch_stability"), params={"std": REWARD_CONFIG["pitch_std"]})
    pitch_rate = RewTerm(func=mdp.pitch_rate_l2, weight=get_reward_weight("pitch_rate"))
    stance_stability = RewTerm(
        func=mdp.stance_stability_reward,
        weight=get_reward_weight("stance_stability"),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "pitch_std": REWARD_CONFIG["stance_pitch_std"],
            "height_std": REWARD_CONFIG["stance_height_std"],
            "force_threshold": get_force_threshold("stance"),
        },
    )
    torso_height_target = RewTerm(
        func=mdp.torso_height_target_reward,
        weight=get_reward_weight("torso_height_target"),
        params={"target_height": REWARD_CONFIG["torso_height_target"], "std": REWARD_CONFIG["torso_height_std"]},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=get_reward_weight("track_lin_vel"),
        params={"command_name": "base_velocity", "std": REWARD_CONFIG["lin_vel_std"]},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=get_reward_weight("track_ang_vel"),
        params={"command_name": "base_velocity", "std": REWARD_CONFIG["ang_vel_std"]},
    )
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
    single_support = RewTerm(
        func=mdp.single_support_reward,
        weight=get_reward_weight("single_support"),
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]), "force_threshold": get_force_threshold("stance")},
    )
    alternate_steps = RewTerm(
        func=mdp.alternate_footstep_reward,
        weight=get_reward_weight("alternate_steps"),
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]), "force_threshold": get_force_threshold("stance"), "command_name": "base_velocity"},
    )
    swing_foot_forward = RewTerm(
        func=mdp.swing_foot_forward_reward,
        weight=get_reward_weight("swing_foot_forward"),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=JOINT_CONFIG["feet_bodies"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "command_name": "base_velocity",
            "target": REWARD_CONFIG["swing_forward_target"],
            "std": REWARD_CONFIG["swing_forward_std"],
            "force_threshold": REWARD_CONFIG["swing_forward_force_threshold"],
        },
    )
    knee_flex = RewTerm(
        func=mdp.knee_flexion_target_exp,
        weight=get_reward_weight("knee_flex"),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"]), "knee_target": REWARD_CONFIG["knee_target"], "std": REWARD_CONFIG["knee_std"]},
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
    knee_symmetry = RewTerm(
        func=mdp.knee_symmetry_reward,
        weight=get_reward_weight("knee_symmetry"),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"]), "command_name": "base_velocity", "std": REWARD_CONFIG["knee_symmetry_std"]},
    )
    leg_phase = RewTerm(
        func=mdp.leg_phase_reward,
        weight=get_reward_weight("leg_phase"),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "knee_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"]),
            "hip_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "command_name": "base_velocity",
            "knee_delta_target": REWARD_CONFIG["leg_phase_knee_delta"],
            "std": REWARD_CONFIG["leg_phase_std"],
            "force_threshold": REWARD_CONFIG["swing_knee_force_threshold"],
        },
    )
    phase_reference = RewTerm(
        func=mdp.phase_reference_reward,
        weight=get_reward_weight("phase_reference"),
        params={
            "hip_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "knee_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["knee_joints"]),
            "command_name": "base_velocity",
            "phase_period": REWARD_CONFIG["phase_period"],
            "hip_amplitude": REWARD_CONFIG["phase_hip_amplitude"],
            "hip_std": REWARD_CONFIG["phase_hip_std"],
            "knee_stance": REWARD_CONFIG["phase_knee_stance"],
            "knee_swing_amp": REWARD_CONFIG["phase_knee_swing_amp"],
            "knee_std": REWARD_CONFIG["phase_knee_std"],
        },
    )
    hip_phase_reference = RewTerm(
        func=mdp.hip_phase_reference_reward,
        weight=get_reward_weight("hip_phase_reference"),
        params={
            "hip_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "command_name": "base_velocity",
            "phase_period": REWARD_CONFIG["phase_period"],
            "hip_amplitude": REWARD_CONFIG["phase_hip_amplitude"],
            "std": REWARD_CONFIG["hip_phase_std"],
        },
    )
    hip_antiphase = RewTerm(
        func=mdp.hip_antiphase_reward,
        weight=get_reward_weight("hip_antiphase"),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_CONFIG["hip_joints"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=JOINT_CONFIG["feet_bodies"]),
            "command_name": "base_velocity",
            "std": REWARD_CONFIG["hip_antiphase_std"],
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
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": TERMINATION_CONFIG["min_torso_height"]})
    roll_threshold = DoneTerm(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": TERMINATION_CONFIG["max_tilt_angle"]})
    pitch_threshold = DoneTerm(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": TERMINATION_CONFIG["max_tilt_angle"]})


@configclass
class DodoManageEnvCfg(ManagerBasedRLEnvCfg):
    scene: DodoManageSceneCfg = DodoManageSceneCfg(num_envs=SCENE_CONFIG["num_envs"], env_spacing=SCENE_CONFIG["env_spacing"])
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = SCENE_CONFIG["decimation"]
        self.episode_length_s = SCENE_CONFIG["episode_length_s"]
        self.sim.dt = SCENE_CONFIG["sim_dt"]
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = SCENE_CONFIG["terrain_friction"]
        self.sim.physics_material.dynamic_friction = SCENE_CONFIG["terrain_friction"]
        self.sim.physics_material.restitution = 0.0
