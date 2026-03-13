# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized constants for Dodo Manage environment configuration."""

SCENE_CONFIG = {
    "num_envs": 4096,
    "env_spacing": 5.0,
    "terrain_friction": 0.8,
    "sim_dt": 1 / 120.0,
    "decimation": 2,
    "episode_length_s": 16.0,
}

CONTROL_CONFIG = {
    "action_scale": 0.5,
    "use_default_offset": True,
}

OBSERVATION_SCALES = {
    "base_ang_vel": 0.25,
    "joint_vel_rel": 0.1,
    "feet_body_forces": 0.01,
}

COMMAND_RANGES = {
    "lin_vel_x": (0.14, 0.40),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (0.0, 0.0),
    "heading": (-3.14159, 3.14159),
    "resampling_time": (3.0, 5.0),
}

RESET_RANGES = {
    "base_pos_x": (-0.5, 0.5),
    "base_pos_y": (-0.5, 0.5),
    "base_pos_z": (0.4, 0.6),
    "base_lin_vel_x": (-0.15, 0.15),
    "base_lin_vel_y": (-0.10, 0.10),
    "base_lin_vel_z": (-0.10, 0.10),
    "joint_pos_offset": (-0.05, 0.05),
    "joint_vel_offset": (-0.05, 0.05),
}

FORCE_THRESHOLDS = {
    "stance": 5.0,
    "swing": 3.0,
    "slide": 8.0,
}

TERMINATION_CONFIG = {
    "min_torso_height": 0.3,
    "max_tilt_angle": 0.8,
}

REWARD_CONFIG = {
    # Reward config is split into two layers:
    # 1) "weights": how much each reward term contributes to the final return
    # 2) the fields below: target/std/threshold parameters used inside reward functions
    "weights": {
        # Core stability / tracking terms.
        "termination": -10.0,
        "action_l2": -0.002,
        "action_rate": -0.006,
        "upright": 0.56,
        "pitch_stability": 0.46,
        "pitch_rate": -0.2,
        "stance_stability": 0.22,
        "torso_height_target": 0.36,
        "track_lin_vel": 1.7,
        "track_ang_vel": 0.0,
        "yaw_rate": -0.2,
        "lin_vel_y": -0.15,
        "lin_vel_z": -0.25,
        "feet_slide": -0.06,
        # Contact-timing and stepping structure.
        "single_support": 0.48,
        "alternate_steps": 0.48,
        "swing_foot_forward": 0.68,
        # Knee and leg-coordination shaping.
        "knee_flex": 0.26,
        "swing_knee": 0.46,
        "knee_symmetry": 0.3,
        "leg_phase": 1.1,
        "phase_reference": 1.15,
        "hip_phase_reference": 0.92,
        "hip_antiphase": 0.36,
        "hip_vel_antiphase": 0.34,
        "symmetry_amp": 0.56,
    },
    # Stability-related reward parameters.
    "upright_threshold": 0.45,
    "pitch_std": 0.30,
    "stance_pitch_std": 0.18,
    "stance_height_std": 0.06,
    "torso_height_target": 0.50,
    "torso_height_std": 0.10,
    # Velocity-tracking parameters.
    "lin_vel_std": 0.20,
    "ang_vel_std": 0.5,
    # Foot placement / swing-foot shaping parameters.
    "swing_forward_target": 0.09,
    "swing_forward_std": 0.06,
    "swing_forward_force_threshold": 6.0,
    "gait_reward_gate_speed": 0.06,
    # Knee-target and left/right coordination parameters.
    "knee_target": -0.42,
    "knee_std": 0.20,
    "swing_knee_target": -0.78,
    "swing_knee_std": 0.22,
    "swing_knee_force_threshold": 6.0,
    "knee_symmetry_std": 0.16,
    "leg_phase_knee_delta": 0.22,
    "leg_phase_std": 0.16,
    # Explicit sinusoidal gait-reference parameters.
    "phase_period": 0.72,
    "phase_hip_amplitude": 0.52,
    "phase_hip_std": 0.20,
    "hip_phase_std": 0.14,
    "phase_knee_stance": -0.28,
    "phase_knee_swing_amp": 0.60,
    "phase_knee_std": 0.18,
    # Hip-specific coordination parameters.
    "hip_antiphase_std": 0.16,
}

JOINT_CONFIG = {
    "hip_joints": ["left_joint_1", "right_joint_1"],
    "knee_joints": ["left_joint_2", "right_joint_2"],
    "feet_bodies": ["left_link_4", "right_link_4"],
    "left_leg_joints": ["left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4"],
    "right_leg_joints": ["right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4"],
}


def get_force_threshold(context: str) -> float:
    return FORCE_THRESHOLDS.get(context, 15.0)


def get_reward_weight(reward_name: str) -> float:
    return REWARD_CONFIG["weights"].get(reward_name, 0.0)
