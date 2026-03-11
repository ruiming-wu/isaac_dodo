# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized constants for Dodo Manage environment configuration.

This module consolidates all magic numbers, thresholds, and scales used throughout
the training pipeline. Modify values here to tune the training behavior globally.
"""

# ============================================================================
# SCENE & SIMULATION SETTINGS
# ============================================================================
SCENE_CONFIG = {
    "num_envs": 4096,
    "env_spacing": 5.0,
    "terrain_friction": 0.8,
    "sim_dt": 1 / 120.0,
    "decimation": 2,
    "episode_length_s": 16.0,
}

# ============================================================================
# ROBOT KINEMATICS & CONTROL
# ============================================================================
CONTROL_CONFIG = {
    "action_scale": 0.8,  # Stability-first: reduce aggressive joint excursions
    "use_default_offset": True,
}

# ============================================================================
# OBSERVATION SCALES & NORMALIZATION
# ============================================================================
OBSERVATION_SCALES = {
    "base_ang_vel": 0.25,
    "joint_vel_rel": 0.1,
    "feet_body_forces": 0.01,
}

# ============================================================================
# COMMAND RANGES
# ============================================================================
COMMAND_RANGES = {
    "lin_vel_x": (0.0, 0.30),   # stage-1 curriculum: stand or very slow walk first
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (0.0, 0.0),
    "heading": (-3.14159, 3.14159),
    "resampling_time": (3.0, 5.0),
}

# ============================================================================
# ENVIRONMENT INITIALIZATION
# ============================================================================
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

# ============================================================================
# CONTACT/FORCE THRESHOLDS
# ============================================================================
FORCE_THRESHOLDS = {
    "stance": 5.0,       # more sensitive contact detection for alternating-gait rewards
    "swing": 3.0,        # swing contact detection
    "slide": 8.0,        # More sensitive sliding detection
}

# ============================================================================
# TERMINATION CONDITIONS
# ============================================================================
TERMINATION_CONFIG = {
    "min_torso_height": 0.3,
    "max_tilt_angle": 0.8,  # rad (for roll/pitch)
}

# ============================================================================
# REWARD FUNCTION PARAMETERS
# ============================================================================
REWARD_CONFIG = {
    # Reward weights
    "weights": {
        "termination": -10.0,
        "action_l2": -0.002,
        "upright": 0.8,
        "track_lin_vel": 0.7,
        "track_ang_vel": 0.0,
        "yaw_rate": -0.2,
        "lin_vel_y": -0.15,
        "lin_vel_z": -0.25,
        "action_rate": -0.006,
        "feet_slide": -0.06,
        "single_support": 0.40,
        "alternate_steps": 0.50,
        "feet_sep": 0.0,
        "feet_clearance": 0.40,
        "knee_flex": 0.0,
        "swing_knee": 0.55,
        "hip_swing": 0.5,
        "symmetry_amp": 0.1,
        "hip_antiphase": 0.0,
        "hip_vel_antiphase": 0.00,
    },
    
    # Upright posture reward
    "upright_threshold": 0.45,
    
    # Velocity tracking reward
    "lin_vel_std": 0.4,
    "ang_vel_std": 0.5,
    "velocity_gate_threshold": 0.1,  # Min command speed to activate rewards
    
    # Feet spacing reward
    "lateral_sep_target": 0.11,
    "lateral_sep_std": 0.06,
    "lateral_sep_force_threshold": 10.0,
    
    # Feet clearance reward
    "clearance_target_height": 0.13,
    "clearance_force_threshold": 15.0,
    
    # Knee flexion reward
    "knee_target": 0.5,
    "knee_std": 0.35,
    "swing_knee_target": 1.1,
    "swing_knee_std": 0.30,
    "swing_knee_force_threshold": 10.0,
    
    # Hip swing reward
    "hip_target_amplitude": 0.75,  # 增大目标摆幅
    "hip_max_amplitude": 1.2,      # 增大最大允许摆幅
    "hip_force_threshold": 10.0,
    "hip_antiphase_std": 0.20,
}

# ============================================================================
# JOINT CONFIGURATION
# ============================================================================
JOINT_CONFIG = {
    "hip_joints": ["left_joint_1", "right_joint_1"],
    "knee_joints": ["left_joint_3", "right_joint_3"],
    "feet_bodies": ["left_link_4", "right_link_4"],
    "left_leg_joints": ["left_joint_1", "left_joint_3"],
    "right_leg_joints": ["right_joint_1", "right_joint_3"],
}

# ============================================================================
# HELPER FUNCTION FOR ACCESSING CONFIG
# ============================================================================
def get_force_threshold(context: str) -> float:
    """Get force threshold for a given context.
    
    Args:
        context: One of 'stance', 'swing', 'slide'
        
    Returns:
        Force threshold in Newtons
    """
    return FORCE_THRESHOLDS.get(context, 15.0)


def get_reward_weight(reward_name: str) -> float:
    """Get reward weight by name.
    
    Args:
        reward_name: Name of the reward term
        
    Returns:
        Weight coefficient
    """
    return REWARD_CONFIG["weights"].get(reward_name, 0.0)
