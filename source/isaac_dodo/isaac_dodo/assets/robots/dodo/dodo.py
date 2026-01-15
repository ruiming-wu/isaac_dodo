# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Dodo robot."""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

_THIS_DIR = Path(__file__).resolve().parent
USD_PATH = _THIS_DIR / "dodobot_v3.usd"

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
        pos=(0.0, 0.0, 0.60),
        joint_pos={
            ".*_joint_1": 0.0,
            ".*_joint_2": -0.35,
            ".*_joint_3": 0.70,
            ".*_joint_4": -0.35,
        },

        # joint_pos={ ".*_joint_1": 0.0, ".*_joint_2": -0.3, ".*_joint_3": 0.90, ".*_joint_4": -0.65,}
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["left_joint_.*", "right_joint_.*"],
            stiffness=42.0,
            damping=2.5,
            armature = 0.01,
            effort_limit_sim=6.0,
            velocity_limit_sim=5.0,
        ),
    },
)
"""Configuration for the Dodo robot."""