# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""

def joints_out_of_range(
    env: ManagerBasedRLEnv,
    ranges,
    joint_ids=None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when specified joints go outside their bounds.

    Args:
        env: RL env.
        ranges: (low, high) for all joints, or a list of (low, high) per joint (same length as joints selected).
        joint_ids: Optional explicit joint indices to check. If None, uses ``asset_cfg.joint_ids`` (or all joints).
        asset_cfg: Robot config.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve joint indices
    ids = joint_ids if joint_ids is not None else asset_cfg.joint_ids
    if ids is None:
        ids = slice(None)

    joint_pos = asset.data.joint_pos[:, ids]

    # normalize ranges to per-joint tensor
    if isinstance(ranges[0], (list, tuple)):
        ranges_tensor = torch.tensor(ranges, device=env.device, dtype=joint_pos.dtype)
    else:
        ranges_tensor = torch.tensor([ranges] * joint_pos.shape[1], device=env.device, dtype=joint_pos.dtype)

    lower_violation = joint_pos < ranges_tensor[None, :, 0]
    upper_violation = joint_pos > ranges_tensor[None, :, 1]

    return torch.any(lower_violation | upper_violation, dim=1)

