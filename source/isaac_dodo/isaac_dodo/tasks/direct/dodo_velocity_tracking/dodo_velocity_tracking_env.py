# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
import math

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from isaaclab.assets import Articulation

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from .dodo_velocity_tracking_env_cfg import DodoVelocityTrackingEnvCfg

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class DodoVelocityTrackingEnv(LocomotionEnv):
    cfg: DodoVelocityTrackingEnvCfg

    def __init__(self, cfg: DodoVelocityTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        lin_low, lin_high = self.cfg.cmd_lin_range
        ang_low, ang_high = self.cfg.cmd_ang_range
        self.command_lin = lin_low + (lin_high - lin_low) * torch.rand((self.num_envs, 1), device=self.sim.device)
        self.command_ang = ang_low + (ang_high - ang_low) * torch.rand((self.num_envs, 1), device=self.sim.device)

        self.applied_torques = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.sim.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.sim.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = getattr(self, "actions", torch.zeros_like(actions))
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.applied_torques = forces.clone()
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.motor_pos = self.robot.data.joint_pos          # 8 joint positions
        self.motor_vel = self.robot.data.joint_vel          # 8 joint velocities
        self.motor_tau = self.applied_torques               # 8 joint efforts

        self.base_lin_vel = self.robot.data.root_lin_vel_w  # 3 base linear velocities
        self.base_ang_vel = self.robot.data.root_ang_vel_w  # 3 base angular velocities

        self.base_position = self.robot.data.root_pos_w     # base position
        q = self.robot.data.root_quat_w                     # roll, pitch, yaw
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        self.roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        self.pitch = torch.asin(torch.clamp(2 * (qw * qy - qz * qx), -1 + 1e-6, 1 - 1e-6))
        self.yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.motor_pos,
                self.motor_vel,
                self.motor_tau,
                self.base_lin_vel,
                self.base_ang_vel,
                self.roll.unsqueeze(-1),
                self.pitch.unsqueeze(-1),
                self.yaw.unsqueeze(-1),
                self.command_lin,
                self.command_ang,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # linear and angular velocity tracking
        v_forward = torch.sqrt(self.base_lin_vel[:, 0] ** 2 + self.base_lin_vel[:, 1] ** 2)
        v_command = self.command_lin.squeeze(-1)
        w_yaw = self.base_ang_vel[:, 2]
        w_command = self.command_ang.squeeze(-1)

        lin_vel_err_sq = (v_forward - v_command) ** 2
        r_lin = self.cfg.reward_lin_vel_w * torch.exp(-lin_vel_err_sq / (2 * self.cfg.lin_vel_sigma ** 2))
        ang_vel_err_sq = (w_yaw - w_command) ** 2
        r_ang = self.cfg.reward_ang_vel_w * torch.exp(-ang_vel_err_sq / (2 * self.cfg.ang_vel_sigma ** 2))

        # orientation reward
        orient_err_sq = self.roll ** 2 + self.pitch ** 2
        r_orient = self.cfg.reward_orientation_w * torch.exp(-orient_err_sq / (2 * self.cfg.orientation_sigma ** 2))

        # energy regularization
        torque_mag = torch.sum(torch.abs(self.motor_tau), dim=1)           # L1
        max_possible = torch.sum(self.joint_gears) * self.action_scale
        r_torque = self.cfg.reward_torque_reg_w * (1.0 - torque_mag / (max_possible + 1e-6)).clamp(min=0.0)

        # action rate regularization
        action_diff_sq = torch.mean((self.actions - self.prev_actions) ** 2, dim=1)
        r_action_rate = self.cfg.reward_action_rate_w * torch.exp(-action_diff_sq / (2 * self.cfg.action_rate_sigma ** 2))

        # alive 
        r_alive = torch.ones(self.num_envs, device=self.sim.device) * self.cfg.reward_alive_w

        # failure penalty
        height_failure = self.base_position[:, 2] < self.cfg.termination_height
        roll_failure = torch.abs(self.roll) > self.cfg.termination_roll
        pitch_failure = torch.abs(self.pitch) > self.cfg.termination_pitch
        failure = height_failure | roll_failure | pitch_failure
        p_fail = failure.float() * self.cfg.reward_failure_penalty

        # total reward
        total_reward = r_lin + r_ang + r_orient + r_torque + r_action_rate + r_alive + p_fail

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        height_failure = self.base_position[:, 2] < self.cfg.termination_height
        roll_failure = torch.abs(self.roll) > self.cfg.termination_roll
        pitch_failure = torch.abs(self.pitch) > self.cfg.termination_pitch
        died = height_failure | roll_failure | pitch_failure
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # new command
        lin_low, lin_high = self.cfg.cmd_lin_range
        ang_low, ang_high = self.cfg.cmd_ang_range
        self.command_lin[env_ids] = lin_low + (lin_high - lin_low) * torch.rand((len(env_ids), 1), device=self.sim.device)
        self.command_ang[env_ids] = ang_low + (ang_high - ang_low) * torch.rand((len(env_ids), 1), device=self.sim.device)

        self.applied_torques[env_ids] = 0.0
        self._compute_intermediate_values()
