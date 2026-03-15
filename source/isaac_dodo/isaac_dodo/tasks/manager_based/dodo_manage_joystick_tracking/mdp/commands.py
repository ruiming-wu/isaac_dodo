"""自定义命令生成器，支持自动 curriculum 的双足速度命令采样。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTermCfg
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CurriculumVelocityCommand(CommandTerm):
    """按训练步数自动扩展命令范围的速度命令。"""

    cfg: "CurriculumVelocityCommandCfg"

    def __init__(self, cfg: "CurriculumVelocityCommandCfg", env: ManagerBasedRLEnv):
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
        self._command_bins = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.long)
        self._tracking_success_ema = torch.zeros(env.num_envs, device=env.device)
        self._bin_mastery = torch.zeros(cfg.num_lin_bins, cfg.num_ang_bins, device=env.device)
        self.robot = env.scene[cfg.asset_name]
        super().__init__(cfg, env)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _curriculum_progress(self) -> float:
        step_count = float(getattr(self._env, "common_step_counter", 0))
        if step_count <= self.cfg.curriculum_start_step:
            return 0.0
        if step_count >= self.cfg.curriculum_end_step:
            return 1.0
        total = self.cfg.curriculum_end_step - self.cfg.curriculum_start_step
        return (step_count - self.cfg.curriculum_start_step) / max(total, 1)

    def _backward_progress(self) -> float:
        step_count = float(getattr(self._env, "common_step_counter", 0))
        if step_count <= self.cfg.backward_start_step:
            return 0.0
        if step_count >= self.cfg.curriculum_end_step:
            return 1.0
        total = self.cfg.curriculum_end_step - self.cfg.backward_start_step
        return (step_count - self.cfg.backward_start_step) / max(total, 1)

    @staticmethod
    def _lerp_tuple(start: tuple[float, float], end: tuple[float, float], alpha: float) -> tuple[float, float]:
        return (
            start[0] + alpha * (end[0] - start[0]),
            start[1] + alpha * (end[1] - start[1]),
        )

    def _update_metrics(self):
        progress = self._curriculum_progress()
        if "curriculum_progress" not in self.metrics:
            self.metrics["curriculum_progress"] = torch.zeros(self.num_envs, device=self.device)
        if "command_x" not in self.metrics:
            self.metrics["command_x"] = torch.zeros(self.num_envs, device=self.device)
        if "command_yaw" not in self.metrics:
            self.metrics["command_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_progress"].fill_(progress)
        self.metrics["command_x"][:] = torch.abs(self._command[:, 0])
        self.metrics["command_yaw"][:] = torch.abs(self._command[:, 2])
        if "adaptive_sampling_strength" not in self.metrics:
            self.metrics["adaptive_sampling_strength"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adaptive_sampling_strength"].fill_(self._adaptive_sampling_strength())

        root_quat_w = self._env.scene["robot"].data.root_quat_w
        root_lin_vel_w = self._env.scene["robot"].data.root_lin_vel_w
        root_ang_vel_w = self._env.scene["robot"].data.root_ang_vel_w
        vel_b = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        lin_err = torch.abs(self._command[:, 0] - vel_b[:, 0])
        yaw_err = torch.abs(self._command[:, 2] - ang_vel_b[:, 2])
        tracking_score = torch.exp(-(lin_err / self.cfg.success_lin_vel_tolerance) ** 2) * torch.exp(
            -(yaw_err / self.cfg.success_ang_vel_tolerance) ** 2
        )
        moving_mask = (torch.abs(self._command[:, 0]) > 0.02) | (torch.abs(self._command[:, 2]) > 0.08)
        tracking_score = torch.where(moving_mask, tracking_score, torch.zeros_like(tracking_score))
        self._tracking_success_ema = (
            (1.0 - self.cfg.env_success_ema_alpha) * self._tracking_success_ema
            + self.cfg.env_success_ema_alpha * tracking_score
        )

    def _adaptive_sampling_strength(self) -> float:
        progress = self._curriculum_progress()
        if progress <= self.cfg.adaptive_sampling_start_progress:
            return 0.0
        denom = max(1e-6, 1.0 - self.cfg.adaptive_sampling_start_progress)
        alpha = (progress - self.cfg.adaptive_sampling_start_progress) / denom
        return float(torch.clamp(torch.tensor(alpha), 0.0, 1.0).item())

    def _update_mastery_from_previous_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        lin_bins = self._command_bins[env_ids, 0]
        ang_bins = self._command_bins[env_ids, 1]
        scores = self._tracking_success_ema[env_ids]
        prev_values = self._bin_mastery[lin_bins, ang_bins]
        self._bin_mastery[lin_bins, ang_bins] = (
            (1.0 - self.cfg.bin_mastery_ema_alpha) * prev_values + self.cfg.bin_mastery_ema_alpha * scores
        )

    def _sample_bin_indices(self, num_samples: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        strength = self._adaptive_sampling_strength()
        if strength <= 0.0:
            lin_bins = torch.randint(self.cfg.num_lin_bins, (num_samples,), device=device)
            ang_bins = torch.randint(self.cfg.num_ang_bins, (num_samples,), device=device)
            return lin_bins, ang_bins

        difficulty = torch.pow(
            torch.clamp(1.0 - self._bin_mastery, min=self.cfg.min_difficulty_floor),
            self.cfg.adaptive_sampling_temperature,
        )
        probs = difficulty / torch.sum(difficulty)
        uniform_probs = torch.full_like(probs, 1.0 / probs.numel())
        probs = (1.0 - strength) * uniform_probs + strength * probs
        flat_ids = torch.multinomial(probs.flatten(), num_samples=num_samples, replacement=True)
        lin_bins = torch.div(flat_ids, self.cfg.num_ang_bins, rounding_mode="floor")
        ang_bins = flat_ids % self.cfg.num_ang_bins
        return lin_bins, ang_bins

    def _bin_to_range(
        self, bin_index: torch.Tensor, low: float, high: float, num_bins: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bin_size = (high - low) / num_bins
        start = low + bin_index.float() * bin_size
        end = start + bin_size
        return start, end

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._update_mastery_from_previous_commands(env_ids)
        progress = self._curriculum_progress()
        backward_progress = self._backward_progress()

        forward_range = self._lerp_tuple(self.cfg.initial_forward_range, self.cfg.final_forward_range, progress)
        backward_range = self._lerp_tuple(self.cfg.initial_backward_range, self.cfg.final_backward_range, backward_progress)
        ang_vel_range = self._lerp_tuple(self.cfg.initial_ang_vel_range, self.cfg.final_ang_vel_range, progress)

        standing_ratio = self.cfg.initial_standing_ratio + progress * (
            self.cfg.final_standing_ratio - self.cfg.initial_standing_ratio
        )
        forward_probability = self.cfg.initial_forward_probability + progress * (
            self.cfg.final_forward_probability - self.cfg.initial_forward_probability
        )

        num_samples = len(env_ids)
        commands = torch.zeros(num_samples, 3, device=self.device)

        standing_mask = torch.rand(num_samples, device=self.device) < standing_ratio
        moving_mask = ~standing_mask

        if moving_mask.any():
            moving_ids = moving_mask.nonzero(as_tuple=False).squeeze(-1)
            move_count = len(moving_ids)
            lin_bins, ang_bins = self._sample_bin_indices(move_count, self.device)

            x_low = backward_range[0]
            x_high = forward_range[1]
            lin_start, lin_end = self._bin_to_range(lin_bins, x_low, x_high, self.cfg.num_lin_bins)
            ang_start, ang_end = self._bin_to_range(ang_bins, ang_vel_range[0], ang_vel_range[1], self.cfg.num_ang_bins)

            sampled_x = torch.rand(move_count, device=self.device) * (lin_end - lin_start) + lin_start
            sampled_yaw = torch.rand(move_count, device=self.device) * (ang_end - ang_start) + ang_start

            valid_forward = sampled_x >= 0.0
            keep_forward = valid_forward & (torch.rand(move_count, device=self.device) < forward_probability)
            use_backward = ~keep_forward
            if use_backward.any() and backward_progress <= 0.0:
                sampled_x[use_backward] = torch.rand(use_backward.sum(), device=self.device) * (
                    forward_range[1] - forward_range[0]
                ) + forward_range[0]
            else:
                backward_vals = torch.rand(use_backward.sum(), device=self.device) * (
                    backward_range[1] - backward_range[0]
                ) + backward_range[0]
                sampled_x[use_backward] = torch.where(
                    sampled_x[use_backward] < 0.0,
                    sampled_x[use_backward],
                    backward_vals,
                )

            commands[moving_ids, 0] = torch.clamp(sampled_x, min=backward_range[0], max=forward_range[1])
            commands[moving_ids, 2] = sampled_yaw
            self._command_bins[env_ids[moving_ids], 0] = lin_bins
            self._command_bins[env_ids[moving_ids], 1] = ang_bins

        if standing_mask.any():
            standing_env_ids = env_ids[standing_mask]
            zero_lin_bin = int(torch.clamp(torch.tensor(self.cfg.zero_command_lin_bin), 0, self.cfg.num_lin_bins - 1).item())
            zero_ang_bin = int(torch.clamp(torch.tensor(self.cfg.zero_command_ang_bin), 0, self.cfg.num_ang_bins - 1).item())
            self._command_bins[standing_env_ids, 0] = zero_lin_bin
            self._command_bins[standing_env_ids, 1] = zero_ang_bin

        self._command[env_ids] = commands

    def _update_command(self):
        # 当前命令不需要在每个仿真步做额外滤波或闭环修正。
        return

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        goal_scale, goal_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        current_scale, current_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(base_pos_w, current_quat, current_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.cfg.goal_vel_visualizer_cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class CurriculumVelocityCommandCfg(CommandTermCfg):
    """逐步扩展命令范围的速度命令配置。"""

    class_type: type = CurriculumVelocityCommand

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (6.0, 10.0)

    initial_forward_range: tuple[float, float] = (0.0, 0.05)
    final_forward_range: tuple[float, float] = (0.0, 0.2)

    initial_backward_range: tuple[float, float] = (0.0, 0.0)
    final_backward_range: tuple[float, float] = (-0.1, 0.0)

    initial_ang_vel_range: tuple[float, float] = (-0.12, 0.12)
    final_ang_vel_range: tuple[float, float] = (-0.5, 0.5)

    initial_standing_ratio: float = 0.25
    final_standing_ratio: float = 0.05

    initial_forward_probability: float = 1.0
    final_forward_probability: float = 0.65

    num_lin_bins: int = 9
    num_ang_bins: int = 11
    adaptive_sampling_start_progress: float = 0.25
    adaptive_sampling_temperature: float = 2.0
    min_difficulty_floor: float = 0.08
    env_success_ema_alpha: float = 0.05
    bin_mastery_ema_alpha: float = 0.15
    success_lin_vel_tolerance: float = 0.08
    success_ang_vel_tolerance: float = 0.12
    zero_command_lin_bin: int = 4
    zero_command_ang_bin: int = 5

    curriculum_start_step: int = 2_000
    curriculum_end_step: int = 20_000
    backward_start_step: int = 7_000

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    debug_vis: bool = True
