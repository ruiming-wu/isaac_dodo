# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from collections.abc import Sequence

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--resume_save_interval",
    type=int,
    default=1,
    help="Checkpoint save interval to use during resume runs (set <=0 to keep config value).",
)
parser.add_argument(
    "--early_stop_on_resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable metric-based early stopping during resume runs.",
)
parser.add_argument(
    "--early_stop_metric",
    type=str,
    default="Episode_Reward/track_lin_vel_xy_exp",
    help="Episode metric key used for resume early stopping.",
)
parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=3,
    help="Number of consecutive non-improving metric updates before early stop.",
)
parser.add_argument(
    "--early_stop_min_metric_updates",
    type=int,
    default=4,
    help="Minimum valid metric updates before early stopping can trigger.",
)
parser.add_argument(
    "--early_stop_min_delta",
    type=float,
    default=0.002,
    help="Minimum improvement required to reset early-stop patience.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import math
import shutil
import torch
from datetime import datetime

import omni
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_dodo.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class _EarlyStopTriggered(RuntimeError):
    """Internal signal used to stop short resume runs once metric degrades."""


def _extract_episode_metric(ep_infos: Sequence[dict], key: str, device: torch.device) -> float | None:
    """Compute mean episode metric from the current logging batch."""
    if not ep_infos:
        return None

    values = torch.tensor([], device=device)
    for ep_info in ep_infos:
        if key not in ep_info:
            continue
        value = ep_info[key]
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], device=device)
        if len(value.shape) == 0:
            value = value.unsqueeze(0)
        values = torch.cat((values, value.to(device)))

    if values.numel() == 0:
        return None
    return float(torch.mean(values).item())


def _write_best_resume_alias(log_dir: str, best_it: int):
    """Create best_resume.pt alias to the best iteration checkpoint."""
    best_model_name = f"model_{best_it}.pt"
    best_model_path = os.path.join(log_dir, best_model_name)
    alias_path = os.path.join(log_dir, "best_resume.pt")
    if not os.path.exists(best_model_path):
        print(f"[INFO]: Best checkpoint missing, alias skipped: {best_model_path}")
        return

    if os.path.lexists(alias_path):
        os.remove(alias_path)
    try:
        # Relative link keeps run folder self-contained if moved.
        os.symlink(best_model_name, alias_path)
        print(f"[INFO]: Created best checkpoint alias: {alias_path} -> {best_model_name}")
    except OSError:
        shutil.copy2(best_model_path, alias_path)
        print(f"[INFO]: Symlink unavailable, copied best checkpoint to: {alias_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    early_stop_state = None
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # Resume can be brittle if exploration std stays high while policy is already near a local gait.
        # Use a semi-conservative setup: moderate LR reduction + std clamp (without freezing std).
        if agent_cfg.resume:
            resume_lr_scale = 0.5
            resume_lr_min = 5.0e-4
            resume_noise_std_max = 0.20

            # Conservative LR on resume: lower the algorithm LR and every optimizer param group.
            if hasattr(runner.alg, "learning_rate"):
                lr_before = float(runner.alg.learning_rate)
                lr_after = max(resume_lr_min, lr_before * resume_lr_scale)
                runner.alg.learning_rate = lr_after
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = lr_after
                print(f"[INFO]: Resume learning rate scaled: {lr_before:.6f} -> {lr_after:.6f}")

            # Clamp exploration noise, but keep std trainable so adaptation is still possible.
            policy = runner.alg.policy
            if hasattr(policy, "std"):
                std_before = policy.std.detach().mean().item()
                policy.std.data.clamp_(max=resume_noise_std_max)
                std_after = policy.std.detach().mean().item()
                print(f"[INFO]: Resume action noise std mean clamped: {std_before:.4f} -> {std_after:.4f}")
            elif hasattr(policy, "log_std"):
                std_before = torch.exp(policy.log_std.detach()).mean().item()
                policy.log_std.data.clamp_(max=math.log(resume_noise_std_max))
                std_after = torch.exp(policy.log_std.detach()).mean().item()
                print(f"[INFO]: Resume action noise std mean clamped: {std_before:.4f} -> {std_after:.4f}")
            if args_cli.resume_save_interval > 0:
                runner.save_interval = args_cli.resume_save_interval
                print(f"[INFO]: Resume save interval set to: {runner.save_interval}")

            # Metric-based early stop for short resume runs:
            # stop when key metric fails to improve for several updates.
            if args_cli.early_stop_on_resume and args_cli.early_stop_patience > 0:
                metric_key = args_cli.early_stop_metric
                patience = args_cli.early_stop_patience
                min_updates = max(1, args_cli.early_stop_min_metric_updates)
                min_delta = args_cli.early_stop_min_delta
                original_log = runner.log
                early_stop_state = {
                    "activated": False,
                    "updates": 0,
                    "best": float("-inf"),
                    "best_it": runner.current_learning_iteration,
                    "bad_updates": 0,
                }

                def _log_with_early_stop(locs: dict, width: int = 80, pad: int = 35):
                    original_log(locs, width=width, pad=pad)
                    # Only start monitoring after real episode statistics become available.
                    # Before that, most episode metrics stay zero and can trigger false early stops.
                    has_episode_stats = len(locs["rewbuffer"]) > 0
                    if not has_episode_stats:
                        if not early_stop_state["activated"]:
                            print("[INFO]: Early-stop monitor waiting for first valid episode statistics...")
                        return
                    if not early_stop_state["activated"]:
                        early_stop_state["activated"] = True
                        print("[INFO]: Early-stop monitor activated (valid episode statistics detected).")

                    metric_value = _extract_episode_metric(locs["ep_infos"], metric_key, runner.device)
                    if metric_value is None:
                        return

                    early_stop_state["updates"] += 1
                    it = int(locs["it"])
                    if metric_value > early_stop_state["best"] + min_delta:
                        early_stop_state["best"] = metric_value
                        early_stop_state["best_it"] = it
                        early_stop_state["bad_updates"] = 0
                        best_iter_path = os.path.join(log_dir, f"model_{it}.pt")
                        if not os.path.exists(best_iter_path):
                            runner.save(best_iter_path)
                            print(f"[INFO]: Saved new best checkpoint: {best_iter_path}")
                    else:
                        early_stop_state["bad_updates"] += 1

                    print(
                        "[INFO]: Early-stop monitor "
                        f"{metric_key}={metric_value:.4f} | best={early_stop_state['best']:.4f} "
                        f"(it {early_stop_state['best_it']}) | bad={early_stop_state['bad_updates']}/{patience}"
                    )

                    if (
                        early_stop_state["updates"] >= min_updates
                        and early_stop_state["bad_updates"] >= patience
                    ):
                        raise _EarlyStopTriggered(
                            f"Early-stop triggered at iter {it} on metric '{metric_key}'. "
                            f"Best value {early_stop_state['best']:.4f} at iter {early_stop_state['best_it']}."
                        )

                runner.log = _log_with_early_stop
                print(
                    "[INFO]: Resume early-stop enabled "
                    f"(metric='{metric_key}', patience={patience}, min_updates={min_updates}, min_delta={min_delta})."
                )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Fresh training benefits from random episode progress, but resuming a mature gait can
    # destabilize immediately if we randomize episode lengths on the first rollout.
    init_at_random_ep_len = not (agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation")

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=init_at_random_ep_len)
    except _EarlyStopTriggered as err:
        print(f"[INFO]: {err}")
        early_stop_path = os.path.join(log_dir, f"model_early_stop_{runner.current_learning_iteration}.pt")
        runner.save(early_stop_path)
        print(f"[INFO]: Saved early-stop checkpoint to: {early_stop_path}")
    finally:
        if early_stop_state is not None and early_stop_state["activated"] and early_stop_state["updates"] > 0:
            _write_best_resume_alias(log_dir, int(early_stop_state["best_it"]))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
