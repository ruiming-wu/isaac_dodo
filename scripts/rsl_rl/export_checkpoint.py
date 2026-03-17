#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an RSL-RL checkpoint to TorchScript and/or ONNX without launching Isaac Sim."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import yaml

from rsl_rl.runners import OnPolicyRunner


class _DummyVecEnv:
    """Minimal VecEnv-like stub used to reconstruct the policy for checkpoint loading."""

    def __init__(self, num_obs: int, num_privileged_obs: int, num_actions: int, device: str):
        self.num_envs = 1
        self.num_actions = num_actions
        self.device = device
        self._num_obs = num_obs
        self._num_privileged_obs = num_privileged_obs

    def get_observations(self):
        obs = torch.zeros(self.num_envs, self._num_obs, device=self.device)
        extras = {"observations": {}}
        if self._num_privileged_obs != self._num_obs:
            extras["observations"]["critic"] = torch.zeros(self.num_envs, self._num_privileged_obs, device=self.device)
        return obs, extras


class _TorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module, normalizer: torch.nn.Module | None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        self.actor = copy.deepcopy(policy.actor)
        if self.is_recurrent:
            self.rnn = copy.deepcopy(policy.memory_a.rnn)
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(x.squeeze(0))

    def forward_gru(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        return self.actor(x.squeeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0


class _OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module, normalizer: torch.nn.Module | None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        self.actor = copy.deepcopy(policy.actor)
        if self.is_recurrent:
            self.rnn = copy.deepcopy(policy.memory_a.rnn)
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward_lstm(self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        obs = self.normalizer(obs)
        out, (h_out, c_out) = self.rnn(obs.unsqueeze(0), (h_in, c_in))
        return self.actor(out.squeeze(0)), h_out, c_out

    def forward_gru(self, obs: torch.Tensor, h_in: torch.Tensor):
        obs = self.normalizer(obs)
        out, h_out = self.rnn(obs.unsqueeze(0), h_in)
        return self.actor(out.squeeze(0)), h_out

    def forward(self, obs: torch.Tensor):
        return self.actor(self.normalizer(obs))


def _infer_dims(checkpoint_path: Path) -> tuple[int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    num_obs = model_state["actor.0.weight"].shape[1]
    num_privileged_obs = model_state["critic.0.weight"].shape[1]
    actor_weight_keys = sorted(key for key in model_state if key.startswith("actor.") and key.endswith(".weight"))
    num_actions = model_state[actor_weight_keys[-1]].shape[0]
    return num_obs, num_privileged_obs, num_actions


def _load_train_cfg(agent_yaml_path: Path) -> dict:
    with agent_yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_runner(checkpoint_path: Path, agent_yaml_path: Path) -> OnPolicyRunner:
    num_obs, num_privileged_obs, num_actions = _infer_dims(checkpoint_path)
    train_cfg = _load_train_cfg(agent_yaml_path)
    dummy_env = _DummyVecEnv(num_obs, num_privileged_obs, num_actions, device="cpu")
    runner = OnPolicyRunner(dummy_env, train_cfg, log_dir=None, device="cpu")
    runner.load(str(checkpoint_path), load_optimizer=False)
    runner.eval_mode()
    return runner


def _export_jit(policy: torch.nn.Module, normalizer: torch.nn.Module | None, output_path: Path):
    exporter = _TorchPolicyExporter(policy, normalizer).to("cpu")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(exporter).save(str(output_path))


def _export_onnx(policy: torch.nn.Module, normalizer: torch.nn.Module | None, output_path: Path):
    exporter = _OnnxPolicyExporter(policy, normalizer).to("cpu")
    exporter.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exporter.is_recurrent:
        obs = torch.zeros(1, exporter.rnn.input_size)
        hidden = torch.zeros(exporter.rnn.num_layers, 1, exporter.rnn.hidden_size)
        if exporter.rnn_type == "lstm":
            cell = torch.zeros(exporter.rnn.num_layers, 1, exporter.rnn.hidden_size)
            torch.onnx.export(
                exporter,
                (obs, hidden, cell),
                str(output_path),
                export_params=True,
                opset_version=11,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            torch.onnx.export(
                exporter,
                (obs, hidden),
                str(output_path),
                export_params=True,
                opset_version=11,
                input_names=["obs", "h_in"],
                output_names=["actions", "h_out"],
                dynamic_axes={},
            )
    else:
        obs = torch.zeros(1, exporter.actor[0].in_features)
        torch.onnx.export(
            exporter,
            obs,
            str(output_path),
            export_params=True,
            opset_version=11,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def _resolve_agent_yaml(checkpoint_path: Path, agent_yaml: str | None) -> Path:
    if agent_yaml is not None:
        return Path(agent_yaml).expanduser().resolve()
    return checkpoint_path.parent / "params" / "agent.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint, e.g. model_180.pt")
    parser.add_argument("--agent-yaml", default=None, help="Optional path to the saved agent.yaml")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to <run>/exported")
    parser.add_argument("--formats", nargs="+", choices=["jit", "onnx"], default=["jit", "onnx"])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    agent_yaml_path = _resolve_agent_yaml(checkpoint_path, args.agent_yaml)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint_path.parent / "exported"

    runner = _build_runner(checkpoint_path, agent_yaml_path)
    policy = runner.alg.policy
    normalizer = runner.obs_normalizer if getattr(runner, "empirical_normalization", False) else None

    if "jit" in args.formats:
        jit_path = output_dir / "policy.pt"
        _export_jit(policy, normalizer, jit_path)
        print(f"[INFO] Exported TorchScript policy to: {jit_path}")

    if "onnx" in args.formats:
        onnx_path = output_dir / "policy.onnx"
        _export_onnx(policy, normalizer, onnx_path)
        print(f"[INFO] Exported ONNX policy to: {onnx_path}")


if __name__ == "__main__":
    main()
