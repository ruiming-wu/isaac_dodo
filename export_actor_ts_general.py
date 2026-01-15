#!/usr/bin/env python3
"""
python3 /home/rczh/workspace/isaac_dodo/export_actor_ts_general.py \
  --ckpt /home/rczh/workspace/isaac_dodo/trained_models/target_pos/2026-01-06_16-08-04/model_1100.pt \
  --out  /home/rczh/workspace/isaac_dodo/trained_models/target_pos/2026-01-06_16-08-04/policy_actor_ts.pt \
  --activation elu
"""
import torch
import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


# ---------- helpers ----------
def pick_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("relu",):
        return nn.ReLU()
    if name in ("elu",):
        return nn.ELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class LinearLayerSpec:
    idx: int                 # actor.<idx>.weight
    out_features: int
    in_features: int
    weight_key: str
    bias_key: Optional[str]


def extract_actor_linear_specs(sd: Dict[str, torch.Tensor]) -> List[LinearLayerSpec]:
    """
    Find keys like:
      actor.0.weight, actor.0.bias
      actor.2.weight, actor.2.bias
    Sort by numeric index (0,2,4,...)
    """
    w_pat = re.compile(r"^actor\.(\d+)\.weight$")
    b_pat = re.compile(r"^actor\.(\d+)\.bias$")

    weights: Dict[int, str] = {}
    biases: Dict[int, str] = {}

    for k in sd.keys():
        m = w_pat.match(k)
        if m:
            i = int(m.group(1))
            weights[i] = k
            continue
        m = b_pat.match(k)
        if m:
            i = int(m.group(1))
            biases[i] = k

    specs: List[LinearLayerSpec] = []
    for i in sorted(weights.keys()):
        w_key = weights[i]
        W = sd[w_key]
        if W.ndim != 2:
            # skip non-linear weights (shouldn't happen in your current case)
            continue
        out_f, in_f = int(W.shape[0]), int(W.shape[1])
        b_key = biases.get(i, None)
        specs.append(LinearLayerSpec(
            idx=i,
            out_features=out_f,
            in_features=in_f,
            weight_key=w_key,
            bias_key=b_key
        ))

    if not specs:
        raise RuntimeError("No actor.<n>.weight found. This exporter supports MLP actors stored as actor.<idx>.*")

    # sanity check: consecutive connectivity
    for a, b in zip(specs, specs[1:]):
        if a.out_features != b.in_features:
            raise RuntimeError(
                f"Actor layer dims not chainable: actor.{a.idx} out={a.out_features} "
                f"!= actor.{b.idx} in={b.in_features}"
            )

    return specs


class ActorMLP(nn.Module):
    """
    Build Sequential(Linear, act, Linear, act, ..., Linear)
    from LinearLayerSpec list.
    """
    def __init__(self, specs: List[LinearLayerSpec], activation: str = "tanh"):
        super().__init__()
        act = pick_activation(activation)

        layers: List[nn.Module] = []
        for li, spec in enumerate(specs):
            layers.append(nn.Linear(spec.in_features, spec.out_features))
            # last linear no activation
            if li != len(specs) - 1:
                layers.append(act.__class__())  # create fresh module each time

        self.net = nn.Sequential(*layers)

        # helpful metadata
        self.obs_dim = specs[0].in_features
        self.action_dim = specs[-1].out_features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorWithStd(nn.Module):
    """
    Optional: output (mu, std) for Gaussian policy.
    `std` can be stored as a parameter named 'std' in state_dict.
    Many implementations store log_std; your key name is literally 'std'.
    We'll provide two modes:
      - interpret as log_std (std = exp(std_param))
      - interpret as std directly (std = clamp(std_param, ...))
    """
    def __init__(self, actor: ActorMLP, std_param: torch.Tensor, std_mode: str = "exp", std_min: float = 1e-6):
        super().__init__()
        self.actor = actor
        self.std_mode = std_mode
        self.std_min = std_min

        # register as parameter for TorchScript export consistency
        self.std = nn.Parameter(std_param.clone().detach())

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.actor(obs)
        if self.std_mode == "exp":
            std = torch.exp(self.std)
        elif self.std_mode == "direct":
            std = self.std
        else:
            raise RuntimeError(f"Unknown std_mode: {self.std_mode}")
        std = torch.clamp(std, min=self.std_min)
        # broadcast to batch
        std = std.expand_as(mu)
        return mu, std


def load_actor_weights_into_model(sd: Dict[str, torch.Tensor], actor_model: ActorMLP, specs: List[LinearLayerSpec]) -> None:
    """
    Map:
      actor.<idx>.weight -> net.<k>.weight  (k is the position in Sequential)
    Sequential layout is: [Linear, Act, Linear, Act, ..., Linear]
    So Linear modules are at positions: 0,2,4,...
    """
    mapped: Dict[str, torch.Tensor] = {}
    linear_pos = 0
    for li, spec in enumerate(specs):
        seq_linear_idx = 2 * li  # 0,2,4,...
        mapped[f"net.{seq_linear_idx}.weight"] = sd[spec.weight_key]
        if spec.bias_key is not None:
            mapped[f"net.{seq_linear_idx}.bias"] = sd[spec.bias_key]

    missing, unexpected = actor_model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        # 如果这里出现内容，说明 state_dict 格式和假设不一致
        raise RuntimeError(f"load_state_dict mismatch. missing={missing}, unexpected={unexpected}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt containing model_state_dict")
    ap.add_argument("--out", required=True, help="Output TorchScript .pt path")
    ap.add_argument("--activation", default="tanh", help="tanh|relu|elu|silu|identity")
    ap.add_argument("--export_std", action="store_true", help="Export (mu,std) instead of only mu")
    ap.add_argument("--std_mode", default="exp", help="exp or direct (how to interpret ckpt['std'])")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (export on cpu recommended)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint must be a dict with key 'model_state_dict'.")

    sd: Dict[str, torch.Tensor] = ckpt["model_state_dict"]

    # 1) infer actor linear stack
    specs = extract_actor_linear_specs(sd)
    obs_dim = specs[0].in_features
    act_dim = specs[-1].out_features
    print(f"[infer] actor layers: {[ (s.in_features, s.out_features) for s in specs ]}")
    print(f"[infer] obs_dim={obs_dim}, action_dim={act_dim}")

    # 2) build actor model
    actor = ActorMLP(specs, activation=args.activation)
    load_actor_weights_into_model(sd, actor, specs)
    actor.eval()

    device = torch.device(args.device)
    actor.to(device)

    # 3) choose export wrapper
    model_to_export: nn.Module
    if args.export_std:
        if "std" not in sd:
            raise RuntimeError("export_std requested but 'std' not found in model_state_dict.")
        std_param = sd["std"].to(device)
        model_to_export = ActorWithStd(actor, std_param=std_param, std_mode=args.std_mode)
        model_to_export.eval()
        print(f"[export] exporting mu+std with std_mode={args.std_mode}")
    else:
        model_to_export = actor
        print("[export] exporting mu-only")

    # 4) trace export (robust for MLP)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
    with torch.no_grad():
        _ = model_to_export(dummy)

    ts = torch.jit.trace(model_to_export, dummy)
    ts.save(args.out)
    print(f"[done] saved TorchScript to: {args.out}")


if __name__ == "__main__":
    main()
