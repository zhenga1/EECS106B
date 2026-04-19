#!/usr/bin/env python3
"""Plot global trajectory generated from DroneRace track config.

Usage (run from repo root EECS106B/):
  python scripts/plot_global_trajectory.py \
      --config cfg/task/DroneRace.yaml \
      --method catmull_rom \
      --num-points 300 \
      --out debug/traj.png

Add --show to open an interactive window when running with a display.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make omni_drones importable when the script is run directly from any cwd.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import torch
import yaml

from omni_drones.utils.global_trajectory_planner import (
    generate_trajectory_from_gate_poses,
)
from omni_drones.utils.torch import euler_to_quaternion


def _load_gate_poses(track_cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    gate_keys = sorted(track_cfg.keys(), key=lambda x: int(x))

    gate_positions = []
    gate_orientations = []
    for key in gate_keys:
        gate = track_cfg[key]
        pos = gate.get("pos", [0.0, 0.0, 1.0])
        yaw = float(gate.get("yaw", 0.0))

        gate_positions.append(pos)
        quat = euler_to_quaternion(torch.tensor([0.0, 0.0, yaw], dtype=torch.float32))
        gate_orientations.append(quat)

    gate_positions_t = torch.tensor(gate_positions, dtype=torch.float32)
    gate_orientations_t = torch.stack(gate_orientations, dim=0)
    return gate_positions_t, gate_orientations_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generated global trajectory")
    parser.add_argument(
        "--config",
        default="cfg/task/DroneRace.yaml",
        help="Path to task yaml with track_config",
    )
    parser.add_argument(
        "--method",
        default="catmull_rom",
        choices=["linear", "spline", "catmull_rom"],
        help="Trajectory interpolation method",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=200,
        help="Number of trajectory points",
    )
    parser.add_argument(
        "--out",
        default="debug/global_trajectory_plot.png",
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (requires display)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task_cfg = cfg.get("task", cfg)
    track_cfg = task_cfg.get("track_config")
    if not isinstance(track_cfg, dict) or len(track_cfg) < 2:
        raise ValueError("track_config with at least 2 gates is required")

    gate_positions, gate_orientations = _load_gate_poses(track_cfg)

    traj = generate_trajectory_from_gate_poses(
        gate_positions=gate_positions,
        gate_orientations=gate_orientations,
        method=args.method,
        num_points=args.num_points,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gate_positions[:, 0].cpu(),
        gate_positions[:, 1].cpu(),
        gate_positions[:, 2].cpu(),
        "ko--",
        label="gates",
    )
    ax.plot(
        traj[:, 0].cpu(),
        traj[:, 1].cpu(),
        traj[:, 2].cpu(),
        "r-",
        linewidth=2.0,
        label=f"trajectory ({args.method})",
    )

    ax.set_title("Global Trajectory Through Gates")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
