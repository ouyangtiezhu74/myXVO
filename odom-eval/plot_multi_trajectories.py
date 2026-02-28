"""Plot GT and multiple predicted trajectories in one figure.

Example:
    python odom-eval/plot_multi_trajectories.py \
        --gt /path/to/gt.txt \
        --pred /path/to/method_a.txt --name Method-A \
        --pred /path/to/method_b.txt --name Method-B \
        --out /tmp/multi_traj.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_poses_from_txt(file_name: str):
    """Load KITTI-style trajectory txt into {frame_idx: 4x4 pose}."""
    poses = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for cnt, line in enumerate(f.readlines()):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ") if i != ""]
            with_idx = len(line_split) == 13
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row * 4 + col + with_idx]
            frame_idx = int(line_split[0]) if with_idx else cnt
            poses[frame_idx] = P
    return poses


def align_to_first_frame(poses: dict[int, np.ndarray]):
    """Left-align poses so first frame becomes identity."""
    aligned = {k: v.copy() for k, v in poses.items()}
    idx0 = sorted(aligned.keys())[0]
    p0_inv = np.linalg.inv(aligned[idx0])
    for k in aligned:
        aligned[k] = p0_inv @ aligned[k]
    return aligned


def poses_to_xz(poses: dict[int, np.ndarray]):
    """Convert trajectory poses to x-z array sorted by frame index."""
    keys = sorted(poses.keys())
    xz = np.array([[poses[k][0, 3], poses[k][2, 3]] for k in keys], dtype=float)
    return xz


def parse_args():
    parser = argparse.ArgumentParser(description="Plot multiple trajectories with GT.")
    parser.add_argument("--gt", required=True, help="GT trajectory txt path.")
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        help="Pred trajectory txt path, can be used multiple times.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Legend name for each --pred, same count/order as --pred.",
    )
    parser.add_argument(
        "--out",
        default="odom-eval/multi_trajectory_plot.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Trajectory Comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    args = parser.parse_args()

    if len(args.pred) == 0:
        raise ValueError("Please provide at least one --pred trajectory file.")
    if len(args.pred) != len(args.name):
        raise ValueError("--pred and --name must have the same count and order.")
    return args


def main():
    args = parse_args()

    gt_poses = align_to_first_frame(load_poses_from_txt(args.gt))
    gt_xz = poses_to_xz(gt_poses)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # GT fixed style: black dashed line
    ax.plot(
        gt_xz[:, 0],
        gt_xz[:, 1],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Ground Truth",
    )

    # Use matplotlib color cycle so each prediction has different color
    cmap = plt.get_cmap("tab10")
    for idx, (pred_path, name) in enumerate(zip(args.pred, args.name)):
        pred_poses = align_to_first_frame(load_poses_from_txt(pred_path))
        pred_xz = poses_to_xz(pred_poses)
        color = cmap(idx % 10)
        ax.plot(
            pred_xz[:, 0],
            pred_xz[:, 1],
            color=color,
            linewidth=1.8,
            label=name,
        )

    ax.set_title(args.title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
