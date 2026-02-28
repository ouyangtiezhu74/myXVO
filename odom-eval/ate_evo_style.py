"""Generate EVO-style ATE diagnostics for one trajectory pair.

Usage:
    python odom-eval/ate_evo_style.py \
        --pred path/to/pred.txt \
        --gt path/to/gt.txt \
        --out-dir odom-eval/ate_outputs
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
    """Left-align poses so frame-0 (or first frame) becomes identity."""
    aligned = {k: v.copy() for k, v in poses.items()}
    idx0 = sorted(aligned.keys())[0]
    p0_inv = np.linalg.inv(aligned[idx0])
    for k in aligned:
        aligned[k] = p0_inv @ aligned[k]
    return aligned


def compute_framewise_ate(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]):
    """Compute per-frame translational ATE on overlapping indices."""
    common = sorted(set(gt.keys()) & set(pred.keys()))
    if len(common) == 0:
        raise ValueError("No overlapping frame indices between GT and prediction.")

    errors = []
    gt_xyz = []
    pred_xyz = []
    for idx in common:
        g = gt[idx][:3, 3]
        p = pred[idx][:3, 3]
        err = np.linalg.norm(g - p)
        errors.append(err)
        gt_xyz.append(g)
        pred_xyz.append(p)

    errors = np.asarray(errors)
    gt_xyz = np.asarray(gt_xyz)
    pred_xyz = np.asarray(pred_xyz)
    return common, errors, gt_xyz, pred_xyz


def summarize(errors: np.ndarray):
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean = float(np.mean(errors))
    median = float(np.median(errors))
    std = float(np.std(errors))
    vmin = float(np.min(errors))
    vmax = float(np.max(errors))
    return {
        "rmse": rmse,
        "mean": mean,
        "median": median,
        "std": std,
        "min": vmin,
        "max": vmax,
    }


def plot_ate_curve(frame_ids, errors, stats, save_path: Path):
    """Per-frame ATE curve with EVO-style summary overlays."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame_ids, errors, color="tab:blue", linewidth=1.5, label="Per-frame ATE (m)")

    mean = stats["mean"]
    std = stats["std"]
    ax.axhline(stats["rmse"], color="tab:red", linestyle="--", linewidth=1.4, label=f"RMSE: {stats['rmse']:.4f} m")
    ax.axhline(stats["median"], color="tab:green", linestyle=":", linewidth=1.4, label=f"Median: {stats['median']:.4f} m")
    ax.axhline(mean, color="tab:purple", linestyle="-.", linewidth=1.2, label=f"Mean: {mean:.4f} m")
    ax.fill_between(
        frame_ids,
        np.full(len(frame_ids), mean - std),
        np.full(len(frame_ids), mean + std),
        color="tab:purple",
        alpha=0.15,
        label=f"Std interval: μ±σ = [{mean - std:.4f}, {mean + std:.4f}] m",
    )

    ax.set_title("Absolute Trajectory Error (ATE) per Frame")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("ATE translation error (m)")
    ax.grid(True, alpha=0.3)

    text = (
        f"min={stats['min']:.4f} m\n"
        f"max={stats['max']:.4f} m\n"
        f"std={stats['std']:.4f} m"
    )
    ax.text(
        0.99,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_colored_trajectory(gt_xyz, pred_xyz, errors, stats, save_path: Path):
    """Trajectory map colored by per-frame ATE magnitude."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="black", linewidth=1.4, label="Ground Truth")

    sc = ax.scatter(
        pred_xyz[:, 0],
        pred_xyz[:, 2],
        c=errors,
        cmap="viridis",
        s=14,
        label="Prediction (colored by ATE)",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ATE (m)")

    ax.set_title("Trajectory Error Colormap (EVO-style)")
    ax.set_xlabel("X (m) [world frame]")
    ax.set_ylabel("Z (m) [world frame]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    metrics_line = (
        f"RMSE={stats['rmse']:.4f}m | Median={stats['median']:.4f}m | "
        f"Std={stats['std']:.4f}m | Min={stats['min']:.4f}m | Max={stats['max']:.4f}m"
    )
    ax.text(
        0.5,
        -0.08,
        metrics_line,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate per-frame ATE diagnostics (EVO-like).")
    parser.add_argument("--pred", required=True, help="Prediction trajectory txt path.")
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory txt path.")
    parser.add_argument("--out-dir", default="odom-eval/ate_outputs", help="Directory to save plots.")
    parser.add_argument("--tag", default=None, help="Optional output tag name.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = align_to_first_frame(load_poses_from_txt(args.pred))
    gt = align_to_first_frame(load_poses_from_txt(args.gt))
    frame_ids, errors, gt_xyz, pred_xyz = compute_framewise_ate(gt, pred)
    stats = summarize(errors)

    tag = args.tag if args.tag else Path(args.pred).stem
    curve_path = out_dir / f"{tag}_ate_curve.png"
    traj_path = out_dir / f"{tag}_trajectory_colored.png"
    stats_path = out_dir / f"{tag}_stats.txt"

    plot_ate_curve(frame_ids, errors, stats, curve_path)
    plot_colored_trajectory(gt_xyz, pred_xyz, errors, stats, traj_path)

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("ATE statistics\n")
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            f.write(f"{k}: {stats[k]:.6f}\n")

    print(f"Saved: {curve_path}")
    print(f"Saved: {traj_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
