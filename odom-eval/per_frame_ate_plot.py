import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kitti_odometry import KittiEvalOdom


def compute_per_frame_ate_like_current_eval(gt_path: str, pred_path: str):
    """Compute per-frame ATE with the same protocol as current eval(alignment=None).

    Protocol consistency points:
      1) Same txt parsing via KittiEvalOdom.load_poses_from_txt.
      2) Same first-frame pose alignment used in KittiEvalOdom.eval.
      3) No 6DoF/7DoF/scale optimization.
      4) Same per-frame error definition as compute_ATE:
         ||gt[:3,3] - pred[:3,3]||_2
    """
    eval_tool = KittiEvalOdom()
    poses_pred = eval_tool.load_poses_from_txt(pred_path)
    poses_gt = eval_tool.load_poses_from_txt(gt_path)

    if len(poses_pred) == 0:
        raise ValueError(f"Empty prediction file: {pred_path}")

    # Match KittiEvalOdom.eval: align both trajectories to their own first frame.
    idx_0 = sorted(list(poses_pred.keys()))[0]
    if idx_0 not in poses_gt:
        raise KeyError(f"First prediction frame {idx_0} not found in GT.")

    pred_0 = poses_pred[idx_0]
    gt_0 = poses_gt[idx_0]

    for cnt in poses_pred:
        if cnt not in poses_gt:
            raise KeyError(f"Frame {cnt} exists in prediction but not in GT.")
        poses_pred[cnt] = np.linalg.inv(pred_0) @ poses_pred[cnt]
        poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

    frame_ids = []
    errors = []
    for i in poses_pred:
        gt_xyz = poses_gt[i][:3, 3]
        pred_xyz = poses_pred[i][:3, 3]
        err = float(np.sqrt(np.sum((gt_xyz - pred_xyz) ** 2)))
        frame_ids.append(i)
        errors.append(err)

    errors_np = np.asarray(errors, dtype=float)
    rmse = float(np.sqrt(np.mean(errors_np ** 2)))
    std = float(np.std(errors_np))
    mean = float(np.mean(errors_np))
    median = float(np.median(errors_np))

    return frame_ids, errors_np, {
        "rmse": rmse,
        "std": std,
        "mean": mean,
        "median": median,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-frame ATE with the same logic as current odom eval (alignment=None)."
    )
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory txt path")
    parser.add_argument("--pred", required=True, help="Predicted trajectory txt path")
    parser.add_argument("--out", required=True, help="Output figure path, e.g. ./ate.png")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    frame_ids, errors, stats = compute_per_frame_ate_like_current_eval(args.gt, args.pred)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(frame_ids, errors, linewidth=1.2, label="Per-frame ATE")
    plt.xlabel("Frame")
    plt.ylabel("ATE (m)")

    title = args.title if args.title else f"Per-frame ATE\nGT: {Path(args.gt).name}, Pred: {Path(args.pred).name}"
    plt.title(title)
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"RMSE: {stats['rmse']:.6f} m\n"
        f"STD: {stats['std']:.6f} m\n"
        f"Mean: {stats['mean']:.6f} m\n"
        f"Median: {stats['median']:.6f} m"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved figure:", out_path)
    print(
        "Stats -> "
        f"RMSE: {stats['rmse']:.6f}, "
        f"STD: {stats['std']:.6f}, "
        f"Mean: {stats['mean']:.6f}, "
        f"Median: {stats['median']:.6f}"
    )


if __name__ == "__main__":
    main()
