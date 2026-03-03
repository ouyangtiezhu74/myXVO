import argparse
import csv
import json
from pathlib import Path

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
    # Keep iteration order exactly aligned with current compute_ATE behavior
    # (dict insertion order from load_poses_from_txt / file line order).
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


def save_curve_csv(csv_path: Path, frame_ids, errors):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ate_m"])
        for frame, err in zip(frame_ids, errors):
            writer.writerow([frame, float(err)])


def plot_curve_png(out_png: Path, title: str, frame_ids, errors, stats: dict):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(frame_ids, errors, linewidth=1.2, label="Per-frame ATE")
    plt.xlabel("Frame")
    plt.ylabel("ATE (m)")
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
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_single(gt_path: Path, pred_path: Path, out_dir: Path, title: str | None = None):
    frame_ids, errors, stats = compute_per_frame_ate_like_current_eval(str(gt_path), str(pred_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pred_path.stem
    out_png = out_dir / f"{stem}_ate.png"
    out_csv = out_dir / f"{stem}_ate.csv"
    out_json = out_dir / f"{stem}_ate_stats.json"

    plot_title = title if title else f"Per-frame ATE\nGT: {gt_path.name}, Pred: {pred_path.name}"
    plot_curve_png(out_png, plot_title, frame_ids, errors, stats)
    save_curve_csv(out_csv, frame_ids, errors)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved figure: {out_png}")
    print(f"Saved per-frame csv: {out_csv}")
    print(f"Saved stats json: {out_json}")
    print(
        "Stats -> "
        f"RMSE: {stats['rmse']:.6f}, "
        f"STD: {stats['std']:.6f}, "
        f"Mean: {stats['mean']:.6f}, "
        f"Median: {stats['median']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-frame ATE with the same logic as current odom eval (alignment=None)."
    )
    parser.add_argument("--gt", help="Ground-truth trajectory txt path (single mode)")
    parser.add_argument("--pred", help="Predicted trajectory txt path (single mode)")
    parser.add_argument("--out", help="Output directory (single mode)")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--gt-dir", help="GT directory for batch mode")
    parser.add_argument("--pred-dir", help="Prediction directory for batch mode")
    parser.add_argument("--seqs", nargs="+", help="Sequence IDs, e.g. 3 4 5 6 7 10")
    parser.add_argument("--out-dir", help="Output directory for batch mode")
    args = parser.parse_args()

    is_single = args.gt and args.pred and args.out
    is_batch = args.gt_dir and args.pred_dir and args.out_dir and args.seqs

    if is_single and is_batch:
        raise ValueError("Please use either single mode (--gt/--pred/--out) or batch mode (--gt-dir/--pred-dir/--seqs/--out-dir), not both.")
    if not is_single and not is_batch:
        raise ValueError("Missing required arguments. Use single mode or batch mode.")

    if is_single:
        run_single(Path(args.gt), Path(args.pred), Path(args.out), title=args.title)
        return

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    for seq in args.seqs:
        seq_stem = f"{int(seq):02d}" if seq.isdigit() else seq
        gt_path = gt_dir / f"{seq_stem}.txt"
        pred_path = pred_dir / f"{seq_stem}.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        run_single(gt_path, pred_path, out_dir, title=f"Seq {seq_stem} Per-frame ATE")


if __name__ == "__main__":
    main()
