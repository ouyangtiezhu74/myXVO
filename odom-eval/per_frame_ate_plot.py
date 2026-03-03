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
    parser.add_argument("--gt", default=None, help="Ground-truth trajectory txt path")
    parser.add_argument("--pred", default=None, help="Predicted trajectory txt path")
    parser.add_argument("--out", default=None, help="Output figure path, e.g. ./ate.png")
    parser.add_argument("--gt-dir", default=None, help="Ground-truth trajectory directory")
    parser.add_argument("--pred-dir", default=None, help="Predicted trajectory directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for multi-sequence plots")
    parser.add_argument("--seqs", nargs="+", default=None, help="Sequence ids for directory mode, e.g. --seqs 3 4 5")
    args = parser.parse_args()

    single_mode = all([args.gt, args.pred, args.out])
    dir_mode = all([args.gt_dir, args.pred_dir, args.out_dir])

    if not single_mode and not dir_mode:
        parser.error(
            "Either provide --gt/--pred/--out for single-file mode, "
            "or provide --gt-dir/--pred-dir/--out-dir for directory mode."
        )

    if single_mode and dir_mode:
        parser.error("Please choose only one mode: single-file or directory mode.")

    def render_plot(gt_path: str, pred_path: str, out: str):
        frame_ids, errors, stats = compute_per_frame_ate_like_current_eval(gt_path, pred_path)

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 5))
        plt.plot(frame_ids, errors, linewidth=1.2, label="Per-frame ATE")
        plt.xlabel("Frame")
        plt.ylabel("ATE (m)")

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

    if single_mode:
        render_plot(args.gt, args.pred, args.out)
        return

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def resolve_seq_file(directory: Path, seq: str) -> Path:
        candidates = [
            directory / f"{seq}.txt",
            directory / f"{int(seq):02d}.txt" if str(seq).isdigit() else None,
        ]
        for cand in candidates:
            if cand is not None and cand.exists():
                return cand
        raise FileNotFoundError(f"Cannot find sequence {seq} in {directory}")

    if args.seqs is None:
        args.seqs = sorted([p.stem for p in pred_dir.glob("*.txt")])

    for seq in args.seqs:
        gt_path = resolve_seq_file(gt_dir, str(seq))
        pred_path = resolve_seq_file(pred_dir, str(seq))
        out_path = out_dir / f"{int(seq):02d}_per_frame_ate.png" if str(seq).isdigit() else out_dir / f"{seq}_per_frame_ate.png"
        print(f"\n[Sequence {seq}] GT={gt_path} PRED={pred_path}")
        render_plot(str(gt_path), str(pred_path), str(out_path))


if __name__ == "__main__":
    main()
