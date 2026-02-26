import pandas as pd
import os
import time
import glob
from kitti_odometry import KittiEvalOdom
import json
from tqdm import tqdm
from pathlib import Path
def add_summary_inplace(ep_results: dict):
    t_err, r_err, ate, s_err = [], [], [], []
    n = 0
    for seq, v2 in ep_results.items():
        if str(seq).startswith("__"):  # 跳过已有的内部字段
            continue
        n += 1
        if v2[0] != 0:
            t_err.append(v2[0])
        if v2[1] != 0:
            r_err.append(v2[1])
        ate.append(v2[2])
        s_err.append(v2[3])

    def mean_or_none(x):
        return (sum(x) / len(x)) if len(x) > 0 else None

    ep_results["__summary__"] = {
        "t_err": mean_or_none(t_err),
        "r_err": mean_or_none(r_err),
        "ate":   mean_or_none(ate),
        "s_err": mean_or_none(s_err),
        "n_seq": n,
    }

eval_dirs = ['train_kitti_sl_A_20260118-00020809',]
#用脚本位置推导路径：无论你从哪里运行都不会错
THIS_DIR = Path(__file__).resolve().parent            # .../odom-eval
ROOT_DIR = THIS_DIR.parent                            # .../XVO-main
RESULTS_ROOT = ROOT_DIR / "results"                   # .../XVO-main/results

kitti_gt_dir = THIS_DIR / "dataset" / "kitti" / "gt_poses"
nusc_gt_dir  = THIS_DIR / "dataset" / "nusc"  / "gt_poses"
argo2_gt_dir = THIS_DIR / "dataset" / "argo2" / "gt_poses"
eval_tool = KittiEvalOdom()
#kitti_gt_dir = "./dataset/kitti/gt_poses/"
#nusc_gt_dir = "./dataset/nusc/gt_poses/"
#argo2_gt_dir = "./dataset/argo2/gt_poses/"

scenes = ['KITTI', 'NUSC', 'ARGO2']
for _dir in eval_dirs:
    print(_dir)
    os.makedirs(f'./evaluation_results/{_dir}', exist_ok=True)

    for scene in scenes:
        if "KITTI" in scene:
            gt_dir = kitti_gt_dir
        elif "NUSC" in scene:
            gt_dir = nusc_gt_dir
        elif "ARGO2" in scene:
            gt_dir = argo2_gt_dir

        epochs = sorted((RESULTS_ROOT / _dir).glob("*"))
        
        json_path = f"./evaluation_results/{_dir}/{scene}.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as json_file:
                results = json.load(json_file)
        else:
            results = {}

        for ep in tqdm(epochs):


            result_dir = Path(ep) / scene
            if not result_dir.exists():
                continue
            # else:
            #     print(result_dir)
            ep_name = Path(ep).name
            if ep_name in results.keys():
                continue

            ep_results = eval_tool.eval(gt_dir, result_dir, alignment=None)
            
            results[ep_name] = ep_results
        
        results = dict(sorted(results.items()))
        with open(json_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
        
    sets = glob.glob(f"./evaluation_results/{_dir}/*.json")
    stats = {}
    keys = set()
    for _set in sets:
        with open(_set, "r") as json_file:
            _result = json.load(json_file)
        stats[Path(_set).stem] = _result
        keys = keys | set(_result.keys())
    keys = sorted(list(keys))

    for ep in keys:
        print(ep, end=" ")
        for k, v in stats.items():
            print(f"{k}:", end=" ")
            if ep in v.keys():
                t_err = []
                r_err = []
                ate = []
                s_err = []
                for _, v2 in v[ep].items():
                    if v2[0] != 0:
                        t_err.append(v2[0])
                    if v2[1] != 0:
                        r_err.append(v2[1])
                    ate.append(v2[2])
                    s_err.append(v2[3])
                print(", ".join(
                    f"{(sum(val)/len(val)):.2f}" if len(val) > 0 else "/" 
                    for val in [t_err, r_err, ate, s_err]
                ), end=" ")
            else:
                print("/, /, /, /", end=" ")
        print("")
