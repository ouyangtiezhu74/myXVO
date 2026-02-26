from pathlib import Path
import torch
from collections import Counter

# 让它能 import 你的 model.py / model 包
from model import VOModel

def human_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"

ckpt_dir = Path(r"D:\XVO-main\saved_models\xvo_kitti_sl")
pt_files = sorted(ckpt_dir.glob("*.pt"))
if not pt_files:
    raise FileNotFoundError(f"No .pt found in {ckpt_dir}")

# 现在你删掉了另一个 checkpoint，这里一般只剩 1 个
ckpt_path = pt_files[0]
print("=== checkpoint file ===")
print(ckpt_path)
print()

ckpt = torch.load(str(ckpt_path), map_location="cpu")
print("=== checkpoint type ===")
print(type(ckpt))
print()

# 顶层键
if isinstance(ckpt, dict):
    print("=== top-level keys ===")
    for k in ckpt.keys():
        v = ckpt[k]
        t = type(v).__name__
        extra = ""
        if torch.is_tensor(v):
            extra = f" shape={tuple(v.shape)} dtype={v.dtype}"
        elif isinstance(v, dict):
            extra = f" dict(len={len(v)})"
        elif isinstance(v, (list, tuple)):
            extra = f" {type(v).__name__}(len={len(v)})"
        else:
            try:
                extra = f" value={v}"
            except Exception:
                extra = ""
        print(f"- {k}: {t}{extra}")
    print()
else:
    raise TypeError("Checkpoint is not a dict; cannot inspect keys.")

# 取 model_state_dict（有些 checkpoint 直接就是 state_dict）
if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
    sd = ckpt["model_state_dict"]
    sd_name = "model_state_dict"
elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    sd = ckpt
    sd_name = "(checkpoint itself as state_dict)"
else:
    raise KeyError("Cannot find model_state_dict and checkpoint is not a plain state_dict.")

print(f"=== using state dict: {sd_name} ===")
print("param tensors:", len(sd))

# 统计参数量/大小
total_numel = 0
total_bytes = 0
dtypes = Counter()
shapes = Counter()
module_prefix = Counter()

for name, t in sd.items():
    if not torch.is_tensor(t):
        continue
    total_numel += t.numel()
    total_bytes += t.numel() * t.element_size()
    dtypes[str(t.dtype)] += 1
    shapes[str(tuple(t.shape))] += 1
    module_prefix[name.split(".")[0]] += 1

print("total parameters (numel):", total_numel)
print("tensor storage size:", human_bytes(total_bytes))
print()

print("=== dtype counts (how many tensors) ===")
for k, v in dtypes.most_common():
    print(f"{k}: {v}")
print()

print("=== top module prefixes (first-level) ===")
for k, v in module_prefix.most_common(30):
    print(f"{k}: {v}")
print()

# 跟当前 VOModel 的 state_dict 对齐检查（这最关键）
model = VOModel()
model_sd = model.state_dict()

missing = [k for k in model_sd.keys() if k not in sd]
unexpected = [k for k in sd.keys() if k not in model_sd]

print("=== state_dict alignment vs current VOModel() ===")
print("model tensors:", len(model_sd))
print("missing keys in checkpoint:", len(missing))
print("unexpected keys in checkpoint:", len(unexpected))
print()

if missing:
    print("missing examples (first 30):")
    for k in missing[:30]:
        print("  ", k)
    print()

if unexpected:
    print("unexpected examples (first 30):")
    for k in unexpected[:30]:
        print("  ", k)
    print()

# 把完整参数清单写到文件（不写数值，只写形状/类型）
out_txt = Path("checkpoint_param_list.txt")
with out_txt.open("w", encoding="utf-8") as f:
    f.write(f"checkpoint: {ckpt_path}\n")
    f.write(f"state_dict: {sd_name}\n\n")
    for name, t in sd.items():
        if torch.is_tensor(t):
            f.write(f"{name}\tshape={tuple(t.shape)}\tdtype={t.dtype}\n")
        else:
            f.write(f"{name}\t(type={type(t)})\n")

print(f"wrote: {out_txt.resolve()}")
