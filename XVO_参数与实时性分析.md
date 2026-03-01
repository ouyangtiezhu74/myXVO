# XVO 参数量与模块拆解（基于仓库现有 checkpoint 参数清单）

## 数据来源与前置检查
- 仓库中未发现任何 `.zip/.tar/.gz/.pdf` 文件，因此无法在本地执行“解压压缩包”与“直接读取 PDF 原文”这两步。
- 但 `README.md` 给出了论文 PDF 链接（ICCV 2023）。
- 参数统计基于仓库内现成的 `checkpoint_param_list.txt`（由 `inspect_ckpt.py` 导出的 state_dict 参数形状清单）。

## 总参数量
- 按 `checkpoint_param_list.txt` 中 331 个参数张量的形状累加，得到：
  - **31,489,523 参数**（约 **31.49M**）

## 模块级参数拆分
以下为按模块名前缀聚合后的参数分布：

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| encoder.maskflownet.MaskFlownet_S | 10,514,256 | 33.390% |
| encoder.maskflownet.head_or_extra | 10,141,460 | 32.206% |
| transformer.patch_embed | 5,013,760 | 15.922% |
| transformer.blocks | 3,159,040 | 10.032% |
| decoder.fc1 | 2,621,568 | 8.325% |
| transformer.pos_embed | 20,480 | 0.065% |
| decoder.fc2 | 16,512 | 0.052% |
| decoder.rot | 1,161 | 0.004% |
| decoder.fc3 | 774 | 0.002% |
| transformer.norm | 512 | 0.002% |

> 说明：`encoder.maskflownet.head_or_extra` 表示 `encoder.maskflownet.` 前缀下、但不属于 `MaskFlownet_S` 子前缀的参数（例如额外卷积/融合/细化头）。

## 模块功能解读（结合参数命名）
- `encoder.maskflownet.MaskFlownet_S`：主光流编码-解码骨干，负责多尺度特征与流场/掩码预测。
- `encoder.maskflownet.head_or_extra`：与主干并行或后续的附加卷积、细化层、上采样/融合等组件。
- `transformer.patch_embed`：将编码特征投影到 Transformer token 表示。
- `transformer.blocks`：多层注意力+MLP 主体，用于全局建模。
- `decoder.fc1/fc2/fc3/rot`：回归头，输出 VO 位姿/旋转相关量。

## 实时性评价（基于仓库信息可得的“审慎结论”）
- 仓库代码和文档未提供明确 FPS / 延迟（ms）基准，也未给出统一硬件上的实时测试表。
- 参数量约 31.49M，且结构中包含较重的光流网络 + Transformer，推理代价通常不低。
- 因此，当前更合理的结论是：
  - **“是否实时”取决于部署硬件与输入分辨率，无法仅凭仓库现有文本直接下定论。**
- 若你需要严格结论，建议补充一次标准 benchmark：
  - 固定分辨率（如 640×384）、batch=1；
  - 固定设备（例如 RTX 3090 / Jetson）；
  - 统计端到端平均延迟与 P50/P90；
  - 输出 FPS，并与目标场景帧率（10Hz/20Hz/30Hz）比较。

## 用 FLOPs 评价实时性：可以，但不够
- **可以用 FLOPs 做“第一层筛选”**：FLOPs 越高，通常推理越慢；在同一硬件、同一实现下，FLOPs 与延迟常有相关性。
- **但 FLOPs 不能单独代表实时性**，因为实际延迟还受以下因素影响：
  - 显存带宽与访存模式（很多算子是 memory-bound）；
  - 算子实现与内核优化（cuDNN/TensorRT/ONNXRuntime 差异明显）；
  - 并行度与 batch size（VO 常用 batch=1，硬件利用率不一定高）；
  - 数据预处理/后处理与 I/O 开销（端到端时间可能显著高于纯模型前向）；
  - 精度与量化策略（FP32/FP16/INT8）以及是否启用 Tensor Core。
- **建议实践**：
  1. 先报告 Params + FLOPs（理论复杂度）；
  2. 再报告真实 latency/FPS（同一分辨率、batch=1、固定硬件）；
  3. 最终以端到端 FPS 是否达到业务帧率目标（如 ≥10Hz/20Hz/30Hz）作为“实时”判据。

> 结论：**FLOPs 可以用来评价“计算复杂度”，但不能替代真实延迟/FPS 测试。**

## 用“跑完整个数据集总耗时”评估实时性：部分有用，但不充分
- **可以作为吞吐量（throughput）参考**，例如：某数据集 N 帧总共跑了 T 秒，可换算平均 FPS = N / T。
- **但这不等同于严格的实时性测试**，原因是：
  - 平均值会掩盖抖动与长尾延迟（实际在线系统更关注单帧延迟与 P90/P95/P99）；
  - 离线批处理可能使用更大 batch、异步预取等策略，与在线逐帧推理场景不一致；
  - 数据加载、解码、缓存命中会显著影响总耗时，导致“模型前向时间”与“端到端时间”混杂。
- **更标准的实时评估建议**：
  1. 在线设置：batch=1、逐帧输入；
  2. 固定输入分辨率与硬件；
  3. 统计 warmup 后的单帧 latency（平均、P50、P90、P99）与端到端 FPS；
  4. 再补充“整套数据集总耗时”作为吞吐量指标。

> 结论：**“跑完整个数据集用了多久”可以作为补充指标，但不能单独定义是否实时。**

## 实时性测试应如何做（避免被数据加载/文件读写干扰）
- 你的担心是对的：**如果把数据加载、图像解码、磁盘读写都混在一起计时，结果会偏离“模型真实推理速度”**。
- 因此建议把实时性拆成两类指标并分别报告：
  1. **模型纯推理延迟（model latency）**：只统计 `forward`，不含 dataloader 与文件 I/O；
  2. **端到端延迟（E2E latency）**：从“拿到一帧输入”到“输出位姿结果”，可含必要预处理/后处理。

### 推荐测试流程（可复现）
1. 固定硬件与软件环境（GPU 型号、CUDA/cuDNN、PyTorch 版本）。
2. 固定输入设置（分辨率如 640×384、batch=1、FP32 或 FP16）。
3. 先 warmup（如 50~100 帧），再正式计时（如 500~2000 帧）。
4. 用 GPU 同步计时（`torch.cuda.synchronize()`）或 CUDA Event，避免异步误差。
5. 输出统计：mean / P50 / P90 / P99 latency + FPS。
6. 单独再测一次 E2E（含预处理/后处理），与纯模型 latency 并列汇报。

### 如何减少 I/O 造成的不准确
- 将测试数据预先放入内存或本地 NVMe，避免网络盘波动。
- 推理计时阶段关闭结果落盘，或把写文件放到异步线程并**不计入模型 latency**。
- dataloader 固定 `num_workers`、`pin_memory`，并在实验中保持一致。
- 若必须计入 I/O，请明确标注为 E2E 指标，不要与模型纯推理混用。

> 实务建议：发布结果时至少同时给出两组数字：
> - **Model-only**（最能反映网络本体速度）
> - **End-to-End**（最贴近实际部署体验）

