# 原始 checkpoint 的剪枝与精度转换

此工具处理原始 PyTorch 训练权重，输出可由本仓库原生推理入口直接加载的 checkpoint，支持 acoustic、variance、all-in-one，以及 WaveNet、LYNXNet、LYNXNet2、DiT。它不处理 ONNX 图，也不进行需要重新训练的通道/注意力头稀疏剪枝。

## 使用方法

在仓库根目录、训练所用的 Python 环境运行：

```bash
python scripts/optimize_checkpoint.py \
  --checkpoint ckpt/my_model/model_ckpt_steps_100000.ckpt \
  --precision int8 \
  --output-dir ckpt/my_model_int8
```

`--config` 默认读取原 checkpoint 同目录的 `config.yaml`。配置中的字典路径按仓库根目录解析。输出目录必须不存在，原权重不会被覆盖。FP16/FP32 对应 `--precision fp16` / `--precision fp32`；FP32 可用于只剪枝而不降低权重精度。

联合 checkpoint 默认保留全部分支，也可以分别生成独立权重：

```bash
python scripts/optimize_checkpoint.py --checkpoint ckpt/joint/model_ckpt_steps_100000.ckpt --component acoustic --precision int8 --output-dir ckpt/acoustic_int8
python scripts/optimize_checkpoint.py --checkpoint ckpt/joint/model_ckpt_steps_100000.ckpt --component variance --precision fp16 --output-dir ckpt/variance_fp16
```

输出包含 `model_ckpt_steps_*.ckpt`、完整 `config.yaml`、复制后的字典、源配置同目录的 speaker/language map（如果存在），以及 `optimization.json`。使用现有命令推理：

```bash
python scripts/infer.py acoustic your_project.ds --exp my_model_int8
python scripts/infer.py variance your_project.ds --exp variance_fp16
```

all-in-one 输出目录可以同时交给 acoustic 和 variance 命令；各自只加载对应分支。输出仅供推理，加载器会拒绝用于恢复训练。部署导出入口也可以加载优化后的 checkpoint；导出验证情况以部署文档和验证脚本为准。

## “直接量化推理”的准确含义

INT8 权重在 checkpoint 和运行期间保持整数状态，不会在加载时还原整套 FP32/FP16 权重，也不会在每层运算前重建浮点矩阵。Linear 和一般 Conv1d 使用 `torch._int_mm`：浮点输入按行动态量化，INT8 × INT8 得到 INT32 累加结果，然后对**输出**应用尺度。深度卷积使用整数乘法与求和。GRU 的输入/隐藏门投影也走整数矩阵乘，sigmoid/tanh 和循环状态保留浮点。

Embedding 保留 INT8 表，查表后只把选中的输出行乘尺度，不还原整个浮点表。偏置、LayerNorm/PReLU 系数、ConvNeXt gamma 等小向量也存为 INT8，在浮点算子使用时临时乘尺度转换。attention、归一化、非线性、扩散采样和中间激活仍需浮点运算。因此这是覆盖整个推理模型的**混合精度 INT8 转换**，不是全程纯整数计算，也不是小向量始终以整数参与归一化。

覆盖范围包含公共 FS2、melody encoder、duration predictor、pitch/variance 生成模块、浅扩散辅助 decoder、speaker/language/retake embedding 和 stretch GRU，四种 backbone 共用同一转换入口。当前测试配置中，可学习参数的 INT8 存储覆盖率为 **100%**；固定 buffer 与量化尺度继续保持必要精度。大型矩阵保持整数存储和整数计算，不重建浮点权重矩阵。旧版仅量化矩阵的 checkpoint 仍可严格加载；扩大覆盖需从原始训练 checkpoint 重新转换。

例如最小 DiT acoustic 预设的完整模型约29.73M参数，其中 FS2 约13.91M、辅助 decoder 约14.47M，两者合计约95.5%。扩大覆盖后的 INT8 state 分别约13.99MB、14.54MB（含尺度）；这两个公共模块都计入完整模型的显存统计。只统计 diffusion/backbone 会遗漏大部分权重。

FP16 模式转换所选完整模型的全部浮点参数，加载时保持 FP16，并自动启用 FP16 autocast；固定扩散 schedule、位置编码等 buffer 保持原精度以避免数值问题。FP32 模式保持完整浮点参数为 FP32。两者都使用相同的推理剪枝规则。

动态激活量化与权重量化是不同口径；这一点也见 [PyTorch 的推理量化说明](https://docs.pytorch.org/ao/stable/workflows/inference.html)。本仓库实现不依赖 torchao，只使用 PyTorch 原生整数算子；并不意味着支持任意 PyTorch/CUDA 组合。

## 剪枝范围与覆盖率

删除 optimizer、scheduler、训练循环状态和模型前缀之外的 loss/metric 状态；选择独立分支时删除另一个分支。DDPM 中只用于初始化的 `alphas_cumprod_prev` 与只用于训练辅助函数的 `log_one_minus_alphas_cumprod` 也会移除。其他 sampler buffer 保留，因此不绑定到某一个采样算法。

不删除浅扩散的辅助 decoder、pitch/variance retake embedding、duration predictor、speaker/language embedding 或 stretch GRU：它们仍有原生推理用途。未实际构建的预测模块自然不占权重。没有随机置零参数，也没有宣称稀疏权重自动加速。

`optimization.json` 记录真实字节数、INT8 权重元素占原始参数的比例、剩余浮点参数逐项列表，以及被删除的 checkpoint 字段/state key。`module_storage` 按分支与模块列出原参数数、转换前后字节数和覆盖率，便于核查 FS2 与辅助 decoder。INT8 存储比例不能当作纯整数算子比例；固定 buffer 和量化尺度仍占空间。未支持的矩阵参数会明确报错，避免悄悄遗留大型浮点矩阵。

## 推理显存预算：仅供参考

对所选完整模型参数数 `N`，FP32 参数下限约 `4N` 字节、FP16 约 `2N`，INT8 参数约 `N`，另加 FP32 尺度与固定 buffer。实际驻留权重以 manifest 的 `optimized_model_bytes` 为准；文件大小还包含序列化元数据。

下表是理论权重下限，不是测量值；M=10^6，B=10^9，GiB=2^30 字节。参数数应包括完整 acoustic/variance 模型；all-in-one 要把两个分支相加，不能只数一个 backbone。

| 完整模型参数数 | FP32 权重 GiB | FP16 权重 GiB | INT8 矩阵为主时的下限 GiB |
|---|---:|---:|---:|
| 100M | 0.37 | 0.19 | 0.09 |
| 500M | 1.86 | 0.93 | 0.47 |
| 1B | 3.73 | 1.86 | 0.93 |
| 2B | 7.45 | 3.73 | 1.86 |

### 完整模型重新估算（扩大 INT8 覆盖后）

按配置实际构建 meta 模型并统计全部 state tensor 字节，不分配大型权重。口径：hidden384、词表65、acoustic 128 mel bins、模板启用的预测功能；acoustic 包含 FS2 与浅扩散辅助 decoder，variance 包含公共编码器、duration、pitch/曲线生成模块。variance 每个生成模块使用 scaling 预设层数的一半，与独立 variance scaling 口径一致。数字依赖配置，不等于任意实际 checkpoint 的大小。

**acoustic：完整模型权重与运行预算（GiB）**

| Backbone / 预设 | 完整参数 M | FP32 权重 | FP16 权重 | INT8 权重 | INT8 运行预算 |
|---|---:|---:|---:|---:|---:|
| wavenet / 最小 | 31.52 | 0.117 | 0.059 | 0.030 | 1.53～4.03 |
| wavenet / 最大 | 2059.21 | 7.671 | 3.836 | 1.920 | 3.42～5.92 |
| lynxnet / 最小 | 29.58 | 0.110 | 0.055 | 0.028 | 1.53～4.03 |
| lynxnet / 最大 | 1963.49 | 7.315 | 3.657 | 1.832 | 3.33～5.83 |
| lynxnet2 / 最小 | 31.09 | 0.116 | 0.058 | 0.029 | 1.53～4.03 |
| lynxnet2 / 最大 | 1998.10 | 7.443 | 3.722 | 1.864 | 3.36～5.86 |
| dit / 最小 | 29.73 | 0.111 | 0.055 | 0.028 | 1.53～4.03 |
| dit / 最大 | 2082.07 | 7.756 | 3.878 | 1.942 | 3.44～5.94 |

**variance：完整模型权重与运行预算（GiB）**

| Backbone / 预设 | 完整参数 M | FP32 权重 | FP16 权重 | INT8 权重 | INT8 运行预算 |
|---|---:|---:|---:|---:|---:|
| wavenet / 最小 | 18.06 | 0.067 | 0.034 | 0.017 | 1.52～4.02 |
| wavenet / 最大 | 2093.44 | 7.799 | 3.899 | 1.952 | 3.45～5.95 |
| lynxnet / 最小 | 16.11 | 0.060 | 0.030 | 0.015 | 1.52～4.02 |
| lynxnet / 最大 | 2002.38 | 7.459 | 3.730 | 1.868 | 3.37～5.87 |
| lynxnet2 / 最小 | 18.12 | 0.067 | 0.034 | 0.017 | 1.52～4.02 |
| lynxnet2 / 最大 | 2061.25 | 7.679 | 3.839 | 1.923 | 3.42～5.92 |
| dit / 最小 | 16.26 | 0.061 | 0.030 | 0.015 | 1.52～4.02 |
| dit / 最大 | 2006.87 | 7.476 | 3.738 | 1.872 | 3.37～5.87 |

运行预算为完整权重加下述估算工作区，**不是实测显存峰值**。all-in-one 权重需把所选两个分支相加；顺序推理与并发推理的工作区不同。各模块字节数、全部六档预设见 [完整估算 JSON](quantized_memory_estimates.json)。复现：

```bash
python scripts/estimate_quantized_memory.py --all-presets --output memory.json
```

对于 batch 1、约 768 帧、顺序采样、原生 SDPA、关闭 conditioner cache 的预算，可暂按“真实驻留权重 + 1～3 GiB”给 FP32/FP16 留运行空间；INT8 暂按“真实驻留权重 + 1.5～4 GiB”留空间，因为这里的激活仍为 FP32，还有动态量化、整数 accumulator、卷积窗口与整数权重重排临时量。**这些附加值只是预算带，未对大模型/长音频实测拟合，不能作为能否装进某张显卡的保证。** 声码器、CUDA 上下文、缓存、其他进程另计。大模型、长序列、SDPA math fallback 和高并发应实际测量。

卷积时间窗口和整数 GEMM 按最多 256 行/帧切块，避免一次展开整段音频；仍需要一个层的整数权重重排空间。加载器先在 CPU 上读取 checkpoint 并转换模块，再移到推理设备，避免先在 GPU 上驻留原始 FP32 模型与训练 optimizer。

## 速度参考与小模型实测

当前 INT8 路径优先确保矩阵保持整数与降低驻留权重。动态量化、重排、分块以及 Python GRU 循环都有开销，**不能假设 INT8 比 FP16/FP32 更快**。FP16 在支持半精度的 GPU 上、计算量足够大时可能获益；小模型、CPU 或算子启动占主导时可能变慢。INT8 对大型矩阵可能更有利，但本仓库尚无大型已训练模型的速度证据，不给未经验证的倍速承诺。

以下为 RTX 5060 Laptop、PyTorch 2.11.0+cu130 的随机权重小模型测量：all-in-one，hidden16，四种 backbone 均 L1/D16，4 mel bins、4 帧、batch 1、Reflow Euler 1 步、浅扩散 decoder 开启、无声码器。FS2/辅助 decoder 等完整模型仍约 15M 参数，不能理解成只有一个 tiny backbone。预热 3 次后测 5 次中位数。GPU 峰值是 PyTorch allocated，**不含 CUDA 上下文和驱动占用**。这组样本是检查开销的例子，不代表正常歌曲或模型规模的性能。

| Backbone | FP32 / FP16 / INT8 延迟 ms | FP32 / FP16 / INT8 权重 MiB | FP32 / FP16 / INT8 allocated 峰值 MiB |
|---|---:|---:|---:|
| wavenet | 37.43 / 44.41 / 99.74 | 57.96 / 28.98 / 14.60 | 72.91 / 43.56 / 31.24 |
| lynxnet | 48.72 / 36.27 / 103.68 | 57.96 / 28.98 / 14.60 | 72.92 / 43.57 / 31.26 |
| lynxnet2 | 30.71 / 45.67 / 91.64 | 57.95 / 28.97 / 14.60 | 72.90 / 43.56 / 31.25 |
| dit | 31.80 / 46.67 / 106.48 | 57.97 / 28.99 / 14.60 | 72.92 / 43.57 / 31.26 |

扩大覆盖后的 INT8 在这组条件下权重减少约75%，allocated 峰值减少约57%；延迟仍增加，不能据此承诺加速。小 backbone 的差别受到公共 FS2/辅助 decoder 的运行占比影响。完整数据见 [benchmark JSON](checkpoint_optimization_benchmark.json)；复现：

```bash
python scripts/benchmark_checkpoint_optimization.py --output benchmark.json --iterations 5
```

## Backbone 量化友好度：结构参考排序

按大型矩阵占比、是否易映射到整数 GEMM、卷积展开开销，结构上的参考排序为 **DiT ≳ LYNXNet2 > LYNXNet > WaveNet**。这不是音质量化耐受度或上面 tiny 配置的速度排名，也不保证不同训练权重保持该顺序。

| 排序 | Backbone | 原因与限制 |
|---|---|---|
| 1 | DiT | Linear 密集，适合整数 GEMM；attention、RoPE、归一化仍浮点，adaptive modulation 对误差可能敏感。 |
| 2 | LYNXNet2 | 通道投影多用 Linear；深度卷积和门控仍有开销，conditioner cache 配置会改变热点。 |
| 3 | LYNXNet | 点卷积可映射 GEMM，深度卷积走整数路径；激活/门控与布局转换影响收益。 |
| 4 | WaveNet | 扩张卷积要采集窗口，门控、残差和 skip 累加较多；小层容易被量化与启动开销拖慢。 |

FS2、melody encoder、duration predictor 和辅助 decoder 同样会影响最终速度。variance 包含多个生成模块，有 stretch GRU 时，整数 GRU 的 Python 循环可能成为新的热点。acoustic 与 variance 的覆盖率、显存和速度应分别测量。

## 面向友好量化与部署的选型建议

如果准备训练新模型，同时重视量化、维护成本和部署便利，GPU 场景建议先以**适中规模的 LYNXNet2 + FP16** 建立部署基线；CPU 场景在 **FP32 与 INT8** 中选择；如果重点是较大规模的矩阵量化，可以优先评估 **DiT + INT8**。这是基于当前结构和实现的工程建议，尚未通过同规模、已训练模型的音质与部署性能对照证明。已有模型不建议仅为上述排序重新训练或更换 backbone。

本文将 CPU 部署推荐明确限定为 FP32/INT8，不把 FP16 作为 CPU 部署选项。这里是兼容性与性能的选型口径，并非断言所有 CPU/PyTorch 组合都无法执行 FP16：前面的 CPU FP16 小模型验证确实通过，但能执行不代表有原生硬件加速或部署收益，也不改变本节推荐。PyTorch 也列出了 CPU 的 FP16 autocast 算子支持，见 [CPU AMP 算子说明](https://docs.pytorch.org/docs/2.14/amp.html#cpu-op-specific-behavior)；实际支持与收益仍应以目标硬件、算子和后端为准。

“矩阵量化友好”和“整体部署方便”是两个维度：DiT 的 Linear 密集，因此前面的结构排序靠前；但其 attention、RoPE、动态序列长度与目标后端支持仍需验证。LYNXNet2 可作为不依赖 attention 算子的候选，不过深度卷积支持与性能同样需要实测，不能据此宣称它在所有设备上最快。

| 目标 | 推荐起点 | 不推荐的选择或假设 |
|---|---|---|
| GPU 上尽快建立稳定部署 | 先比较 FP32 与 FP16 的完整推理、音质和延迟，达标后选 FP16；新模型可从适中规模 LYNXNet2 起步 | 默认认为 INT8 更快；只凭量化文件大小决定部署方案 |
| 显存或模型分发体积受限 | 评估原生 INT8；矩阵密集的新模型优先比较 DiT、LYNXNet2，核对 manifest 覆盖率与目标设备峰值 | 把权重缩小 75% 当作总显存也缩小 75%；为量化刻意堆到 1B/2B 参数 |
| CPU、小模型或短音频低延迟 | 只在 FP32/INT8 中选择：优先 FP32 建立基线，INT8 在内存、体积或实测延迟有收益且音质达标时采用；使用满足音质要求的较小模型 | 选 FP16 作为 CPU 部署方案；直接套用 GPU 排序；默认认为动态 INT8 更快 |
| 创作者友好、效果够用即可 | 新模型从较小规模 LYNXNet/LYNXNet2 起步，先保证训练与试听迭代顺畅；GPU 优先比较 FP16，CPU 优先 FP32，资源不足再评估 INT8 | 为理论上限选择巨大 DiT；为了极限量化牺牲音高/时长可编辑性；在没有资源压力时增加量化调试成本 |
| 只部署 acoustic 或 variance | 用 `--component` 分别生成独立推理权重，各自测量，并保留需要的 predictor | 部署时同时驻留两个不需要的分支；删除仍用于推理的辅助 decoder/retake embedding |
| ONNX / LibTorch 部署 | 优化 checkpoint 可交给新增部署入口，按 [部署文档](DeploymentExports.md) 验证整数图或 `.pt`；先建立浮点对照 | 假定不同引擎、GPU provider 或 LibTorch 版本都支持并加速所有整数循环；省略真实歌曲验证 |
| 音质优先、INT8 对照不达标 | GPU 回退该模型到 FP16 或 FP32，CPU 回退 FP32，保留原始权重作为对照 | 强行量化全部浮点算子；只检查输出有限而不检查音高、时长与试听结果 |

### 创作者友好：优先可用与容易迭代

这个场景的建议与“最大化矩阵量化收益”不同：优先选择创作者能稳定训练、试听和修改的模型，音准、时长、咬字与可编辑性达到用途要求即可。**新项目可先评估小规模 LYNXNet/LYNXNet2；已有模型效果够用时继续使用原 backbone。** 这不是已经证明两者音质优于 DiT/WaveNet 的结论，也不意味着单靠小模型就能弥补数据不足。

- **模型规模与数据**：先处理录音、标注、音域覆盖和验证集，再按实际不足调整容量。用最小的、能够满足成品需求的配置起步，不为了量化友好直接选择 1B/2B，也不在效果已经足够时继续堆参数。
- **精度选择**：GPU 从 FP32 对照到 FP16；CPU 直接从 FP32 起步。只有模型装不下、分发体积受限，或者目标设备实测 INT8 更合适时再量化。若 INT8 影响创作体验，GPU 回退 FP16/FP32、CPU 回退 FP32。
- **创作控制**：保留实际需要的 pitch/duration、retake 和曲线控制。若 stretch 是工作流所需功能，保留并测量 GRU 开销；若训练前已确定不需要，再评估关闭。不要为了减少几个模块而让创作者失去常用编辑能力。
- **训练与分发**：acoustic/variance 可分开训练、定位问题和更新；联合训练是否采用由训练资源与维护需要决定。联合权重分发时也可以用 `--component` 拆成独立推理权重，不要求使用者同时驻留全部模块。
- **验收方式**：用创作者常用歌曲做短句预览和完整成品试听，确认等待时间、内存、音准、字头和编辑结果可接受。可以在已验证的采样配置中分别记录预览与成品参数；减少预览步数之前先确认结果仍有参考价值。预览与成品参数需要自行记录和验证。

因此，创作者场景的默认顺序是“已有可用模型或较小模型 → 原精度功能基线 → GPU FP16 / CPU FP32 → 有明确资源需求时再评估 INT8”，而不是优先追求最高 INT8 覆盖率。这里的建议同样仅供参考。

### 训练前可以做的配置选择

- 优先控制模型规模，分别预算 acoustic、variance、辅助 decoder 和声码器。按目标设备做完整链路测量，不能只测 backbone；低延迟需求尤其要关注 variance 的 duration/pitch/曲线预测和声码器耗时。
- 隐藏宽度/投影宽度可优先选择 32 或 64 的倍数，以减少当前整数 GEMM 对齐填充；DiT 同时必须满足 attention head 等原有配置约束。对齐只减少额外开销，不保证速度或音质。不要修改已训练 checkpoint 的宽度来“对齐”。
- 只训练业务实际需要的预测项。若无需 stretch 功能，可以在**训练前**评估关闭 `use_stretch_embed`，减少 GRU 的部署负担；需要该功能时应保留。不要在训练完成后直接关闭开关并删权重来替代重新验证。
- 浅扩散可能减少生成部分的工作量，但会引入辅助 decoder；应比较完整配置的音质、采样步数和延迟后决定。不能把“模块更少”或“采样更少”单独当成更快的结论。
- conditioner cache、并发数和长音频切段按目标负载逐项测量。缓存可能用更多显存换取时间，切段要验证边界连贯性；不要一次更改精度、采样器、步数和分段方式，否则难以定位退化原因。

### 部署落地顺序

1. 保留原始 checkpoint，先用 FP32 纯剪枝版本确定功能与音质基线；固定 `.ds`、seed、采样器和步数。
2. GPU 比较 FP16，再在确有显存/体积需求时比较 INT8；CPU 只比较 FP32 与 INT8。真实歌曲应覆盖长短句、音域、不同说话人/语言，以及实际启用的 retake/stretch 功能。
3. 在目标设备分别记录 acoustic、variance、声码器和完整链路的预热后延迟、峰值显存及音质；根据业务限制选择精度，不按理论量化排序替代测量。
4. 固定已验证的仓库版本、PyTorch/CUDA 与设备组合，随模型保存 config、字典、映射、manifest 和对照结果。原生/LibTorch INT8 依赖私有 API，跨环境升级后应重新运行验证；ONNX 使用标准整数算子，但各目标引擎的算子、循环与 provider 支持仍需验证。

当前实现提供原生、ONNX 与 TorchScript 的整数权重路径，不依赖 torchao，也没有接入 QAT 或校准工具。量化收益会受到矩阵形状、算子启动和融合开销影响，见 [PyTorch 的量化性能说明](https://docs.pytorch.org/ao/stable/workflows/inference.html)。目标引擎的校准与精度调试仍需独立评估，见 [ONNX Runtime 量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。本节建议仅供参考。

## 验证与适用范围

```bash
python scripts/validate_checkpoint_optimization.py
```

验证包含 CPU/CUDA 上四种 backbone × DDPM/Reflow × 三种精度共 48 组完整 acoustic/variance 推理，整数 Linear/普通/分组/深度卷积及双向多层 GRU 的数值对照（卷积另测513帧跨分块边界及脚本化前后一致性），真实 checkpoint 转换与严格重载，逐模块100%参数存储覆盖率断言，联合分支提取，现有推理类的加载，FP32 纯剪枝前后同随机种子结果完全一致，以及拒绝恢复训练。部署导出检查与限制见 [Deployment Exports](DeploymentExports.md)。

该验证使用随机小模型，检查算子、有限输出与接口兼容性，不代表训练后音质验证。INT8 为无校准的逐输出通道对称权重量化、逐行动态激活量化；音高、时长与扩散迭代误差可能积累。正式使用时用同一批 `.ds`、固定 seed/采样配置对照原模型，检查音高/时长、mel 差异并试听。不要仅凭文件缩小或有限输出判断量化成功保持音质。

INT8 使用私有 `torch._int_mm` API，已在上述 PyTorch 2.11 CPU/CUDA 环境验证；其他版本/设备需运行验证脚本，不支持时明确报错，不退回整层浮点权重计算。FP16 同样应在目标环境验证。大型模型的转换需要 CPU 内存容纳原 checkpoint、原模型与输出状态；当前脚本不是流式转换工具，2B 模型应预留足够 RAM。
