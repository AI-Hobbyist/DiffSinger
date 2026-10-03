# Model scaling：小型到约 2B 的 backbone 参考

原始 checkpoint 的 INT8/FP16/FP32 转换、公共 FS2/辅助 decoder 的量化覆盖率、完整 acoustic/variance 显存重估与速度参考见 [Checkpoint Optimization](CheckpointOptimization.md)；ONNX/LibTorch 导出见 [Deployment Exports](DeploymentExports.md)。本页原有的 FP16 理论预算不代表 INT8 路径的实测性能。

本文分别列出单 backbone、非 all-in-one 的完整 acoustic 模型，以及完整 variance 模型；每种 backbone 各有 6 个不同规模，覆盖小型到约 2B 参数。参数数量按当前仓库的真实构造器计算。**推理显存仅供参考，属于预算估算，不是实测峰值，也不是显卡容量保证。** 大型配置只验证了构造与参数计数，不代表已训练、音质更好或已经完成 2B GPU 推理。

## 统计口径与显存假设

- `hidden_size=384`、`out_dims=128`、`num_feats=1`，对应单个 acoustic backbone。M 为 10^6 参数，B 为 10^9 参数。GiB 为 2^30 字节。
- 参数数包含该 backbone 的输入/条件/时间投影、各层和输出投影；不包含 FS2、duration predictor、stretch GRU、辅助 decoder、声码器、说话人/语言 embedding，以及另外两个生成 backbone。
- 权重列假设所有 backbone 浮点参数以 FP16/BF16 存储，理论权重占用为 `参数数 × 2 / 2^30`。实际 buffer、分配器对齐与混合精度权重例外另计。
- 推理估算假设 batch 1、768 帧（44.1kHz / hop 512 下约 8.9 秒）、`eval()` + `no_grad()`、单个 backbone 驻留、顺序采样、无 CFG、无 conditioner cache，采用 `权重 GiB + 1～3 GiB` 的运行预算，预留 CUDA 上下文、临时激活、算子 workspace 与缓存。**1～3 GiB 是统一经验预算带，没有根据实际设备进行拟合。** 它不能预测具体显卡或后端的峰值。
- DiT 假设使用内存高效的 SDPA；ONNX adapter 的普通 attention 会物化 attention 矩阵。此表的预算不能直接套用到 ONNX、SDPA math fallback 或长序列。FP32、其他精度或保留 FP32 主权重也需重新计算；只开 autocast 不会把 FP32 驻留权重变成每参数 2 字节。

## 可复现参数表

`L` 对应 `backbone_args.num_layers`，`D` 对应 `num_channels`；其余固定参数见下一节。数值由 [scripts/model_scaling.py](../scripts/model_scaling.py) 的真实构造器在 meta device 上生成，不分配巨型权重。

| Backbone | L | D | 参数量 (M) | FP16/BF16 权重 (GiB) | 推理显存估算 (GiB，仅供参考) |
| --- | ---: | ---: | ---: | ---: | ---: |
| wavenet | 12 | 128 | 3.14 | 0.006 | 1.01–3.01 |
| wavenet | 20 | 384 | 33.92 | 0.063 | 1.06–3.06 |
| wavenet | 24 | 512 | 68.64 | 0.128 | 1.13–3.13 |
| wavenet | 32 | 896 | 260.89 | 0.486 | 1.49–3.49 |
| wavenet | 40 | 1472 | 845.56 | 1.575 | 2.57–4.57 |
| wavenet | 40 | 2304 | 2030.84 | 3.783 | 4.78–6.78 |
| lynxnet | 6 | 128 | 1.21 | 0.002 | 1.00–3.00 |
| lynxnet | 12 | 384 | 15.78 | 0.029 | 1.03–3.03 |
| lynxnet | 16 | 512 | 35.35 | 0.066 | 1.07–3.07 |
| lynxnet | 24 | 768 | 112.47 | 0.209 | 1.21–3.21 |
| lynxnet | 32 | 2048 | 1003.70 | 1.870 | 2.87–4.87 |
| lynxnet | 40 | 2560 | 1935.11 | 3.604 | 4.60–6.60 |
| lynxnet2 | 6 | 256 | 2.72 | 0.005 | 1.01–3.01 |
| lynxnet2 | 12 | 512 | 18.40 | 0.034 | 1.03–3.03 |
| lynxnet2 | 16 | 768 | 52.88 | 0.099 | 1.10–3.10 |
| lynxnet2 | 24 | 1024 | 135.84 | 0.253 | 1.25–3.25 |
| lynxnet2 | 32 | 2560 | 1105.86 | 2.060 | 3.06–5.06 |
| lynxnet2 | 40 | 3072 | 1969.72 | 3.669 | 4.67–6.67 |
| dit | 4 | 128 | 1.35 | 0.003 | 1.00–3.00 |
| dit | 8 | 384 | 22.07 | 0.041 | 1.04–3.04 |
| dit | 12 | 640 | 90.40 | 0.168 | 1.17–3.17 |
| dit | 16 | 1024 | 306.31 | 0.571 | 1.57–3.57 |
| dit | 24 | 1536 | 1028.23 | 1.915 | 2.92–4.92 |
| dit | 27 | 2048 | 2053.69 | 3.825 | 4.83–6.83 |

## 独立 acoustic 模型（非 all-in-one）

以下为完整 `DiffSingerAcoustic` 的参数数，包含声学 FS2、生成 backbone 和浅扩散辅助 decoder，不包含声码器。采用 [独立 acoustic DiT 模板](../configs/templates/config_acoustic_dit.yaml) 的其余设置：65 个词表 ID、hidden_size 384、4 层条件编码器、key_shift/speed 条件开启、其他曲线条件关闭、无说话人/语言 embedding，辅助 decoder 为 6 层、宽度 512 的 ConvNeXt。替换 backbone 后其余模块固定。

显存按完整声学模型的 FP16/BF16 权重加 1～3 GiB 运行预算估算，仍是 batch 1 / 768 帧 / 单模型驻留，**仅供参考**。声码器若同时在 GPU 上，需要再加其权重与工作区。最小模型仍包含固定的编码器与辅助 decoder，因而参数数大于单 backbone 表。

| Backbone | L | D | 完整模型参数量 (M) | 其中生成 backbone 合计 (M) | FP16/BF16 权重 (GiB) | 推理显存估算 (GiB，仅供参考) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wavenet | 12 | 128 | 31.52 | 3.14 | 0.059 | 1.06–3.06 |
| wavenet | 20 | 384 | 62.30 | 33.92 | 0.116 | 1.12–3.12 |
| wavenet | 24 | 512 | 97.02 | 68.64 | 0.181 | 1.18–3.18 |
| wavenet | 32 | 896 | 289.27 | 260.89 | 0.539 | 1.54–3.54 |
| wavenet | 40 | 1472 | 873.94 | 845.56 | 1.628 | 2.63–4.63 |
| wavenet | 40 | 2304 | 2059.21 | 2030.84 | 3.836 | 4.84–6.84 |
| lynxnet | 6 | 128 | 29.58 | 1.21 | 0.055 | 1.06–3.06 |
| lynxnet | 12 | 384 | 44.16 | 15.78 | 0.082 | 1.08–3.08 |
| lynxnet | 16 | 512 | 63.73 | 35.35 | 0.119 | 1.12–3.12 |
| lynxnet | 24 | 768 | 140.85 | 112.47 | 0.262 | 1.26–3.26 |
| lynxnet | 32 | 2048 | 1032.08 | 1003.70 | 1.922 | 2.92–4.92 |
| lynxnet | 40 | 2560 | 1963.49 | 1935.11 | 3.657 | 4.66–6.66 |
| lynxnet2 | 6 | 256 | 31.09 | 2.72 | 0.058 | 1.06–3.06 |
| lynxnet2 | 12 | 512 | 46.77 | 18.40 | 0.087 | 1.09–3.09 |
| lynxnet2 | 16 | 768 | 81.26 | 52.88 | 0.151 | 1.15–3.15 |
| lynxnet2 | 24 | 1024 | 164.22 | 135.84 | 0.306 | 1.31–3.31 |
| lynxnet2 | 32 | 2560 | 1134.24 | 1105.86 | 2.113 | 3.11–5.11 |
| lynxnet2 | 40 | 3072 | 1998.10 | 1969.72 | 3.722 | 4.72–6.72 |
| dit | 4 | 128 | 29.73 | 1.35 | 0.055 | 1.06–3.06 |
| dit | 8 | 384 | 50.44 | 22.07 | 0.094 | 1.09–3.09 |
| dit | 12 | 640 | 118.77 | 90.40 | 0.221 | 1.22–3.22 |
| dit | 16 | 1024 | 334.68 | 306.31 | 0.623 | 1.62–3.62 |
| dit | 24 | 1536 | 1056.61 | 1028.23 | 1.968 | 2.97–4.97 |
| dit | 27 | 2048 | 2082.07 | 2053.69 | 3.878 | 4.88–6.88 |

## 独立 variance 模型（非 all-in-one）

以下为完整 `DiffSingerVariance` 的参数数，包含条件/旋律编码器、duration predictor、pitch backbone、多曲线 backbone 及条件 embedding，不包含 acoustic 模型或声码器。采用 [独立 variance DiT 模板](../configs/templates/config_variance_dit.yaml) 的其余设置：65 个词表 ID、hidden_size 384、4 层条件编码器，dur/pitch/energy/breathiness/voicing/tension 全部开启，pitch repeat_bins 64、多曲线 total_repeat_bins 72，use_variance_scaling 开启、stretch embedding 关闭、无说话人/语言 embedding。

每一行的 **L/D 同时用于 pitch 和 multi-variance 两个独立 backbone**。为使完整 variance 模型覆盖小型到约 2B，而不是两个 2B backbone 相加到约 4B，每个生成器的层数采用单 backbone 表对应行的一半（向下取整，至少 1 层）。关闭某些预测器或启用 stretch embedding 后应重新计算，不能直接套用此表。

显存按完整 variance 模型的 FP16/BF16 权重加 1～3 GiB 运行预算估算；两个 backbone 同时驻留，顺序生成，batch 1 / 768 帧，**仅供参考**。若继续生成声学，需另计 acoustic/声码器的驻留与临时工作区。

| Backbone | L | D | 完整模型参数量 (M) | 其中生成 backbone 合计 (M) | FP16/BF16 权重 (GiB) | 推理显存估算 (GiB，仅供参考) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wavenet | 6 | 128 | 18.06 | 3.29 | 0.034 | 1.03–3.03 |
| wavenet | 10 | 384 | 50.03 | 35.26 | 0.093 | 1.09–3.09 |
| wavenet | 12 | 512 | 85.78 | 71.01 | 0.160 | 1.16–3.16 |
| wavenet | 16 | 896 | 282.91 | 268.14 | 0.527 | 1.53–3.53 |
| wavenet | 20 | 1472 | 879.87 | 865.10 | 1.639 | 2.64–4.64 |
| wavenet | 20 | 2304 | 2093.44 | 2078.66 | 3.899 | 4.90–6.90 |
| lynxnet | 3 | 128 | 16.11 | 1.34 | 0.030 | 1.03–3.03 |
| lynxnet | 6 | 384 | 31.74 | 16.97 | 0.059 | 1.06–3.06 |
| lynxnet | 8 | 512 | 52.23 | 37.46 | 0.097 | 1.10–3.10 |
| lynxnet | 12 | 768 | 131.98 | 117.21 | 0.246 | 1.25–3.25 |
| lynxnet | 16 | 2048 | 1052.07 | 1037.30 | 1.960 | 2.96–4.96 |
| lynxnet | 20 | 2560 | 2002.38 | 1987.60 | 3.730 | 4.73–6.73 |
| lynxnet2 | 3 | 256 | 18.12 | 3.35 | 0.034 | 1.03–3.03 |
| lynxnet2 | 6 | 512 | 35.48 | 20.70 | 0.066 | 1.07–3.07 |
| lynxnet2 | 8 | 768 | 72.69 | 57.92 | 0.135 | 1.14–3.14 |
| lynxnet2 | 12 | 1024 | 159.42 | 144.65 | 0.297 | 1.30–3.30 |
| lynxnet2 | 16 | 2560 | 1174.11 | 1159.34 | 2.187 | 3.19–5.19 |
| lynxnet2 | 20 | 3072 | 2061.25 | 2046.47 | 3.839 | 4.84–6.84 |
| dit | 2 | 128 | 16.26 | 1.49 | 0.030 | 1.03–3.03 |
| dit | 4 | 384 | 37.54 | 22.77 | 0.070 | 1.07–3.07 |
| dit | 6 | 640 | 106.82 | 92.05 | 0.199 | 1.20–3.20 |
| dit | 8 | 1024 | 324.90 | 310.13 | 0.605 | 1.61–3.61 |
| dit | 12 | 1536 | 1051.10 | 1036.33 | 1.958 | 2.96–4.96 |
| dit | 13 | 2048 | 2006.87 | 1992.10 | 3.738 | 4.74–6.74 |

## 使用表中配置

在完整实验 YAML 中设置下面的字段，并将 `L`、`D` 换成对应行中的整数。这些片段只描述 backbone，仍需沿用 acoustic/variance/all-in-one 模板中的数据、编码器、扩散类型和训练参数。它们不是预训练模型。

```yaml
# WaveNet
backbone_type: wavenet
backbone_args:
  num_layers: L
  num_channels: D
  dilation_cycle_length: 4
```

```yaml
# LYNXNet
backbone_type: lynxnet
backbone_args:
  num_layers: L
  num_channels: D
  expansion_factor: 2
  kernel_size: 31
```

```yaml
# LYNXNet2：本项目参数名为 dropout_rate
backbone_type: lynxnet2
backbone_args:
  num_layers: L
  num_channels: D
  expansion_factor: 1
  kernel_size: 31
  dropout_rate: 0.0
  glu_type: softsign_glu
  use_conditioner_cache: false
```

```yaml
# DiT：num_heads = D / 64，例如 D=384 时填 6
backbone_type: dit
backbone_args:
  num_layers: L
  num_channels: D
  num_heads: D_DIV_64
  mlp_ratio: 4
  time_embed_dim: 256
  patch_size: 1
  rope_base: 10000.0
  layer_norm_eps: 0.000001
  attention_dropout: 0.0
  mlp_dropout: 0.0
  use_gradient_checkpointing: true
```

LYNXNet2 的激活选择不改变权重数；`use_fused_kernels` 沿用本项目的 SoftSignGLU 支持条件。Muon/AdamW 专用参数标记不会增加参数个数。DiT 的 head 数在固定宽度下也不改变参数数，但影响实际 kernel 选择、attention 张量与效率。双时间步会额外保存逐帧时间调制张量，其训练显存不能按单时间步预算估计。

## 长度、精度与 all-in-one

卷积 backbone 的主要激活随 `batch × 帧数 × 宽度` 增长，WaveNet 还会保存各层 skip 输出。DiT 的普通 attention 矩阵随 `batch × heads × 帧数²` 增长：以 FP32、32 heads 为例，一份 768 帧矩阵约 72 MiB；一份 4096 帧矩阵约 2 GiB，score、softmax 和 workspace 可能同时存在多份。内存高效 SDPA 减少矩阵物化，计算量仍随长度明显增长。768 帧表格不能作为整首歌曲不切段推理的预算。

顺序扩散/流采样不需要保存全部步骤的激活，步数主要影响耗时；高阶求解器、缓存、CFG 或并发请求仍可能增加峰值。逐块 gradient checkpointing 有助于训练显存，对 `no_grad()` 推理没有同等节省。

all-in-one 训练同时持有 acoustic、pitch 和 multi-variance 生成器、两个编码器、可选辅助 decoder 与 duration/stretch 模块。总参数量按所有已启用模块相加：

`P_joint = P_acoustic_backbone + P_pitch_backbone + P_variance_backbone + P_encoders + P_duration + P_stretch + P_aux + P_embeddings`

这三个 backbone 不共享权重。若三个分支都使用约 2B 配置，仅 FP16/BF16 生成器权重就约 11.2 GiB，还没计训练状态。原生推理若只加载一个分支，则只计算对应模块；variance 推理通常还要同时持有 pitch 与 multi-variance 两个 backbone。声码器也应单独计入。上表的“单 backbone 推理估算”不能称作 joint 总显存。

pitch 使用 `repeat_bins`，multi-variance 使用 `total_repeat_bins / 已启用曲线数` 与相应 `num_feats`，因此输入/输出投影参数会与 128-bin 声学表有所不同。切换分支宽度时继续独立设置嵌套 `backbone_args`，不要误改条件编码器的 `hidden_size`。

FP32 理论权重是上表权重列的两倍。开启 autocast 时，若模型仍以 FP32 保存权重，应按 4 字节参数计入；额外的 autocast 权重缓存还可能提高峰值。INT8/INT4 的权重理论下限分别约为 `P×1` / `P×0.5` 字节，但比例不能直接套用到整体显存，量化尺度、固定 buffer、工作区和后端支持都要另外核实。当前提供 INT8 推理转换与导出，不提供 INT4 或量化训练；扩大覆盖后的完整模型统计见 [Checkpoint Optimization](CheckpointOptimization.md)。

训练还需要梯度、优化器状态、反向激活与重计算 workspace。不能从“2B 推理可能容纳”推导“同一显卡可以全参训练 2B”；尤其 Muon 与 AdamW 的状态布局不同，需实测。

## 重新计算与实测

```powershell
conda activate diffs
python scripts/model_scaling.py
python scripts/model_scaling.py --json
# 独立完整模型，分别计算 acoustic 与 variance
python scripts/model_scaling.py --scope acoustic
python scripts/model_scaling.py --scope variance
python scripts/model_scaling.py --scope variance --json
# 例如 pitch 的 64-bin 输入/输出
python scripts/model_scaling.py --out-dims 64 --hidden-size 384
# 例如四条曲线、总共 72 repeat bins
python scripts/model_scaling.py --out-dims 18 --num-feats 4 --hidden-size 384
```

实测时用最终 checkpoint、设备、精度、完整推理 pipeline 与代表性最长片段，先预热，再分别记录 `torch.cuda.max_memory_allocated()`、`max_memory_reserved()` 和设备总占用；CUDA 同步后读取峰值。比较不同模型时在独立进程中测量，注明帧长、batch、采样器、步数、SDPA/ONNX 后端，以及是否同时驻留 variance 和声码器。参数表可复现，显存范围需要以该实测结果校正。
