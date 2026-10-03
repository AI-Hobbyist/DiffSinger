# DiT backbone 与 all-in-one 训练

从本地参考项目 `dsrx_ref` 移植 DiT 和 all-in-one，同时沿用本项目的条件编码器、stretch embedding、variance scaling、AdamW 专用投影层、双时间步 Reflow、Muon 优化器与 LYNXNet2 fused kernel 接口。模型代码不会从 `dsrx_ref` 导入；该目录只作为移植参考。

## DiT

声学模型使用根级 `backbone_type: dit` / `backbone_args`；pitch 和其他 variance 分别使用 `pitch_prediction_args` / `variances_prediction_args` 内的同名字段。可以混用不同 backbone，时长预测仍使用现有 duration predictor。

DiT 是逐帧双向 attention，包含 RoPE、Q/K normalization、adaLN-Zero、受限时间调制与零初始化输出头。输入和输出保持 `[B,F,M,T]`，条件为 `[B,H,T]`。`H` 是本项目的 `hidden_size`，不需要等于 DiT 的 `num_channels`。训练和所有原生采样器从 `mel2ph > 0` 传递布尔 `valid_mask`，True 表示有效帧；padding key 被屏蔽，padding 输出清零，每条样本必须至少包含一个有效帧。

本项目的 `use_dual_timestep: true` 也支持 DiT：先分别编码两个时间步，再按照原有逐帧选择 mask 混合时间 embedding，供各块调制使用。这个时间选择 mask 与 padding 的 `valid_mask` 是两个独立参数。DDPM 保持单时间步噪声预测。默认模板先关闭双时间步，可按实验需要开启。

可直接修改以下模板的数据路径、字典、说话人映射和训练预算：

- [声学 DiT](../configs/templates/config_acoustic_dit.yaml)
- [variance DiT](../configs/templates/config_variance_dit.yaml)
- [all-in-one DiT](../configs/templates/all_in_one_dit.yaml)
- [all-in-one LYNXNet2 + Muon](../configs/templates/all_in_one_lynxnet2_muon.yaml)
- [all-in-one WaveNet + AdamW](../configs/templates/all_in_one_wavenet_adamw.yaml)

切换 backbone 类型时 `backbone_args` 作为整个映射替换，不会继承上个 backbone 的卷积参数；切换优化器或调度器类时，对应的 `optimizer_args` / `lr_scheduler_args` 也整组替换。相同类型的局部覆盖沿用本项目原有递归继承行为，嵌套 pitch/variance 的其他字段仍保留。DiT 参数名严格校验，拼错或未知字段会报错。需要自定义参数时写完整的对应映射。

DiT 使用 PyTorch SDPA，沿用当前项目 PyTorch >= 2.4 的环境要求。`use_gradient_checkpointing` 只影响训练；`max_sample_frames` 可在加载二进制数据时拒绝过长的样本。长度限制不会截断对齐标签，需先在数据准备阶段切分样本。

## All-in-one

all-in-one 是所有 backbone 通用的选项：WaveNet、LYNXNet、LYNXNet2、DiT 均可使用，acoustic、pitch、曲线三个生成器也可以各选不同类型。通用基础配置为 [configs/all_in_one.yaml](../configs/all_in_one.yaml)，它不锁定任何 backbone。

在已有 acoustic/variance 实验配置中加入以下开关即可自动选择 joint task 和 binarizer，并补齐另一分支缺少的默认配置。请检查两分支预测开关、数据标注和预算；若不指定另一分支参数，它使用默认 backbone 配置。也可以直接继承通用基础配置后分别指定三个生成器。

```yaml
all_in_one:
  enabled: true
```

该模式将 acoustic 和 variance 两个现有模型装入一个 `DiffSingerAllInOne` 容器，使用同一个优化器和一个 checkpoint。它们保留各自条件编码器、参数和损失，没有共享或合并两个 backbone。variance 各预测开关按配置生效，至少需要启用一个 variance 分支（dur、pitch 或曲线）；不会强制开启不需要的预测器。

```powershell
conda activate diffs
python scripts/binarize.py --config configs/templates/all_in_one_dit.yaml
python scripts/train.py --config configs/templates/all_in_one_dit.yaml --exp_name joint_dit --reset
```

请先编辑模板；示例中的数据与验证样本路径需要对应自己的语料。原始语料必须同时具备 acoustic 与已启用 variance 预测器所需的标注。二值化输出为 `binary_data_dir/acoustic` 和 `binary_data_dir/variance`，使用同一语料独立生成两种表示，继续使用本项目现有的多扩展名音频定位与标签检查。

每个训练步取两个数据流的 batch，损失相加，较短的数据流以 `max_size_cycle` 循环。验证分别计算两个数据流的损失和指标，正确切换各自元数据。`max_batch_frames`、`max_batch_size` 是两个训练流的总预算，每个流使用一半（向下取整），因此 `max_batch_size` 至少为 2；验证限额则对每个流分别应用。循环和两个模型共同驻留会影响吞吐量、语料采样权重和显存，详见 [Model scaling](ModelScaling.md)。

LYNXNet2 的 fused kernel 选择和预热会覆盖 acoustic、pitch 与 variance 三个生成器；仍仅适用于现有 kernel 支持的 backbone/激活。没有 Triton 的环境沿用原有 eager 回退。Muon 继续按本项目参数标记区分矩阵与 AdamW 参数；DiT 的常规 Linear 矩阵可参与 Muon。

训练 checkpoint 的 category 为 `all_in_one`，权重分别在 `model.acoustic.*` / `model.variance.*` 下。现有原生 acoustic/variance 推理和两类 ONNX exporter 会自动严格加载对应分支。断点续训使用整个 joint checkpoint，保留两个模型与优化器状态，不能用一个单分支 checkpoint 假装恢复 joint 训练。

ONNX 导出沿用当前项目的 opset 17 和 TorchScript 路线。DiT 导出 adapter 将 SDPA 展开为普通 attention 算子，并保留动态帧长。导出模型按现有单样本无 padding 协议运行；训练的 padding mask 和双时间步增强不增加宿主输入字段。旧 backbone 权重不能直接转换成 DiT；已有编码器的微调加载仍需遵守现有 shape/key 检查。

所有 backbone 及混合 backbone 的 all-in-one 模式均支持 `val_with_variance`。可选 `val_with_variance.enable: true` 会在验证时从配置的语言 `.ds` 片段预测时长、音高和曲线，再合成声学预览。需要 dur 与 pitch 开启，并设置可用声码器；该功能默认关闭。语言映射沿用二值化的排序/编号；使用当前模型内存中的分支，不重复加载训练权重。训练初期时长全为零时会提示并跳过该次预览。声学启用的曲线条件需要对应预测器已开启，或验证源内提供该曲线。

```yaml
val_with_variance:
  enable: true
  zh:
    - samples/02_不老梦_variance.ds
```

## 辅助数据集

`aux_datasets` 从参考项目移植，支持独立 variance 和 all-in-one，适用于所有 backbone。`module` 指定由辅助语料监督的预测模块：`dur`、`pitch`、`energy`、`breathiness`、`voicing`、`tension`。指定模块必须已启用。主 variance 数据监督其余模块，辅助数据只计算选定模块的损失；曲线生成器通过 feature mask 隔离对应通道，保持当前项目原有 padding 损失约定。

```yaml
aux_datasets:
  enable: true
  module: [pitch, voicing]
  spk_ids: [0]   # 与 data 列表逐项对应，必须位于 num_spk 范围内
  datasets:
    data:
      - zh: data/aux_singer
    val:
      zh: [validation_prefix]
```

`data` 也接受语言到路径的映射；同语言多套数据使用上述列表形式。`val` 按语言设置验证前缀，至少配置一个有效验证前缀；说话人名称来自目录名。开启 speaker embedding 时，明确填写 `spk_ids`，检查它与主数据中的身份对应关系；留空时按辅助语料独立自动编号。辅助二值化仍使用已开启预测器的完整数据格式，因此要提供这些预测器需要的标签；`module` 只控制监督损失，不放宽标签格式。

使用相同配置执行 `scripts/binarize.py`，辅助数据写入 `binary_data_dir/aux`。独立 variance 将主/辅助流按 `max_size_cycle` 组合，批次限额按参考项目对每个流分别应用；all-in-one 加入第三个流，总训练预算均分为三份，`max_batch_size` 至少为 3。验证限额按每个流分别应用，辅助流记录选定损失且不输出额外图表。若辅助语料不能覆盖字典全部音素，可显式设置 `binarization_args.allow_missing_phonemes: true`，保留缺失提示；默认仍严格检查覆盖率。

## ep/step 切换

沿用参考项目的配置字段；这两个字段可独立选择单位，适用于独立 acoustic、variance 和 all-in-one，与 backbone 无关：

```yaml
max_updates: 100ep
val_check_interval: 5ep
```

整数沿用原有 optimizer step 含义，也支持 `100000step`、`4000step`。`100ep` 表示总共训练 100 个 epoch，`5ep` 表示每 5 个完整 epoch 验证并保存 checkpoint。epoch 间隔不乘梯度累积次数；step 验证沿用当前项目的换算，将间隔乘 `accumulate_grad_batches` 作为训练 batch 间隔。学习率调度器仍按 optimizer step 更新，永久 checkpoint 的开始/间隔仍按 step 计算，文件名也保持 `model_ckpt_steps_*`，以兼容现有恢复、推理和导出。

```powershell
python scripts/train.py --config configs/templates/all_in_one_dit.yaml --exp_name joint_dit --hparams "max_updates=100ep,val_check_interval=5ep"
```

恢复已有实验时，已有 `config.yaml` 优先于模板，因此用 `--hparams` 临时切换周期单位，或修改保存的配置。两种模式可互相切换恢复，Lightning 保留已完成的 epoch、global_step 和优化器状态；上限代表总训练量，而非本次额外训练量。开启 all-in-one/aux 时，一个 epoch 按最长训练流遍历，其余流循环，并保留现有分布式/梯度累积采样补齐规则。选定训练上限若早于下一次验证，则可能尚未产生新的 checkpoint，请合理设置验证间隔。

## 预处理显存控制

参考项目新增的机制是特征提取后的 CUDA cache 清理和多 GPU worker 设备分配。移植后，mel 先转回 CPU 并释放临时 CUDA tensor，再清理缓存；对齐提取也清理缓存。RMVPE、mel、平滑算子、谐波分解和数据增强遵循同一个预处理设备。二值化各阶段结束时释放全局 pitch/平滑算子，避免 all-in-one 的 acoustic、variance、aux 三阶段重复驻留模型；worker 和结果队列也在阶段结束或异常时关闭。

```yaml
binarization_args:
  device: auto              # auto / cpu / cuda / cuda:0
  num_workers: 1            # 0 = 当前进程串行；CPU 模式保持指定并行数
  # num_workers_per_gpu: 1  # 显式设置后按每张可用 GPU 的 worker 数启动
  clear_cuda_cache: true
```

GPU 并行默认每张卡最多一个 worker，并且总数不超过 `num_workers`；指定 `cuda:0` 时仅使用该卡。显式 `num_workers_per_gpu` 或参考项目兼容选项 `workers_per_gpu: true` 可以增加每卡 worker 数，但同时增加模型副本和峰值显存。单卡小显存优先选择 `num_workers: 0` 或 1；也可设 `device: cpu`，全流程不使用 CUDA 特征模型。`clear_cuda_cache: false` 可关闭反复 cache 清理以评估吞吐量。

这些控制减少并发模型与阶段间驻留，不承诺任意长音频都能装入显存。参考项目没有长音频特征分块；本次保留原始音频与标签对齐，不擅自截断。单条过长时先按标注边界切分，或使用 CPU 预处理。

## 回归验证

```powershell
conda activate diffs
python -m unittest discover -s tests -v
python scripts/validate_dit_p1.py
python scripts/validate_all_in_one.py
python scripts/validate_port.py
python scripts/validate_training_modes.py
python scripts/validate_preprocessing.py
python scripts/model_scaling.py
```

验证包括 DiT padding 隔离、形状/参数校验、重计算梯度、Muon 参数分组；四种 backbone 的 DDPM/Reflow 前向、反向、原生推理与本项目条件功能；双时间步；两个真实二进制数据流的 Lightning 训练及恢复；joint checkpoint 分支严格加载；ONNX 动态长度数值对照与完整 exporter；可用 CUDA 环境的 FP16/BF16 joint 训练。还覆盖四种 backbone × DDPM/Reflow 的独立/joint 辅助流训练与验证、实际 variance→acoustic 预览（声码器用 stub）、ep/step 验证频率和切换单位恢复、CUDA worker affinity 与重复 mel/对齐数值对照。GPU 不可用时 CUDA 部分显式跳过。

这些是合成输入的工程回归。大型 scaling 配置仅在 meta device 上计算参数数量，没有进行 2B 训练、音质评价或全尺寸显存实测。本地 Windows `diffs` 环境没有 Triton，因此 fused kernel 验证覆盖选择/路由，实际 Triton kernel 数值与吞吐量仍需在具备该运行时的设备上测量。
