# 优化 checkpoint 的 ONNX 与 LibTorch 导出

acoustic、variance 和 all-in-one 的原始或优化 checkpoint 共用部署模型构造器，支持 WaveNet、LYNXNet、LYNXNet2、DiT。优化 checkpoint 先在 CPU 严格加载，再移到导出设备；不会先在 GPU 恢复完整浮点权重。

## ONNX

先按 [量化文档](CheckpointOptimization.md) 生成推理 checkpoint。输出放在 `checkpoints/my_model_int8` 时，使用现有入口：

```bash
python scripts/export.py acoustic --exp my_model_int8 --out artifacts/my_model_int8/acoustic
python scripts/export.py variance --exp my_model_int8 --out artifacts/my_model_int8/variance
```

all-in-one 分别执行这两个命令，导出器只读取需要的分支。两套附件与配置放在独立目录。原有 speaker 混合、gender/velocity、glide/expr 导出选项仍由该入口处理。

INT8 矩阵投影映射为标准 ONNX `MatMulInteger`：激活从有符号字节平移为 UINT8，并声明零点128；权重保持 INT8，结果为 INT32。激活量化、输出尺度、attention、非线性和采样继续使用浮点。深度卷积使用整数乘法/累加；Embedding 只转换查出的行。不是先恢复整套浮点权重再导出。算子定义见 [ONNX MatMulInteger](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html)。

导出优化会折叠小向量的尺度计算，因此 ONNX/TorchScript 的小归一化系数可能成为浮点常量；导出时构建的固定 pitch 平滑滤波器也保留浮点。checkpoint 中的100%可学习参数 INT8存储覆盖率，不等于导出文件中每个常量都为 INT8，也不等于100%算子为整数。大型学习矩阵仍保留整数计算。

本机 ONNX Runtime **1.23.0 CPU** 在部分 FP32 图上开启图优化时出现显著数值偏差；关闭图优化后对照恢复到约1e-6量级。FP32 验证因此使用以下设置，部署时也应按此建立基线，逐项恢复优化并对照；INT8 的完整检查已在默认优化下通过：

```python
import onnxruntime as ort
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession('model.onnx', options, providers=['CPUExecutionProvider'])
```

动态整数卷积、GEMM、GRU 保留循环和序列操作；优化 checkpoint 的图跳过 ONNX Simplifier，保留图检查与阶段合并，避免反复常量折叠或损坏循环捕获的父图引用。目标运行时需要支持这些标准算子；已验证的 ONNX Runtime CPU 执行路径不能推导其他引擎或 GPU provider 也能加速。TensorRT、移动端及其他后端需另行验证，不能仅改扩展名。FP16 导出保留半精度矩阵计算模块及必要的浮点接口；归一化使用 FP32 累加，避免低精度统计退化。FP16 ONNX 转换不预先冻结模型，TorchScript 保存则冻结并移除未使用模块。

## TorchScript：用于 LibTorch

```bash
python scripts/export_torchscript.py \
  --checkpoint checkpoints/my_model_int8/model_ckpt_steps_100000.ckpt \
  --output-dir artifacts/my_model_libtorch
```

`--config` 默认 checkpoint 同目录的 `config.yaml`；输出目录必须不存在。旧训练 checkpoint 没有 category 时，指定 `--component acoustic`、`variance` 或 `all_in_one`。默认 `--device cpu`；CPU 推荐 FP32/INT8。FP16 使用 `--device cuda` 导出并在目标 GPU 验证。普通阶段 tracing 会记录部分设备选择，部署应匹配 manifest 的 `target_device`；切换 CPU/CUDA 时重新按目标设备导出，不把 `map_location` 当作跨设备正确性保证。

不同配置/精度的部署检查按 CLI 工作流使用独立进程，避免复用全局 JIT/ONNX 状态；批量分发时也分别启动导出命令。

每个分支输出独立 `.pt` 阶段、`<component>.torchscript.json`、字典、phoneme/language 表、speaker embedding（配置需要时）和 `metadata.yaml`。不会生成引用缺失 ONNX 文件的 `dsconfig.yaml`。LibTorch 应以 manifest 的 `forward_schema`、`input_names`、`arguments` 和 `outputs` 接线；不同配置的输入数量会改变。导出器保存后立即重新加载，并对同种子结果做数值对照。

| 分支 | 阶段与调用关系 |
|---|---|
| acoustic | `fs2_aux`（未启用浅扩散时为 `fs2`）输出 condition/aux_mel → `diffusion` 输出 mel。后者按模型配置运行 DDPM 或 Reflow。 |
| variance | `linguistic` 输出编码与 mask → 可选 `dur`；`pitch_pre` → `pitch` → `pitch_post`；`variance_pre` → `variance` → `variance_post`。只导出配置启用的预测项。 |

普通阶段将字典中的 Tensor 输入展平为位置参数，顺序由 manifest 列出；sampler 保留 TorchScript 的运行时整型 `steps` 与所需 depth Tensor。调用者负责字典编码、音符/时长/retake 输入、speaker 混合、阶段数据流和声码器。导出的分阶段模块没有 `.ds` 解析器，也不自动合成最终波形。

Python 加载示例：

```python
import torch
module = torch.jit.load('artifacts/my_model_libtorch/acoustic/acoustic.diffusion.pt')
# 浅扩散 sampler：condition/aux_mel 来自 FS2+辅助 decoder 阶段。
mel = module(condition, aux_mel, torch.tensor(0.5), 10)
```

无浅扩散时按 manifest 的 schema 调用，不能套用上述四参数示例。Tensor 放在与模块一致的设备上；输入类型按 manifest，浮点接口通常为 FP32，tokens/durations 为 INT64，mask/retake 为 BOOL。

提供 [LibTorch sampler 示例](../deployment/libtorch_example/sampler.cpp) 与 CMake：

```bash
cmake -S deployment/libtorch_example -B build/libtorch -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build/libtorch --config Release
# 浅扩散模型的随机条件 smoke 示例，不是歌曲推理：
diffsinger_sampler acoustic.diffusion.pt 384 128 32
```

这里的 C++ 示例需在自己的编译环境验证；本机没有完成 C++ 编译测试。`.pt` 的保存、独立 `torch.jit.load` 与执行对照由 Python 验证。LibTorch 应匹配导出 PyTorch 版本及 CPU/CUDA 构建；INT8 依赖内置但私有的 `aten::_int_mm`，升级或换设备后需复验。

## 复现检查

```bash
python scripts/validate_checkpoint_optimization.py
python scripts/validate_deployment_exports.py --output export_validation.json
# 可额外检查浮点精度；FP16 要有 CUDA：
python scripts/validate_deployment_exports.py --backbones dit --precisions fp32 fp16
```

部署检查使用随机小模型，覆盖严格加载、TorchScript 保存/重载对照、ONNX 阶段运行、完整图合并与运行，以及动态长度和采样步数。它不代表训练权重音质已验收，也不代表大型模型在目标显卡上的峰值已实测。原 checkpoint、真实歌曲对照与显存预算见 [量化文档](CheckpointOptimization.md)。

INT8 已完成四种 backbone × DDPM/Reflow × acoustic/variance 的16组完整导出执行检查，结果见 [INT8 检查记录](deployment_export_validation.json)。FP32/FP16 的附加检查使用 DiT；FP32 运行 ONNX CPU，FP16 检查 ONNX 结构和 TorchScript CUDA 重载执行。**当前环境没有 ONNX Runtime CUDA provider，FP16 ONNX 的 GPU 执行尚未验证；不推荐通过 CPU 执行 FP16 来替代这项验证。** 浮点检查记录中的 `onnx_execution` 标明是否实际运行 ONNX。
