# LoRA fine-tuning and export

LoRA is optional and disabled by default. It adds low-rank adapters to selected `Linear` layers, loads a compatible base checkpoint, and freezes the base model. The implementation follows `dsrx_ref`'s adapter orientation: `A=[in, rank]`, `B=[rank, out]`, with scaling `alpha/rank`.

Add this to your experiment YAML:

```yaml
lora:
  enabled: true
  base_ckpt: ckpt/base_model/model_ckpt_steps_100000.ckpt
  rank: 8
  alpha: 16
  target_modules: [linear]
  train_bias: false
```

```bash
python scripts/train.py --config configs/my_lora.yaml --exp_name my_lora --reset
```

`target_modules` accepts regular expressions matched against full module names. `linear` or `*` selects all Linear layers. Inspect the model's module names before restricting targets; unmatched patterns raise an error. This covers shared FS2/auxiliary decoder linears as well as eligible backbone linears. Convolution and embedding layers are not adapted. WaveNet, LYNXNet, LYNXNet2, and DiT can all use LoRA, but the adapted layers depend on their structure; a convolution-only target has no eligible Linear layers.

Acoustic, variance, and all-in-one tasks share the same LoRA setup, for both DDPM and Rectified Flow. With all-in-one, names have `acoustic.` or `variance.` prefixes. A compatible all-in-one base checkpoint can also supply one standalone branch. The base configuration must match the model's architecture, vocabulary, speaker embeddings, and tensor shapes. Base loading is strict; incompatible weights fail rather than leaving randomly initialized frozen parameters.

`train_bias: true` also trains model biases. Leave `finetune_enabled` and `freezing_enabled` disabled: LoRA handles base loading and freezing. With the default Muon/AdamW optimizer, adapter matrices use AdamW. Backbones containing adapters skip fused Linear kernels because those kernels read base weights directly and would bypass adapter gradients. Other backbones can still use `use_fused_kernels`.

Training checkpoints contain the complete base model, adapters, optimizer state, and LoRA metadata. Resume using the same `--exp_name` without `--reset`; the saved checkpoint supplies the base weights, so the original base checkpoint is not required on resume. Adapter module names, rank, and alpha must remain unchanged. Optimizer and scheduler state are retained. Metadata mismatch or an incomplete adapter fails explicitly.

## Inference and deployment

The shared checkpoint loader merges LoRA into ordinary weights before loading a plain inference/export model. Merge uses the saved per-module rank and alpha. It also preserves trained biases. Original checkpoints are not modified. All-in-one branch extraction remains supported.

LoRA exports retain checker-validated ONNX graphs rather than running ONNX simplifier, which can break captured values inside sampler Loop graphs. ONNX Runtime execution and TorchScript save/reload are covered by `scripts/validate_deployment_exports.py --lora`.

Existing ONNX export commands work directly with the LoRA experiment:

```bash
python scripts/export.py acoustic --exp my_lora --ckpt 10000
python scripts/export.py variance --exp my_lora --ckpt 10000
```

TorchScript export also merges the adapters:

```bash
python scripts/export_torchscript.py --checkpoint ckpt/my_lora/model_ckpt_steps_10000.ckpt --config ckpt/my_lora/config.yaml --output-dir artifacts/my_lora_ts --component auto --device cpu
```

To save a merged standalone FP32 checkpoint, or merge and then quantize, use a new output directory:

```bash
python scripts/optimize_checkpoint.py --checkpoint ckpt/my_lora/model_ckpt_steps_10000.ckpt --config ckpt/my_lora/config.yaml --precision fp32 --output-dir ckpt/my_lora_merged
python scripts/optimize_checkpoint.py --checkpoint ckpt/my_lora/model_ckpt_steps_10000.ckpt --config ckpt/my_lora/config.yaml --precision int8 --output-dir ckpt/my_lora_int8
```

These output configurations disable LoRA because adapters have already been merged. Merged/optimized artifacts are inference artifacts; resume fine-tuning from the original training checkpoint. See [deployment exports](DeploymentExports.md) and [checkpoint optimization](CheckpointOptimization.md) for device and precision constraints.

Reference-project full checkpoints without LoRA metadata can be merged using the original saved `lora.alpha`. Adapter-only checkpoints without base weights are not accepted by the shared full-checkpoint loader.
