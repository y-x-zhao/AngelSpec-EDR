# Quickstart

This walks through EDR finetuning of the released **DFly** draft model for **Qwen3-8B**
(`AngelSlim/Qwen3-8B-DFly-Block8`), one of the paper's experiments.

## Prerequisites

- One NVIDIA H200 (as in the paper), B200 or RTX PRO 6000 GPU (single-process HF target + draft)
- Local copies of `Qwen/Qwen3-8B` and `AngelSlim/Qwen3-8B-DFly-Block8`, plus the prepared
  training cache — see [`examples/qwen3-8b-dfly-cpt-edr`](../../examples/qwen3-8b-dfly-cpt-edr/)
- AngelSpec installed ([Installation](installation.md))

## Run it

```bash
./examples/qwen3-8b-dfly-cpt-edr/run.sh
```

This launches `angelspec.train_single_gpu` with `configs/vllm_qwen3_8b_dfly_edr.yaml` on one
GPU (no Ray, Mooncake or vLLM). The global batch is 96 (micro batch 1 x 96 accumulation steps).

## Common overrides

Config values can be overridden directly on the command line:

```bash
# Shorter run
./examples/qwen3-8b-dfly-cpt-edr/run.sh training.num_train_steps=50

# Different learning rate
./examples/qwen3-8b-dfly-cpt-edr/run.sh training.learning_rate=2e-5

# Select one GPU on a multi-GPU host (the launcher requires exactly one visible GPU)
CUDA_VISIBLE_DEVICES=0 ./examples/qwen3-8b-dfly-cpt-edr/run.sh
```

## What happens under the hood

A frozen Hugging Face copy of the target model prefills cached, target-regenerated training
conversations and extracts hidden states from selected layers on the same GPU; the DFly draft
model then runs its forward/backward on those tensors directly. The disaggregated multi-GPU
pipeline is described in [Disaggregated Architecture](../concepts/disaggregated_architecture.md)
but is not used by the paper recipes.

## Other recipes

The other paper recipes are the E2E baseline for the same drafter and the Qwen3-4B DSpark pair:

```bash
./examples/qwen3-8b-dfly-cpt-e2e/run.sh
./examples/qwen3-4b-dspark-edr/run.sh
./examples/qwen3-4b-dspark-e2e/run.sh
```

See [The Draft-Model Family](../concepts/draft_model_family.md) for architecture details.

## Next steps

- Evaluate MAL offline: [`examples/eval`](../../examples/eval/)
- Scale to multi-node: [Multi-Node Training](../advanced_features/multi_node.md)
- Convert a checkpoint for serving: [Checkpoint Conversion](../basic_usage/checkpoint_conversion.md)
