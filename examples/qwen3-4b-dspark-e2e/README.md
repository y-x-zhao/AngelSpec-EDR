# Qwen3-4B DSpark — E2E TV

Uses `configs/vllm_qwen3_4b_dspark_e2e.yaml`: pure E2E TV, global batch 96,
and the shared non-thinking Qwen3-4B responses sampled with
`T=0.7, top_p=0.8, top_k=20, min_p=0`. Distribution-aware loss is enabled:
TV uses the filtered target from `dataset.target_sampling` and the unfiltered
draft at the same positive temperature. Target filtering follows vLLM's PyTorch
sampling path. `training.dflash_distill_distribution_aware=false` uses the
full-vocabulary target/draft `T=1` loss instead; the response data and cache
selection are the same either way.

Checkpoint import, data generation, resume instructions and hardware presets
are shared with the EDR recipe; see [qwen3-4b-dspark-edr](../qwen3-4b-dspark-edr/README.md).

```bash
export DRAFT_CKPT=/absolute/path/to/your/qwen3-4b-dspark-initialization
CUDA_VISIBLE_DEVICES=0 bash examples/qwen3-4b-dspark-e2e/run.sh
```
