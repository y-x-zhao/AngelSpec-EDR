# Examples

Training recipes for the Expected Decoding Rounds (EDR) paper experiments:
second-stage finetuning of released drafters with the EDR objective and the
E2E multi-step TV baseline. Each recipe runs on a single GPU (the paper used
one NVIDIA H200; one B200 or RTX PRO 6000 also works) with a global batch of 96.

| Example | Drafter | Target | Objective | Config |
|---------|---------|--------|-----------|--------|
| [qwen3-4b-dspark-edr](qwen3-4b-dspark-edr/) | DSpark | Qwen3-4B | EDR | `configs/vllm_qwen3_4b_dspark_edr.yaml` |
| [qwen3-4b-dspark-e2e](qwen3-4b-dspark-e2e/) | DSpark | Qwen3-4B | E2E | `configs/vllm_qwen3_4b_dspark_e2e.yaml` |
| [qwen3-8b-dfly-cpt-edr](qwen3-8b-dfly-cpt-edr/) | DFly Block8 | Qwen3-8B | EDR | `configs/vllm_qwen3_8b_dfly_edr.yaml` |
| [qwen3-8b-dfly-cpt-e2e](qwen3-8b-dfly-cpt-e2e/) | DFly Block8 | Qwen3-8B | E2E | `configs/vllm_qwen3_8b_dfly_e2e.yaml` |

Sample the training data first with [generate_training_data](generate_training_data/)
from the same training config (so it uses exactly the config's
`dataset.target_sampling`), then train. The 4B EDR
README also covers the DSpark checkpoint import and hardware presets shared by
both DSpark recipes. Offline MAL evaluation is in [eval](eval/).
