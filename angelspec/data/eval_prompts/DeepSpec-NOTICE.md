# DeepSpec evaluation prompt provenance

The nine JSONLs are adapted from the evaluation data distributed in
https://github.com/deepseek-ai/DeepSpec at revision
`005e03b81cec38b7da6399833d609ee89a2587f2`.
Only the default evaluated subset and the first user turn are retained;
`turns[0]` is renamed to `prompt`. See `deepspec_manifest.json` for file URLs
and hashes.

DeepSpec's converter identifies these underlying datasets:

- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- HumanEval: https://huggingface.co/datasets/openai/openai_humaneval
- MBPP (sanitized): https://huggingface.co/datasets/google-research-datasets/mbpp
- MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- AIME25: https://huggingface.co/datasets/MathArena/aime_2025
- MT-Bench: https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts
- LiveCodeBench: https://huggingface.co/datasets/livecodebench/code_generation_lite
- Alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca
- Arena-Hard v2: https://huggingface.co/datasets/lmarena-ai/arena-hard-auto

DeepSpec's default prompt counts are GSM8K 500, MATH-500 500, AIME25 30,
HumanEval 164, MBPP 256, LiveCodeBench 500, MT-Bench 80, Alpaca 500 and
Arena-Hard v2 500. Subsets use the upstream evaluation seed `980406`.

Underlying dataset terms continue to apply. The following is DeepSpec's
repository license, not a replacement for individual dataset licenses.

## DeepSpec license

MIT License

Copyright (c) 2026 The DeepSpec Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
