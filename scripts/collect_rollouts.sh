#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Step 3: Collect student rollouts on OPD prompts.
#
# Uses data_curation/ to run the SFT model on OPD prompts (e.g. DAPO-Math-17k)
# and collect response rollouts for Lightning OPD data preparation.
#
# Required environment variables:
#   SFT_CHECKPOINT - Path to the SFT model checkpoint
#   OPD_PROMPTS    - Path to the OPD prompt dataset (.jsonl or .parquet)
#   OUTPUT_DIR     - Directory for collected rollout data
#
# Optional:
#   NUM_GPUS       - Number of GPUs to use (default: 8)
#   TP_SIZE        - Tensor parallel size per worker (default: 1)
#   VLLM_VENV      - vLLM virtualenv used for `source .../activate` before `vllm serve`
#                    (default: /data/zhangdw12/work/uv-venv/qwen35-vllm019)
#   VLLM_USE_SERVER - Use vLLM OpenAI-compatible server mode (default: 1 for rollouts)
#   VLLM_BASE_PORT - First vLLM server port; rank is added (default: 18000)
#
# Extra args are passed through to data_curation/pipeline.py, e.g.:
#   bash scripts/collect_rollouts.sh --num-samples 10

set -euo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the SFT model path}"
: "${OPD_PROMPTS:?Set OPD_PROMPTS to the OPD prompt dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for collected rollout data}"

# Resolve to absolute paths (workers may run from different cwd)
SFT_CHECKPOINT="$(cd "$(dirname "${SFT_CHECKPOINT}")" && pwd)/$(basename "${SFT_CHECKPOINT}")"
OPD_PROMPTS="$(cd "$(dirname "${OPD_PROMPTS}")" && pwd)/$(basename "${OPD_PROMPTS}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
# Rollout collection uses served vLLM by default so the SFT environment does not
# import vLLM directly. Override VLLM_USE_SERVER=0 only for the legacy offline path.
VLLM_USE_SERVER="${VLLM_USE_SERVER:-1}"
VLLM_VENV="${VLLM_VENV:-/data/zhangdw12/work/uv-venv/qwen35-vllm019}"

VLLM_USE_SERVER="${VLLM_USE_SERVER}" \
VLLM_VENV="${VLLM_VENV}" \
bash data_curation/run_curation.sh \
    --model "${SFT_CHECKPOINT}" \
    --input "${OPD_PROMPTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-gpus "${NUM_GPUS}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "$@"
