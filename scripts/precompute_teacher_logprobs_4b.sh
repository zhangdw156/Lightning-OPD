#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Precompute teacher logprobs for Lightning OPD (4B scale, teacher=Qwen3-8B).
#
# Required environment variables:
#   SFT_CHECKPOINT  - Path to the SFT checkpoint (used as tokenizer)
#   ROLLOUT_PARQUET - Path to the student rollout parquet file
#   OUTPUT_DIR      - Directory for the output parquet with teacher logprobs
#
# This script starts a Qwen3-8B teacher server, then runs Phase 1+2 of
# prepare_lightning_opd.py to tokenize and precompute teacher logprobs.

set -euo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the SFT model path}"
: "${ROLLOUT_PARQUET:?Set ROLLOUT_PARQUET to the student rollout parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the output parquet}"

TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_URL="${TEACHER_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/generate}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
CONCURRENCY="${CONCURRENCY:-64}"

# Start teacher server. For 4x H20, use TEACHER_TP=4.
bash scripts/serve_teacher_8b.sh

python3 data_curation/prepare_lightning_opd.py \
    --tokenizer-path "${SFT_CHECKPOINT}" \
    --input-parquet "${ROLLOUT_PARQUET}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-response-len "${MAX_RESPONSE_LEN}" \
    --compute-teacher-logprobs \
    --teacher-url "${TEACHER_URL}" \
    --concurrency "${CONCURRENCY}"
