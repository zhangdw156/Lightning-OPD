#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data/zhangdw12/models}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-8B}"
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_TP="${TEACHER_TP:-8}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.6}"
TEACHER_CONTEXT_LENGTH="${TEACHER_CONTEXT_LENGTH:-8192}"
TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"

LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP}" \
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
    --context-length "${TEACHER_CONTEXT_LENGTH}" \
    > "${LOG_FILE}" 2>&1 &

until curl -sf "http://${TEACHER_HOST}:${TEACHER_PORT}/health_generate" > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "${LOG_FILE}"
    sleep 5
done

echo "Teacher model ready at http://${TEACHER_HOST}:${TEACHER_PORT}/generate (log: ${LOG_FILE})"
