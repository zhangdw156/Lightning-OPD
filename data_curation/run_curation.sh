#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Launch data curation across multiple GPUs / nodes.
#
# Each GPU runs one independent vLLM worker that processes a disjoint shard
# of the dataset. No torch.distributed communication is needed — each worker
# is a standalone process with its own rank derived from environment variables.
#
# ── Single node, 8 GPUs (tp=1, 8 workers) ────────────────────────────────
#   bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-4B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 8
#
# ── Single node, 2 GPUs (tp=2, 1 worker) ─────────────────────────────────
#   bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-8B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 2 \
#       --tensor-parallel-size 2
#
# ── Multi-node (2 nodes × 8 GPUs, tp=1, 16 workers) ─────────────────────
#   # On node 0:
#   NODE_RANK=0 NUM_NODES=2 bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-4B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 8
#
#   # On node 1:
#   NODE_RANK=1 NUM_NODES=2 bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-4B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 8
#
# Environment variables (optional):
#   NUM_NODES   – total number of nodes (default: 1)
#   NODE_RANK   – rank of this node (default: 0)
#   VLLM_PYTHON      – Python executable from a vLLM-capable environment for offline mode
#   VLLM_USE_SERVER  – if 1, start vLLM OpenAI-compatible servers and query them
#   VLLM_VENV        – vLLM virtualenv to activate before running `vllm serve`
#   VLLM_HOST        – server bind host (default: 127.0.0.1)
#   VLLM_BASE_PORT   – first server port; rank is added to this value (default: 18000)
#   VLLM_SERVER_EXTRA_ARGS – extra arguments appended to `vllm serve`
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VLLM_VENV="/data/zhangdw12/work/uv-venv/qwen35-vllm019"
DEFAULT_VLLM_PYTHON="${DEFAULT_VLLM_VENV}/bin/python"
VLLM_USE_SERVER="${VLLM_USE_SERVER:-0}"
VLLM_VENV="${VLLM_VENV:-${DEFAULT_VLLM_VENV}}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-18000}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-lightning-opd-rollout}"

if [[ -n "${VLLM_PYTHON:-}" ]]; then
    PYTHON_BIN="${VLLM_PYTHON}"
elif [[ -x "${DEFAULT_VLLM_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_VLLM_PYTHON}"
else
    PYTHON_BIN="${PYTHON:-python}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set VLLM_PYTHON to a Python executable in a vLLM-capable environment." >&2
    exit 1
fi

if [[ "${VLLM_USE_SERVER}" == "1" ]]; then
    if [[ ! -f "${VLLM_VENV}/bin/activate" ]]; then
        echo "vLLM virtualenv activation script not found: ${VLLM_VENV}/bin/activate" >&2
        echo "Set VLLM_VENV=/path/to/a/vLLM environment." >&2
        exit 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required for vLLM server health checks." >&2
        exit 1
    fi
fi

# ── Parse --num-gpus and --tensor-parallel-size from args ─────────────────
NUM_GPUS=1
TP=1
MODEL_ARG=""
OUTPUT_ROOT=""
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --tensor-parallel-size)
            TP="$2"; PIPELINE_ARGS+=("--tensor-parallel-size" "$2"); shift 2 ;;
        --model)
            MODEL_ARG="$2"; PIPELINE_ARGS+=("--model" "$2"); shift 2 ;;
        --output-dir)
            OUTPUT_ROOT="$2"; PIPELINE_ARGS+=("--output-dir" "$2"); shift 2 ;;
        *)
            PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

if [[ "${VLLM_USE_SERVER}" == "1" && -z "${MODEL_ARG}" ]]; then
    echo "--model is required when VLLM_USE_SERVER=1." >&2
    exit 1
fi

# ── Compute worker layout ────────────────────────────────────────────────
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORKERS_PER_NODE=$(( NUM_GPUS / TP ))
WORLD_SIZE=$(( WORKERS_PER_NODE * NUM_NODES ))

echo "=== Data Curation Launch ==="
echo "  Nodes:            ${NUM_NODES} (this node: ${NODE_RANK})"
echo "  GPUs per node:    ${NUM_GPUS}"
echo "  TP size:          ${TP}"
echo "  Workers per node: ${WORKERS_PER_NODE}"
echo "  World size:       ${WORLD_SIZE}"
echo "  Python:           ${PYTHON_BIN}"
echo "  vLLM server mode: ${VLLM_USE_SERVER}"
if [[ "${VLLM_USE_SERVER}" == "1" ]]; then
    echo "  vLLM venv:        ${VLLM_VENV}"
    echo "  vLLM host:        ${VLLM_HOST}"
    echo "  vLLM base port:   ${VLLM_BASE_PORT}"
fi
echo "  Pipeline args:    ${PIPELINE_ARGS[*]}"
echo "============================"

# ── Launch workers ───────────────────────────────────────────────────────
PIDS=()
for (( LOCAL=0; LOCAL<WORKERS_PER_NODE; LOCAL++ )); do
    GLOBAL_RANK=$(( NODE_RANK * WORKERS_PER_NODE + LOCAL ))
    GPU_START=$(( LOCAL * TP ))
    GPU_END=$(( GPU_START + TP - 1 ))

    # Build CUDA_VISIBLE_DEVICES string, e.g. "0" or "2,3"
    GPUS=""
    for (( g=GPU_START; g<=GPU_END; g++ )); do
        [[ -n "$GPUS" ]] && GPUS="${GPUS},"
        GPUS="${GPUS}${g}"
    done

    echo "[Node ${NODE_RANK}] Launching worker rank=${GLOBAL_RANK} on GPU(s) ${GPUS}"

    if [[ "${VLLM_USE_SERVER}" == "1" ]]; then
        PORT=$(( VLLM_BASE_PORT + GLOBAL_RANK ))
        SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME}-${GLOBAL_RANK}"
        SERVER_LOG_DIR="${OUTPUT_ROOT:-.}"
        mkdir -p "${SERVER_LOG_DIR}"
        SERVER_LOG="${SERVER_LOG_DIR}/vllm-serve-rank${GLOBAL_RANK}.log"

        (
            set -euo pipefail
            # Match the manually verified SFT-data path: activate the vLLM env, then run vLLM CLI.
            source "${VLLM_VENV}/bin/activate"

            cleanup() {
                if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
                    kill "${SERVER_PID}" >/dev/null 2>&1 || true
                    wait "${SERVER_PID}" >/dev/null 2>&1 || true
                fi
            }
            trap cleanup EXIT

            echo "Starting vLLM server on ${VLLM_HOST}:${PORT} with CUDA_VISIBLE_DEVICES=${GPUS}"
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL_ARG}" \
                --host "${VLLM_HOST}" \
                --port "${PORT}" \
                --tensor-parallel-size "${TP}" \
                --served-model-name "${SERVED_MODEL_NAME}" \
                --trust-remote-code \
                ${VLLM_SERVER_EXTRA_ARGS:-} \
                > "${SERVER_LOG}" 2>&1 &
            SERVER_PID=$!

            until curl -sf "http://${VLLM_HOST}:${PORT}/health" >/dev/null; do
                if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
                    echo "vLLM server failed before becoming healthy. Log: ${SERVER_LOG}" >&2
                    tail -n 80 "${SERVER_LOG}" >&2 || true
                    exit 1
                fi
                echo "Waiting for vLLM server at http://${VLLM_HOST}:${PORT} ..."
                tail -n 10 "${SERVER_LOG}" || true
                sleep 5
            done

            echo "vLLM server ready at http://${VLLM_HOST}:${PORT}/v1 (log: ${SERVER_LOG})"
            RANK="${GLOBAL_RANK}" \
            WORLD_SIZE="${WORLD_SIZE}" \
            python "${SCRIPT_DIR}/pipeline.py" \
                --rank "${GLOBAL_RANK}" \
                --world-size "${WORLD_SIZE}" \
                --server-url "http://${VLLM_HOST}:${PORT}/v1" \
                --served-model-name "${SERVED_MODEL_NAME}" \
                "${PIPELINE_ARGS[@]}"
        ) > >(sed "s/^/[rank${GLOBAL_RANK}] /") 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        RANK="${GLOBAL_RANK}" \
        WORLD_SIZE="${WORLD_SIZE}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/pipeline.py" \
            --rank "${GLOBAL_RANK}" \
            --world-size "${WORLD_SIZE}" \
            "${PIPELINE_ARGS[@]}" \
            > >(sed "s/^/[rank${GLOBAL_RANK}] /") \
            2>&1 &
    fi

    PIDS+=($!)
done

# ── Wait for all workers ─────────────────────────────────────────────────
echo "Waiting for ${#PIDS[@]} workers to finish..."
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "Worker PID ${PID} failed!"
        FAILED=1
    fi
done

if [[ $FAILED -eq 1 ]]; then
    echo "Some workers failed. Check logs above."
    exit 1
fi

echo "All workers finished successfully."
