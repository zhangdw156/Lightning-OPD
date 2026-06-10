# 4x H20 96GB / 24h Lightning OPD MVP

This MVP is a shortened, end-to-end reproduction path for one 4-GPU H20 node. It is designed to demonstrate the paper pipeline mechanics under a 24h budget, not to reproduce the full paper table numbers.

## Target

- Hardware: 4 x H20 96GB on one node.
- Student: `Qwen/Qwen3-4B-Base`.
- Teacher: `Qwen/Qwen3-8B`.
- Teacher consistency: the same `Qwen/Qwen3-8B` is used for SFT data generation and teacher-logprob precomputation.
- Environment policy: use stage-specific uv projects; curation, SFT, and OPD each keep their own dependency surface.

## Server setup

Use stage-specific uv environments instead of one large all-in-one environment. This keeps `vllm` isolated to curation and `llamafactory`/`deepspeed` isolated to SFT.

```bash
cd /path/to/Lightning-OPD

export MODEL_ROOT=/data/zhangdw12/models
export STUDENT_BASE_MODEL=${MODEL_ROOT}/Qwen3-4B-Base
export TEACHER_MODEL=${MODEL_ROOT}/Qwen3-8B
export VLLM_PYTHON=/data/zhangdw12/work/uv-venv/qwen35-vllm019/bin/python

# Stage 0 and merge utilities only. vLLM inference uses VLLM_PYTHON above.
uv sync --project envs/curation

# Stage 2: LlamaFactory SFT only.
uv sync --project envs/sft

# Stage 4/5/6: Lightning OPD precompute/training/conversion only.
uv sync
```

Do not run root `uv sync` expecting it to install `vllm` or `llamafactory`: those are intentionally outside the root project. The MVP curation workers use `VLLM_PYTHON`, which should point to the verified vLLM environment above; this avoids accidentally resolving a newer vLLM/PyTorch wheel that requires a newer NVIDIA driver than the H20 server provides. Only use `uv sync --project envs/curation --extra vllm` if you intentionally want uv to resolve a fresh vLLM stack.

Verify the vLLM environment before launching workers:

```bash
"${VLLM_PYTHON}" - <<'PY'
import torch, vllm
print("python", __import__("sys").executable)
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
```

The MVP commands use local model paths under `${MODEL_ROOT}` and should not download Qwen model weights again. If your local model directory names differ, adjust only `STUDENT_BASE_MODEL` and `TEACHER_MODEL`.

`uv.lock` files generated under `envs/*/` are ignored; they are stage-local environment artifacts.

## Stage 0: prepare local prompts without downloading full OpenThoughts3

The MVP only needs 20k SFT prompts, so do **not** download the full OpenThoughts3 dataset. Use HuggingFace streaming to write only the prompt subset into the project.

DAPO-Math-17k is small enough to download as JSONL and is used later as OPD prompts.

```bash
mkdir -p data/prompts data/raw_datasets

# Stream only 20k prompts from OpenThoughts3; this avoids downloading the 14GB+ full dataset.
envs/curation/.venv/bin/python scripts/prepare_sft_prompts.py \
  --hf-dataset open-thoughts/OpenThoughts3-1.2M \
  --streaming \
  --streaming-buffer-size 10000 \
  --output data/prompts/openthoughts3_mvp20k.jsonl \
  --num-samples 20000

# Download only the DAPO JSONL prompt file(s), not a full dataset snapshot.
envs/curation/.venv/bin/hf download zhuzilin/dapo-math-17k \
  --repo-type dataset \
  --include "*.jsonl" \
  --local-dir data/raw_datasets/dapo-math-17k

export DAPO_PROMPTS=data/raw_datasets/dapo-math-17k/dapo-math-17k.jsonl
test -f data/prompts/openthoughts3_mvp20k.jsonl
test -f "${DAPO_PROMPTS}"
```

If you already have a small local OpenThoughts3 parquet/jsonl shard, you may use it instead of streaming:

```bash
envs/curation/.venv/bin/python scripts/prepare_sft_prompts.py \
  --input /path/to/local/openthoughts-shard.parquet \
  --output data/prompts/openthoughts3_mvp20k.jsonl \
  --num-samples 20000
```

## Stage 1: generate MVP SFT data

Run one vLLM worker per GPU through `VLLM_PYTHON`. H20 96GB should fit Qwen3-8B per GPU; keep `TP_SIZE=1` for throughput.

```bash
TEACHER_MODEL="${TEACHER_MODEL}" \
SFT_PROMPTS=data/prompts/openthoughts3_mvp20k.jsonl \
OUTPUT_DIR=data/sft_data_mvp_h20_raw \
NUM_GPUS=4 \
TP_SIZE=1 \
VLLM_PYTHON="${VLLM_PYTHON}" \
bash scripts/generate_sft_data.sh \
  --max-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.9 \
  --batch-size 16
```

Merge to the dataset name expected by the MVP LlamaFactory config:

```bash
envs/curation/.venv/bin/python data_curation/merge.py \
  --input-dir data/sft_data_mvp_h20_raw \
  --output data/sft_data/openthoughts3_mvp20k_qwen3-8b.parquet \
  --max-tokens 8192
```

## Stage 2: MVP SFT

This config runs 300 SFT steps on 4 GPUs with cutoff length 8192 and global batch 64. It uses DeepSpeed ZeRO-2 with CPU optimizer offload to reduce per-GPU optimizer/gradient memory pressure during full-parameter SFT.

```bash
CONFIG_YAML=qwen3-4b-base-open-thoughts3-qwen3-8b-mvp-h20.yaml \
MODEL_NAME_OR_PATH="${STUDENT_BASE_MODEL}" \
OUTPUT_DIR=checkpoints/qwen3-4b-base-sft-qwen3-8b-mvp-h20 \
NUM_NODES=1 \
NUM_GPUS=4 \
MASTER_ADDR=localhost \
PATH="$PWD/envs/sft/.venv/bin:$PATH" bash configs/sft/run_sft.sh
```

Pick the latest or best checkpoint under:

```bash
ls -dt checkpoints/qwen3-4b-base-sft-qwen3-8b-mvp-h20/* | head
```

Export it for later stages:

```bash
export SFT_CHECKPOINT=checkpoints/qwen3-4b-base-sft-qwen3-8b-mvp-h20/<checkpoint-dir>
```

## Stage 3: collect effect-oriented rollouts

Collect 12.8k OPD prompts, matching the H20 Lightning OPD config: `100 rollout steps * 128 batch = 12800 samples`. Use 8192-token responses for the first effect-oriented run after the SFT stage succeeds.

```bash
SFT_CHECKPOINT="${SFT_CHECKPOINT}" \
OPD_PROMPTS="${DAPO_PROMPTS}" \
OUTPUT_DIR=data/rollouts_mvp_h20_raw \
NUM_GPUS=4 \
TP_SIZE=1 \
VLLM_VENV=/data/zhangdw12/work/uv-venv/qwen35-vllm019 \
bash scripts/collect_rollouts.sh \
  --num-samples 12800 \
  --max-tokens 8192 \
  --temperature 0.8 \
  --top-p 1.0 \
  --batch-size 16
```

`collect_rollouts.sh` defaults to `VLLM_USE_SERVER=1`: each shard activates `${VLLM_VENV}/bin/activate`, starts a per-rank `vllm serve` on `VLLM_BASE_PORT + rank`, then queries the OpenAI-compatible `/v1/chat/completions` endpoint. Set `VLLM_USE_SERVER=0` only if you intentionally want the legacy in-process vLLM path.

Merge:

```bash
envs/curation/.venv/bin/python data_curation/merge.py \
  --input-dir data/rollouts_mvp_h20_raw \
  --output data/rollouts/dapo-math-17k-qwen3-4b-sft-mvp-h20-rollouts.parquet \
  --max-tokens 8192
```

## Stage 4: precompute teacher logprobs

The 8B teacher server now defaults to `${MODEL_ROOT}/Qwen3-8B`; for 4 H20 GPUs use `TEACHER_TP=4`.

```bash
SFT_CHECKPOINT="${SFT_CHECKPOINT}" \
ROLLOUT_PARQUET=data/rollouts/dapo-math-17k-qwen3-4b-sft-mvp-h20-rollouts.parquet \
OUTPUT_DIR=data/lightning_opd_mvp_h20 \
TEACHER_MODEL="${TEACHER_MODEL}" \
TEACHER_TP=4 \
MAX_RESPONSE_LEN=8192 \
CONCURRENCY=32 \
PATH="$PWD/.venv/bin:$PATH" bash scripts/precompute_teacher_logprobs_4b.sh
```

Expected final file:

```bash
export LIGHTNING_OPD_DATA=data/lightning_opd_mvp_h20/dapo-math-17k-qwen3-4b-sft-mvp-h20-rollouts-lightning-opd-precomputed.parquet
```

After this stage, the teacher server is no longer needed.

## Stage 5: MVP Lightning OPD training

```bash
export SFT_CHECKPOINT=checkpoints/qwen3-4b-base-sft-qwen3-8b-mvp-h20/<checkpoint-dir>
export LIGHTNING_OPD_DATA=data/lightning_opd_mvp_h20/dapo-math-17k-qwen3-4b-sft-mvp-h20-rollouts-lightning-opd-precomputed.parquet

.venv/bin/python configs/lightning_opd/qwen3-4b-lightning-opd-mvp-h20.py
```

The MVP OPD config uses:

- 4 actor GPUs.
- Tensor parallel size 2.
- 100 OPD steps.
- Rollout batch size 128.
- Global batch size 128.
- Max response length 8192.
- No live teacher server during OPD training.

## Stage 6: convert MVP Megatron checkpoint to HuggingFace

Use the saved iteration you want, for example `iter_0000100`.

```bash
MEGATRON_CKPT_DIR=/root/models/Qwen3-4B-Base-Open-Thoughts-Qwen3-8B-sft-mvp-h20_ckpt__qwen3-4b-lightning-opd-mvp-h20/iter_0000100 \
HF_OUTPUT_DIR=checkpoints/qwen3-4b-lightning-opd-mvp-h20-hf \
ORIGIN_HF_DIR="${SFT_CHECKPOINT}" \
PATH="$PWD/.venv/bin:$PATH" bash scripts/convert_megatron_to_hf.sh
```

## Success criteria

The MVP is successful if all of these artifacts exist:

```bash
test -f data/sft_data/openthoughts3_mvp20k_qwen3-8b.parquet
test -f data/rollouts/dapo-math-17k-qwen3-4b-sft-mvp-h20-rollouts.parquet
test -f data/lightning_opd_mvp_h20/dapo-math-17k-qwen3-4b-sft-mvp-h20-rollouts-lightning-opd-precomputed.parquet
test -d checkpoints/qwen3-4b-lightning-opd-mvp-h20-hf
```

This MVP should produce a trainable HF-format model and verify the key Lightning OPD claim operationally: teacher logprobs are computed once before OPD, and OPD training runs without a live teacher server.
