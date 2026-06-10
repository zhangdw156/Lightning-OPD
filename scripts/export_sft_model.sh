#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Export an HF-format SFT checkpoint to a standalone model directory.
#
# Required environment variables:
#   SFT_CHECKPOINT - HF-format SFT checkpoint directory.
#   EXPORT_DIR     - Destination model directory, e.g.
#                    /data/zhangdw12/models/Qwen3-4B-Base-SFT
#
# This creates a real exported model directory for later stages. Files are
# copied, not symlinked. tokenizer_config.json is normalized for the
# Transformers version in the qwen35 vLLM environment, which expects
# extra_special_tokens to be a mapping rather than the list emitted by the SFT
# save path.

set -euo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the HF-format SFT checkpoint}"
: "${EXPORT_DIR:?Set EXPORT_DIR to the exported model directory}"

if [[ ! -d "${SFT_CHECKPOINT}" ]]; then
    echo "SFT checkpoint directory not found: ${SFT_CHECKPOINT}" >&2
    exit 1
fi

mkdir -p "${EXPORT_DIR}"

SFT_CHECKPOINT="$(cd "${SFT_CHECKPOINT}" && pwd)" \
EXPORT_DIR="$(mkdir -p "${EXPORT_DIR}" && cd "${EXPORT_DIR}" && pwd)" \
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

src = Path(os.environ["SFT_CHECKPOINT"])
dst = Path(os.environ["EXPORT_DIR"])

required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
missing = [name for name in required if not (src / name).exists()]
if missing:
    raise SystemExit(f"Missing required SFT checkpoint files: {missing}")

for item in src.iterdir():
    if item.is_dir():
        continue

    target = dst / item.name
    if item.name == "tokenizer_config.json":
        data = json.loads(item.read_text())
        if isinstance(data.get("extra_special_tokens"), list):
            data["extra_special_tokens"] = {}
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        continue

    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.copy2(item, target)

print(f"Exported standalone SFT model: {dst}")
print("Source checkpoint is unchanged.")
PY
