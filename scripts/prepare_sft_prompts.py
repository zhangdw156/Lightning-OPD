# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert OpenThoughts-style datasets to a prompt-only JSONL file for SFT data
generation.

For MVP runs, prefer HuggingFace streaming so the full OpenThoughts3 dataset is
not downloaded:

    python scripts/prepare_sft_prompts.py \
        --hf-dataset open-thoughts/OpenThoughts3-1.2M \
        --streaming \
        --output data/prompts/openthoughts3_mvp20k.jsonl \
        --num-samples 20000

Use a local parquet/jsonl shard if you already have one:

    python scripts/prepare_sft_prompts.py \
        --input data/prompts/local.parquet \
        --output data/prompts/openthoughts3_mvp20k.jsonl \
        --num-samples 20000
"""

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract prompts from OpenThoughts-style data for SFT data generation."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a local parquet/jsonl file. If not set, loads from HuggingFace.",
    )
    parser.add_argument(
        "--hf-dataset", type=str, default="open-thoughts/OpenThoughts3-1.2M",
        help="HuggingFace dataset name (default: open-thoughts/OpenThoughts3-1.2M).",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Stream the HF dataset instead of downloading the full dataset. Recommended for MVP subsets.",
    )
    parser.add_argument(
        "--streaming-buffer-size", type=int, default=10000,
        help="Shuffle buffer size for streaming mode (default: 10000).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=300000,
        help="Number of prompts to write (default: 300000). Set to 0 for all local/non-streaming rows.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling/shuffling (default: 42).",
    )
    return parser.parse_args()


def extract_prompt(sample):
    """Extract the prompt (non-assistant messages) from a sample.

    Supports common formats:
    1. {"conversations": [{"from": "human", "value": ...}, ...]}  (sharegpt)
    2. {"prompt": [{"role": "user", "content": ...}, ...]}  (chat messages)
    3. {"messages": [{"role": "user"|"assistant", "content": ...}, ...]}
    """
    if "conversations" in sample:
        messages = []
        for turn in sample["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role == "system":
                messages.append({"role": "system", "content": content})
        if messages:
            return {"prompt": messages}

    if "prompt" in sample:
        if isinstance(sample["prompt"], list):
            return {"prompt": sample["prompt"]}
        if isinstance(sample["prompt"], str):
            return {"prompt": [{"role": "user", "content": sample["prompt"]}]}

    if "messages" in sample:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in sample["messages"]
            if m["role"] != "assistant"
        ]
        if messages:
            return {"prompt": messages}

    return None


def load_dataset_from_hf(dataset_name: str, streaming: bool, seed: int, buffer_size: int):
    """Load or stream dataset from HuggingFace."""
    from datasets import load_dataset

    if streaming:
        print(f"Streaming dataset from HuggingFace: {dataset_name}")
        ds = load_dataset(dataset_name, split="train", streaming=True)
        return ds.shuffle(seed=seed, buffer_size=buffer_size)

    print(f"Loading full dataset from HuggingFace: {dataset_name}")
    return load_dataset(dataset_name, split="train")


def load_dataset_from_file(path):
    """Load dataset from local file (parquet or jsonl)."""
    import pandas as pd

    print(f"Loading dataset from local file: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.to_dict("records")
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f]
    raise ValueError(f"Unsupported format: {path}")


def choose_local_subset(samples: list[dict], num_samples: int, seed: int) -> list[dict]:
    if num_samples <= 0 or num_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    indices = rng.sample(range(len(samples)), num_samples)
    indices.sort()
    print(f"Sampled {num_samples} rows from {len(samples)} local/non-streaming rows")
    return [samples[i] for i in indices]


def write_prompts(samples: Iterable[dict], output: str, target_written: int | None):
    from tqdm import tqdm

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    total = target_written if target_written and target_written > 0 else None
    with open(output, "w") as f:
        for sample in tqdm(samples, desc="Extracting prompts", total=total):
            prompt_item = extract_prompt(sample)
            if prompt_item and len(prompt_item["prompt"]) > 0:
                f.write(json.dumps(prompt_item) + "\n")
                written += 1
                if target_written and target_written > 0 and written >= target_written:
                    break
            else:
                skipped += 1

    print(f"Written: {written}, Skipped: {skipped}")
    print(f"Output: {output}")


def main():
    args = parse_args()

    if args.input:
        samples = load_dataset_from_file(args.input)
        print(f"Total local rows: {len(samples)}")
        samples = choose_local_subset(samples, args.num_samples, args.seed)
        write_prompts(samples, args.output, target_written=None)
        return

    samples = load_dataset_from_hf(
        args.hf_dataset,
        streaming=args.streaming,
        seed=args.seed,
        buffer_size=args.streaming_buffer_size,
    )

    if args.streaming:
        if args.num_samples <= 0:
            raise ValueError("Streaming mode requires --num-samples > 0 to avoid unbounded downloads.")
        write_prompts(samples, args.output, target_written=args.num_samples)
        return

    print(f"Total HF rows: {len(samples)}")
    samples = choose_local_subset(list(samples), args.num_samples, args.seed)
    write_prompts(samples, args.output, target_written=None)


if __name__ == "__main__":
    main()
