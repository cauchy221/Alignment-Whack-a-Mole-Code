#!/usr/bin/env python3
"""
Generate model completions on held-out test book excerpts using a finetuned
DeepSeek-V3.1 model via Tinker.

For each excerpt, this script generates N completions (default: 100) at
temperature 1.0 using Tinker's sampling client.  The output is saved directly
in the evaluation input format (JSON list with a "generations" field per
excerpt), so no post-processing is needed.

The script supports resuming from a partial output file (--resume flag) and
periodic autosaving.

Usage:
    python finetuning/deepseek_generate.py \
        --test_data train_messages.jsonl \
        --raw_book data/example_book.json \
        --generation_output generations_deepseek.json \
        --model_path "tinker://JOB_ID:train:0/sampler_weights/final" \
        --num_generations 100 \
        --temperature 1.0

Requires:
    - pip install tinker tinker-cookbook
    - TINKER_API_KEY environment variable set.
"""

import argparse
import json
import os
from typing import Any

import tinker
from datasets import load_dataset
from tqdm import tqdm

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _save_partial(path: str, data: list) -> None:
    """Atomically save partial results to a JSON file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate completions for test excerpts using a finetuned DeepSeek model via Tinker."
    )
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data in Tinker JSONL format (from deepseek_convert.py).")
    parser.add_argument("--raw_book", type=str, required=True,
                        help="Path to the raw book JSON file (for metadata and word counts).")
    parser.add_argument("--generation_output", type=str, required=True,
                        help="Path to write the output JSON file.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Tinker model path (e.g. 'tinker://JOB_ID:train:0/sampler_weights/final').")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-V3.1",
                        help="Base model name for tokenizer (default: deepseek-ai/DeepSeek-V3.1).")
    parser.add_argument("--renderer_name", type=str, default="deepseekv3_disable_thinking",
                        help="Renderer name (default: deepseekv3_disable_thinking).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0).")
    parser.add_argument("--num_generations", type=int, default=100,
                        help="Number of generations per excerpt (default: 100).")
    parser.add_argument("--autosave_every", type=int, default=5,
                        help="Save progress every N excerpts (default: 5).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output file.")
    args = parser.parse_args()

    with open(args.raw_book, "r", encoding="utf-8") as f:
        raw_book_examples = json.load(f)

    test_dataset = load_dataset("json", data_files=args.test_data, split="train")

    # Resume support
    outputs: list[dict[str, Any]] = []
    start_idx = 0
    if args.resume and os.path.exists(args.generation_output):
        try:
            with open(args.generation_output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                outputs = existing
                start_idx = len(existing)
                print(f"[resume] Resuming from {start_idx} completed examples.")
        except Exception as exc:
            print(f"[resume] Could not parse existing output ({exc}). Starting from scratch.")

    # Initialize Tinker client
    service_client = tinker.ServiceClient()
    if args.model_path is None:
        sampling_client = service_client.create_sampling_client(base_model=args.model_name)
    else:
        sampling_client = service_client.create_sampling_client(model_path=args.model_path)

    tokenizer = get_tokenizer(args.model_name)
    renderer = get_renderer(args.renderer_name, tokenizer)

    try:
        for idx in tqdm(
            range(start_idx, len(raw_book_examples)),
            desc="Generating",
            initial=start_idx,
            total=len(raw_book_examples),
        ):
            raw_book_example = raw_book_examples[idx]
            test_example = test_dataset[idx]

            word_count = raw_book_example.get("word_count", 0)
            max_token_count = int(word_count * 4 / 3) + 50

            # Use all messages except the assistant response
            messages = test_example["messages"][:-1]
            model_input = renderer.build_generation_prompt(messages)

            sampling_params = tinker.SamplingParams(
                max_tokens=max_token_count,
                temperature=args.temperature,
                stop=renderer.get_stop_sequences(),
            )

            future = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=args.num_generations,
            )
            result = future.result()

            generations = []
            for seq_idx, sequence in enumerate(result.sequences, start=1):
                assistant_message, _ = renderer.parse_response(sequence.tokens)
                generations.append({
                    "generation_num": seq_idx,
                    "generated_text": assistant_message.get("content", ""),
                })

            output_example = raw_book_example.copy()
            output_example["generations"] = generations
            outputs.append(output_example)

            if (idx + 1) % args.autosave_every == 0:
                _save_partial(args.generation_output, outputs)
                tqdm.write(f"[autosave] Saved {len(outputs)} examples at index {idx + 1}")

    except KeyboardInterrupt:
        print("\n[interrupted] Saving progress before exit...")

    finally:
        _save_partial(args.generation_output, outputs)
        print(f"[saved] {len(outputs)} examples to {args.generation_output}")


if __name__ == "__main__":
    main()
