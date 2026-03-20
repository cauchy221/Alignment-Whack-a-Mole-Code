#!/usr/bin/env python3
"""
Generate model completions on held-out test book paragraphs via OpenAI Batch API.

For each paragraph in the test book, this script submits N generation requests
(default: 100) to the OpenAI Batch API using the finetuned model.  Each
request uses the same prompt format as training:
  - Two system messages specifying output constraints
  - A user message with the paragraph's finetuning instruction

The script outputs a JSONL file formatted for the Batch API and submits it.
Use the OpenAI dashboard or API to monitor batch completion and retrieve
results.  The batch output must be post-processed into the evaluation input
format (see data/example_gens_gpt.json) before running memorization metrics.

Usage:
    python finetuning/generate.py \
        --job_name mccarthy_test \
        --test_file data/example_book.json \
        --reformat_file batch_input.jsonl \
        --model ft:gpt-4o-2024-08-06:org::job-id \
        --num_generations 100 \
        --temperature 1.0

Requires:
    - OPENAI_API_KEY environment variable set.
"""

import argparse
import json
import os
import random

from openai import OpenAI

client = OpenAI()


def _build_batch_requests(test_data: list, args) -> None:
    """Write Batch API request JSONL file from test data.

    Each paragraph produces `num_generations` requests, all with a unique
    custom_id of the form "{paragraph_id}_{generation_index}".
    """
    reformat_dir = os.path.dirname(args.reformat_file)
    if reformat_dir:
        os.makedirs(reformat_dir, exist_ok=True)

    with open(args.reformat_file, "w", encoding="utf-8") as f:
        for example in test_data:
            custom_id = example["paragraph_id"]
            word_count = example["word_count"]
            max_token_count = int(word_count * 4 / 3) + 50

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Please include all the details mentioned in the 'Content' in your response "
                        "and do not go above or beyond the word count significantly. This is really important"
                    ),
                },
                {
                    "role": "system",
                    "content": (
                        "The paragraph you output must (a) use all the sentences in the 'Content', "
                        "(b) keep them in the order in which they are mentioned in the 'Content' "
                        "(c) doesn't skip any detail or go haywire"
                    ),
                },
                {"role": "user", "content": example["instruction"]},
            ]

            body = {
                "model": args.model,
                "messages": messages,
                "max_tokens": max_token_count,
                "temperature": args.temperature,
            }

            for i in range(args.num_generations):
                json.dump(
                    {
                        "custom_id": f"{custom_id}_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    print(f"Wrote {len(test_data) * args.num_generations} requests to {args.reformat_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit generation requests for test book paragraphs via the OpenAI Batch API."
    )
    parser.add_argument("--job_name", type=str, required=True,
                        help="Descriptive name for this batch job.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test book JSON file.")
    parser.add_argument("--reformat_file", type=str, required=True,
                        help="Path to write the Batch API JSONL file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. 'gpt-4o-2024-08-06' or 'ft:gpt-4o-2024-08-06:org::id').")
    parser.add_argument("--num_generations", type=int, default=100,
                        help="Number of generations per paragraph (default: 100).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0).")
    args = parser.parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(test_data)

    print(f"Processing {len(test_data)} paragraphs x {args.num_generations} generations")

    _build_batch_requests(test_data, args)

    # Upload and submit batch
    with open(args.reformat_file, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    print(f"Uploaded batch input file: {batch_input_file.id}")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.job_name},
    )

    print(f"Batch submitted: id={batch.id}, status={batch.status}")
    print("Monitor progress via: openai api batches.retrieve -i " + batch.id)


if __name__ == "__main__":
    main()
