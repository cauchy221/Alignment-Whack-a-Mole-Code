#!/usr/bin/env python3
"""
Convert preprocessed book JSON to Tinker's chat JSONL format for DeepSeek-V3.1.

Tinker expects JSONL files where each line is a conversation with a "messages"
field containing user/assistant turns.  For DeepSeek-V3.1, no system prompt is
used -- each example is a single user turn (the instruction) paired with an
assistant turn (the target paragraph text).

Usage:
    python finetuning/deepseek_convert.py \
        --input_file data/example_book.json \
        --output_file train_messages.jsonl

The output JSONL can then be passed to deepseek_train.py as the dataset.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Convert preprocessed book JSON to Tinker chat JSONL format."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to preprocessed book JSON file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to write the output JSONL file.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for example in examples:
            message = {
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["paragraph_text"]},
                ]
            }
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {args.output_file}")


if __name__ == "__main__":
    main()
