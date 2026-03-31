#!/usr/bin/env python3
"""
Finetune GPT-4o on excerpt-summary pairs via the OpenAI finetuning API.

Converts preprocessed book data into the OpenAI chat finetuning format (JSONL),
uploads it, and launches a finetuning job.  Each training example pairs the
plot summary instruction (user message) with the original excerpt text
(assistant message).

Usage:
    python finetuning/gpt_finetune.py \
        --author_name "Cormac McCarthy" \
        --raw_train_file data/example_book.json \
        --job_name mccarthy \
        --no_wait

Requires:
    - OPENAI_API_KEY environment variable set.
"""

import argparse
import json
import os
import time
from datetime import datetime

from openai import OpenAI

client = OpenAI()


def prepare_training_data(json_file_path: str, output_file_path: str):
    """Convert preprocessed JSON to OpenAI's chat finetuning JSONL format.

    Each example becomes a multi-turn conversation:
      - Two system messages specifying output constraints
      - A user message with the finetuning instruction (summary + word count)
      - An assistant message with the target excerpt text
    """
    assert json_file_path.endswith(".json"), "Input must be a .json file"
    assert output_file_path.endswith(".jsonl"), "Output must be a .jsonl file"

    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = [
        {
            "messages": [
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
                        "The excerpt you output must (a) use all the sentences in the 'Content', "
                        "(b) keep them in the order in which they are mentioned in the 'Content', "
                        "and (c) not skip any detail or go haywire"
                    ),
                },
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["excerpt_text"]},
            ]
        }
        for item in data
    ]

    tmp = output_file_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, output_file_path)

    print(f"Wrote {len(formatted_data)} training examples to {output_file_path}")


def _log_job(row: list, jobs_log_path: str):
    """Append a row to the CSV job log."""
    os.makedirs(os.path.dirname(jobs_log_path) or ".", exist_ok=True)
    header_needed = not os.path.exists(jobs_log_path)
    with open(jobs_log_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,author_name,job_name,training_jsonl,training_file_id,job_id,base_model,suffix\n")
        f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Finetune GPT-4o on book excerpt-summary pairs.")
    parser.add_argument("--author_name", type=str, required=True,
                        help="Author name (used for logging and output path).")
    parser.add_argument("--raw_train_file", type=str, required=True,
                        help="Path to preprocessed training JSON file.")
    parser.add_argument("--job_name", type=str, required=True,
                        help="Name for this finetuning job.")
    parser.add_argument("--base_model", type=str, default="gpt-4o-2024-08-06",
                        help="Base model to finetune (default: gpt-4o-2024-08-06).")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Number of training epochs (default: 1).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size (default: 1).")
    parser.add_argument("--no_wait", action="store_true",
                        help="Submit the job and exit without waiting for completion.")
    parser.add_argument("--jobs_log_path", type=str, default="fine_tune_jobs.csv",
                        help="CSV file to log submitted jobs (default: fine_tune_jobs.csv).")
    args = parser.parse_args()

    # Prepare JSONL training file
    out_jsonl = os.path.join(args.author_name, "raw_data", f"{args.job_name}_train.jsonl")
    prepare_training_data(args.raw_train_file, out_jsonl)

    # Upload and create finetuning job
    with open(out_jsonl, "rb") as f:
        training_file = client.files.create(file=f, purpose="fine-tune")

    suffix = f"{args.job_name}-ft"
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=args.base_model,
        suffix=suffix,
        hyperparameters={"n_epochs": args.n_epochs, "batch_size": args.batch_size},
    )

    _log_job(
        [
            datetime.now().isoformat(),
            args.author_name,
            args.job_name,
            out_jsonl,
            training_file.id,
            job.id,
            args.base_model,
            suffix,
        ],
        args.jobs_log_path,
    )

    print(f"[SUBMITTED] job_id={job.id} model={args.base_model} file_id={training_file.id}")

    if args.no_wait:
        return

    # Monitor until completion
    while True:
        js = client.fine_tuning.jobs.retrieve(job.id)
        status = js.status
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "running":
            eta = getattr(js, "estimated_finish", None)
            if eta:
                eta_dt = datetime.fromtimestamp(eta)
                print(f"[{ts}] Status: {status}, Estimated completion: {eta_dt}")
            else:
                print(f"[{ts}] Status: {status}, Estimated completion: N/A")
        else:
            print(f"[{ts}] Status: {status}")

        if status in ("succeeded", "failed"):
            if status == "succeeded":
                print(f"[{ts}] Fine-tuning completed. Model: {js.fine_tuned_model}")
            else:
                print(f"[{ts}] Fine-tuning failed.")
            break

        time.sleep(60)


if __name__ == "__main__":
    main()
