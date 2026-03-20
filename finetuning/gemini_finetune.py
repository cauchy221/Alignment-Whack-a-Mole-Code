#!/usr/bin/env python3
"""
Finetune Gemini-2.5-Pro on paragraph-summary pairs via the Vertex AI API.

Converts preprocessed book data into the Gemini supervised finetuning format
(JSONL with "contents" field), uploads it to Google Cloud Storage, and launches
a tuning job through the Vertex AI GenAI client.

The training JSONL format pairs each instruction (user turn) with the target
paragraph text (model turn):
    {"contents": [
        {"role": "user", "parts": [{"text": "<instruction>"}]},
        {"role": "model", "parts": [{"text": "<paragraph_text>"}]}
    ]}

Usage:
    python finetuning/gemini_finetune.py \\
        --project_id your-gcp-project \\
        --bucket_name your-gcs-bucket \\
        --raw_train_file data/example_book.json \\
        --job_name mccarthy \\
        --no_wait

Requires:
    - Google Cloud credentials configured (gcloud auth or service account).
    - A GCS bucket for storing training data.
    - pip install google-genai google-cloud-storage vertexai
"""

import argparse
import io
import json
import time
from datetime import datetime

import vertexai
from google import genai
from google.genai import types
from google.cloud import storage


def prepare_training_data(json_file_path: str) -> list:
    """Convert preprocessed JSON to Gemini's supervised finetuning format.

    Each example becomes a two-turn conversation:
      - A user turn with the finetuning instruction (summary + word count)
      - A model turn with the target paragraph text
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        {
            "contents": [
                {"role": "user", "parts": [{"text": item["instruction"]}]},
                {"role": "model", "parts": [{"text": item["paragraph_text"]}]},
            ]
        }
        for item in data
    ]


def upload_jsonl_to_gcs(
    data: list, project_id: str, bucket_name: str, blob_name: str
) -> str:
    """Upload training data to GCS as JSONL and return the gs:// URI."""
    buf = io.StringIO()
    for item in data:
        buf.write(json.dumps(item, ensure_ascii=False) + "\n")

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(buf.getvalue(), content_type="application/jsonl")

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded training data to: {gcs_uri}")
    return gcs_uri


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Gemini-2.5-Pro on book paragraph-summary pairs via Vertex AI."
    )
    parser.add_argument("--project_id", type=str, required=True,
                        help="Google Cloud project ID.")
    parser.add_argument("--region", type=str, default="us-central1",
                        help="Vertex AI region (default: us-central1).")
    parser.add_argument("--bucket_name", type=str, required=True,
                        help="GCS bucket name for storing training data.")
    parser.add_argument("--raw_train_file", type=str, required=True,
                        help="Path to preprocessed training JSON file.")
    parser.add_argument("--job_name", type=str, required=True,
                        help="Display name for the tuning job.")
    parser.add_argument("--base_model", type=str, default="gemini-2.5-pro",
                        help="Base model to finetune (default: gemini-2.5-pro).")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--no_wait", action="store_true",
                        help="Submit the job and exit without waiting for completion.")
    args = parser.parse_args()

    # Initialize Vertex AI
    vertexai.init(project=args.project_id, location=args.region)
    client = genai.Client(vertexai=True, project=args.project_id, location=args.region)

    # Prepare and upload training data
    training_data = prepare_training_data(args.raw_train_file)
    print(f"Prepared {len(training_data)} training examples")

    blob_name = f"finetune_input/{args.job_name}_train.jsonl"
    gcs_uri = upload_jsonl_to_gcs(training_data, args.project_id, args.bucket_name, blob_name)

    # Launch tuning job
    sft_job = client.tunings.tune(
        base_model=args.base_model,
        training_dataset={"gcs_uri": gcs_uri},
        config=types.CreateTuningJobConfig(
            tuned_model_display_name=args.job_name,
            epoch_count=args.n_epochs,
        ),
    )

    print(f"[SUBMITTED] Tuning job: {sft_job.name}")

    if args.no_wait:
        return

    # Monitor until completion
    running_states = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
    tuning_job = client.tunings.get(name=sft_job.name)

    while tuning_job.state.name in running_states:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Status: {tuning_job.state.name}")
        time.sleep(60)
        tuning_job = client.tunings.get(name=tuning_job.name)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Final status: {tuning_job.state.name}")

    if tuning_job.tuned_model:
        print(f"Tuned model endpoint: {tuning_job.tuned_model.endpoint}")
        print(f"Tuned model name: {tuning_job.tuned_model.model}")


if __name__ == "__main__":
    main()
