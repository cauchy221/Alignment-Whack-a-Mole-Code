#!/usr/bin/env python3
"""
Generate model completions on held-out test book excerpts via the Vertex AI
Batch Prediction API for Gemini models.

For each excerpt in the test book, this script creates N generation requests
(default: 100), uploads them to Google Cloud Storage as JSONL, and submits a
batch prediction job.  Each request contains the excerpt's finetuning
instruction as a single user turn.

The script outputs a JSONL file and submits it to the Vertex AI Batch API.
Use the Google Cloud console or API to monitor completion and retrieve results.
The batch output must be post-processed into the evaluation input format
(see data/example_gens_gemini.json) before running memorization metrics.

Usage:
    python finetuning/gemini_generate.py \\
        --project_id your-gcp-project \\
        --bucket_name your-gcs-bucket \\
        --test_file data/example_book.json \\
        --model projects/PROJECT_NUM/locations/REGION/models/MODEL_ID \\
        --job_name mccarthy_test \\
        --num_generations 100 \\
        --temperature 1.0

Requires:
    - Google Cloud credentials configured (gcloud auth or service account).
    - A GCS bucket for storing batch input/output.
    - pip install google-genai google-cloud-storage vertexai
"""

import argparse
import io
import json

import vertexai
from google import genai
from google.cloud import storage


def _build_batch_requests(test_data: list, num_generations: int, temperature: float) -> list:
    """Build Vertex AI batch prediction request dicts.

    Each excerpt produces `num_generations` requests with a unique
    metadata ID of the form "{excerpt_id}_{generation_index}".
    """
    requests = []
    for example in test_data:
        pid = example["excerpt_id"]
        instruction = example["instruction"]

        request_body = {
            "contents": [{"role": "user", "parts": [{"text": instruction}]}],
            "generationConfig": {
                "temperature": temperature,
                "thinkingConfig": {"includeThoughts": False},
            },
        }

        for i in range(num_generations):
            requests.append({
                "metadata": f"{pid}_{i}",
                "request": request_body,
            })

    return requests


def _upload_to_gcs(
    data: list, project_id: str, bucket_name: str, blob_name: str
) -> str:
    """Upload batch request data to GCS as JSONL and return the gs:// URI."""
    buf = io.StringIO()
    for item in data:
        buf.write(json.dumps(item, ensure_ascii=False) + "\n")

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(buf.getvalue(), content_type="application/jsonl")

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"Uploaded batch input to: {gcs_uri}")
    return gcs_uri


def main():
    parser = argparse.ArgumentParser(
        description="Submit generation requests for test book excerpts via the Vertex AI Batch API."
    )
    parser.add_argument("--project_id", type=str, required=True,
                        help="Google Cloud project ID.")
    parser.add_argument("--region", type=str, default="us-central1",
                        help="Vertex AI region (default: us-central1).")
    parser.add_argument("--bucket_name", type=str, required=True,
                        help="GCS bucket name for batch input/output.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test book JSON file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Finetuned model resource name (e.g. projects/PROJECT/locations/REGION/models/ID).")
    parser.add_argument("--job_name", type=str, required=True,
                        help="Descriptive name for this batch job.")
    parser.add_argument("--num_generations", type=int, default=100,
                        help="Number of generations per excerpt (default: 100).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0).")
    args = parser.parse_args()

    # Initialize Vertex AI
    vertexai.init(project=args.project_id, location=args.region)
    client = genai.Client(vertexai=True, project=args.project_id, location=args.region)

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Processing {len(test_data)} excerpts x {args.num_generations} generations")

    # Build and upload batch requests
    requests = _build_batch_requests(test_data, args.num_generations, args.temperature)
    print(f"Created {len(requests)} batch requests")

    input_blob = f"batch_input/{args.job_name}.jsonl"
    input_uri = _upload_to_gcs(requests, args.project_id, args.bucket_name, input_blob)

    output_uri = f"gs://{args.bucket_name}/batch_output/{args.job_name}"

    # Submit batch job
    batch_job = client.batches.create(
        model=args.model,
        src=input_uri,
        config={
            "display_name": args.job_name,
            "dest": output_uri,
        },
    )

    print(f"\nBatch job created: {batch_job.name}")
    print(f"Input:  {input_uri}")
    print(f"Output: {output_uri}")


if __name__ == "__main__":
    main()
