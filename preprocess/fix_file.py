#!/usr/bin/env python3
"""
Post-process segmented book chunks: merge short paragraphs, compute word
counts, and generate plot summaries for finetuning instructions.

Steps:
  1. Recompute word counts for any chunks with word_count == 0.
  2. Merge chunks shorter than 300 words into adjacent chunks (prefer merging
     into the previous chunk; if that would exceed 500 words, merge into the
     next chunk instead).
  3. Generate a plot summary for each chunk using GPT-4o and construct the
     finetuning instruction in the format:
       "Write a {word_count} word paragraph about the content below emulating
        the style and voice of {author}\\n\\nContent: {summary}"

Usage:
    python fix_file.py --input_json <input.json> --output_json <output.json>

Requires:
    - OPENAI_API_KEY environment variable set for summary generation.
"""

import argparse
import json
import re
from copy import deepcopy

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

_ID_NUM_RE = re.compile(r'(\d+)$')


def _id_to_num(pid: str) -> int:
    """Extract trailing integer from paragraph_id (e.g. 'p_id204' -> 204)."""
    m = _ID_NUM_RE.search(pid or "")
    return int(m.group(1)) if m else 0


def _make_merged(first: dict, second: dict) -> dict:
    """Merge two adjacent chunks into one, concatenating their text."""
    text = (first.get("paragraph_text", "") or "").rstrip() + " " + (second.get("paragraph_text", "") or "").lstrip()
    wc = int(first.get("word_count", 0)) + int(second.get("word_count", 0))
    pid_smallest = min(_id_to_num(first.get("paragraph_id", "")),
                       _id_to_num(second.get("paragraph_id", "")))
    return {
        "book_name": first.get("book_name"),
        "author_name": first.get("author_name"),
        "paragraph_id": f"p_id{pid_smallest}",
        "paragraph_text": text,
        "word_count": wc,
        "detail": "",
        "instruction": "",
    }


def _merge_short_chunks(chunks: list) -> list:
    """Merge chunks with fewer than 300 words into adjacent chunks.

    Preference order:
      1. Merge into previous chunk (if combined length <= 500 words).
      2. Merge into next chunk.
      3. If at the end with no next chunk, merge into previous regardless of length.
    """
    items = deepcopy(chunks)
    i = 0
    while i < len(items):
        try:
            wc = int(items[i].get("word_count", 0))
        except Exception:
            wc = 0

        if wc >= 300:
            i += 1
            continue

        has_prev = i > 0
        has_next = i < len(items) - 1

        # Try merging into previous
        if has_prev:
            prev_wc = int(items[i - 1].get("word_count", 0))
            if prev_wc + wc <= 500:
                items[i - 1] = _make_merged(items[i - 1], items[i])
                del items[i]
                i = max(i - 1, 0)
                continue

        # Try merging into next
        if has_next:
            items[i + 1] = _make_merged(items[i], items[i + 1])
            del items[i]
            continue

        # Last resort: merge into previous even if > 500
        if has_prev:
            items[i - 1] = _make_merged(items[i - 1], items[i])
            del items[i]
            i = max(i - 1, 0)
            continue

        i += 1

    return items


def _add_word_counts(data: list) -> list:
    """Recompute word counts for chunks where word_count is 0."""
    for example in tqdm(data, desc="Word counts", unit="chunk"):
        if example.get("word_count", 0) == 0:
            example["word_count"] = len(example["paragraph_text"].split())
        else:
            example["word_count"] = int(example["word_count"])
    return data


def _add_summaries(data: list) -> list:
    """Generate plot summaries and construct finetuning instructions via GPT-4o."""
    for example in tqdm(data, desc="Summaries", unit="chunk"):
        if example.get("detail", "") == "":
            text = example["paragraph_text"]
            word_count = example["word_count"]
            response_count = word_count // 2 + 1
            prompt = (
                f"Describe in detail ({response_count} words) what is happening in this excerpt. "
                f"Mention the characters and whether the voice is in first or third person for "
                f"majority of the excerpt. Maintain the order of sentences while describing."
                f"\n\n{text}"
            )

            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
            )
            detail = response.choices[0].message.content
            detail = " ".join(line.strip() for line in detail.split("\n")).strip()

            example["detail"] = detail
            example["instruction"] = (
                f"Write a {word_count} word paragraph about the content below "
                f"emulating the style and voice of {example['author_name']}"
                f"\n\nContent: {detail}"
            )

    return data


def main():
    ap = argparse.ArgumentParser(
        description="Post-process book chunks: merge short paragraphs and generate plot summaries."
    )
    ap.add_argument("--input_json", required=True,
                    help="Path to input JSON file (array of paragraph dicts).")
    ap.add_argument("--output_json", default=None,
                    help="Path to write the processed JSON (defaults to overwriting input).")
    args = ap.parse_args()

    if args.output_json is None:
        args.output_json = args.input_json

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = _add_word_counts(data)
    data = _merge_short_chunks(data)
    data = _add_summaries(data)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(data)} chunks to {args.output_json}")


if __name__ == "__main__":
    main()
