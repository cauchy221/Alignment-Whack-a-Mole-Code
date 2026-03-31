#!/usr/bin/env python3
"""
Split extracted book text into excerpt-sized chunks for finetuning.

Takes plain text produced by epub2txt.py and segments it into excerpts of
approximately 300-500 words.  First, the text is split on double newlines and
excerpts are merged/split to stay within word count bounds.  If any excerpt
exceeds 500 words after the first pass, GPT-4o is used to re-segment it at
grammatically natural boundaries.

Usage:
    python split.py <input.txt> <output.json> <book_name> <author_name>

Requires:
    - OPENAI_API_KEY environment variable set for GPT-based segmentation.
"""

import json
import re
import sys

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()


def _segment_with_gpt(excerpt: str) -> list[str]:
    """Use GPT-4o to split a long excerpt at natural grammatical boundaries."""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": (
                    "Segment it into excerpts of minimum length 300-500 words such that each excerpt "
                    "is grammatical from the start and doesn't feel abruptly cut off. There should be zero "
                    "deletion and break into excerpts at grammatically natural places. Maintain the original "
                    "word count. Avoid breaking into too many small excerpts. Start directly. Don't say "
                    "Here's or Here is ....\n\n" + excerpt
                ),
            }
        ],
    )
    response = completion.choices[0].message.content
    return [seg.strip() for seg in response.split('\n\n') if seg.strip()]


def _segment_by_word_count(text: str, min_words: int = 300, max_words: int = 500) -> list[str]:
    """Split text on double newlines and merge adjacent excerpts to reach target size."""
    excerpts = re.split(r'\n\n+', text)
    segmented = []
    current = ""
    count = 0

    for para in excerpts:
        words = para.split()
        if count + len(words) > max_words:
            if count >= min_words:
                segmented.append(current.strip())
                current = para
                count = len(words)
            else:
                current += " " + para
                count += len(words)
        else:
            current += ("\n\n" if current else "") + para
            count += len(words)

    if current.strip():
        segmented.append(current.strip())

    return segmented


def main(input_file_path: str, output_file_path: str, book_name: str, author_name: str):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # First pass: merge/split by word count heuristics
    initial_excerpts = _segment_by_word_count(text)

    # Second pass: use GPT to re-segment any oversized excerpts
    final_excerpts = []
    for para in tqdm(initial_excerpts, desc="Processing excerpts"):
        word_count = len(para.split())
        if word_count > 500:
            try:
                splits = _segment_with_gpt(para)
                final_excerpts.extend(splits)
            except Exception as e:
                print(f"Segmentation API failed: {e}", file=sys.stderr)
                final_excerpts.append(para)
        else:
            final_excerpts.append(para)

    output = [
        {
            "book_name": book_name,
            "author_name": author_name,
            "excerpt_id": f"p_id{i + 1}",
            "excerpt_text": para,
        }
        for i, para in enumerate(final_excerpts)
    ]

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(output)} excerpts to {output_file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python split.py <input.txt> <output.json> <book_name> <author_name>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
