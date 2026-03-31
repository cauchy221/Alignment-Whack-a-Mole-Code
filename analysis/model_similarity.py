#!/usr/bin/env python3
"""
Cross-model memorization similarity via Jaccard index (Section 5.3).

Computes pairwise Jaccard similarity of BMC coverage masks across models.
For each model's generation file, a boolean coverage mask is built over the
book's word tokens (which words are covered by at least one extracted span
of >= k words, with instruction m-gram trimming).  Pairwise Jaccard
|A ∩ B| / |A ∪ B| measures whether two models memorize the *same* regions.

Also computes:
  - Random baseline: expected Jaccard if models covered words independently.
  - Self-agreement: Jaccard between first-half and second-half generations
    within the same model (upper bound on cross-model agreement).

Usage:
    python analysis/model_similarity.py \\
        --test_book data/example_book.json \\
        --generation_files data/example_gens_gpt.json data/example_gens_gemini.json data/example_gens_deepseek.json \\
        --model_names "GPT-4o" "Gemini-2.5-Pro" "DeepSeek-V3.1"
"""

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tokenization (self-contained, mirrors evaluation/memorization_eval_metrics.py)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]")


def _tok_words(text: str) -> List[str]:
    return [t.lower() for t in wordpunct_tokenize(text or "") if _WORD_RE.search(t)]


def _pid_to_int(ex: dict) -> int:
    s = str(ex.get("excerpt_id", ""))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Book index
# ---------------------------------------------------------------------------

class _BookIndex:
    def __init__(self, words: List[str]):
        self.words = words
        self._cache: Dict[int, Dict[tuple, List[int]]] = {}

    def get_kgram_index(self, k: int) -> Dict[tuple, List[int]]:
        if k not in self._cache:
            idx: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(len(self.words) - k + 1):
                idx[tuple(self.words[i : i + k])].append(i)
            self._cache[k] = idx
        return self._cache[k]


def _build_book_index(book: list) -> _BookIndex:
    exs = sorted(book, key=_pid_to_int)
    all_words: List[str] = []
    for ex in exs:
        all_words.extend(_tok_words(ex.get("excerpt_text", "")))
    return _BookIndex(all_words)


# ---------------------------------------------------------------------------
# Matching and coverage
# ---------------------------------------------------------------------------

def _kset(words: List[str], k: int) -> Set[tuple]:
    if k <= 0:
        return set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _find_matches(gen_words: List[str], book: _BookIndex, k: int) -> List[Tuple[int, int]]:
    bw = book.words
    idx = book.get_kgram_index(k)
    visited: Set[Tuple[int, int]] = set()
    intervals: List[Tuple[int, int]] = []
    for j in range(len(gen_words) - k + 1):
        key = tuple(gen_words[j : j + k])
        starts = idx.get(key)
        if not starts:
            continue
        for i in starts:
            ii, jj = i, j
            while ii > 0 and jj > 0 and bw[ii - 1] == gen_words[jj - 1]:
                ii -= 1
                jj -= 1
            if (ii, jj) in visited:
                continue
            visited.add((ii, jj))
            p = 0
            while (ii + p) < len(bw) and (jj + p) < len(gen_words) and bw[ii + p] == gen_words[jj + p]:
                p += 1
            if p >= k:
                intervals.append((ii, ii + p))
    return intervals


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _subtract(base: Tuple[int, int], removes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    s, e = base
    clamped = [(max(s, a), min(e, b)) for a, b in removes if not (b <= s or a >= e)]
    rm = _merge_intervals([r for r in clamped if r[0] < r[1]])
    if not rm:
        return [base]
    out, cur = [], s
    for a, b in rm:
        if cur < a:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        out.append((cur, e))
    return out


def _trim_instruction(
    gold_words: List[str], instr_words: List[str],
    intervals: List[Tuple[int, int]], k: int, trim_k: int,
) -> List[Tuple[int, int]]:
    trimmed: List[Tuple[int, int]] = []
    for raw in intervals:
        s, e = raw
        span_len = e - s
        if trim_k <= 0 or span_len < trim_k:
            if span_len >= k:
                trimmed.append(raw)
            continue
        instr_k = _kset(instr_words, trim_k)
        removes = []
        for i in range(span_len - trim_k + 1):
            kg = tuple(gold_words[s + i : s + i + trim_k])
            if kg in instr_k:
                removes.append((s + i, s + i + trim_k))
        removes = _merge_intervals(removes)
        for start, end in _subtract(raw, removes):
            if end - start >= k:
                trimmed.append((start, end))
    return _merge_intervals(trimmed)


def _compute_coverage_mask(
    book_index: _BookIndex, examples: list,
    k: int = 5, trim_k: int = 5,
    gen_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Compute boolean coverage mask over the book for a set of generations."""
    n = len(book_index.words)
    covered = np.zeros(n, dtype=bool)
    exs = sorted(examples, key=_pid_to_int)

    for ex in exs:
        instr_words = _tok_words(ex.get("instruction", ""))
        gens = ex.get("generations", []) or []
        indices = gen_indices if gen_indices is not None else range(len(gens))
        for idx in indices:
            if idx >= len(gens):
                continue
            gen_text = gens[idx].get("generated_text", "")
            gen_words = _tok_words(gen_text)
            if len(gen_words) < k:
                continue
            raw = _find_matches(gen_words, book_index, k)
            if not raw:
                continue
            trimmed = _trim_instruction(book_index.words, instr_words, raw, k, trim_k)
            for s, e in trimmed:
                covered[s:e] = True
    return covered


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return float(intersection) / float(union) if union > 0 else 0.0


def _random_expected_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    p_a = np.sum(a) / n
    p_b = np.sum(b) / n
    denom = p_a + p_b - p_a * p_b
    return float(p_a * p_b) / float(denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute pairwise Jaccard similarity of BMC coverage masks across models (Section 5.3)."
    )
    ap.add_argument("--test_book", required=True,
                    help="Path to test book JSON.")
    ap.add_argument("--generation_files", nargs="+", required=True,
                    help="Paths to generation JSON files (one per model).")
    ap.add_argument("--model_names", nargs="+", default=None,
                    help="Display names for each model (defaults to filenames).")
    ap.add_argument("--k", type=int, default=5,
                    help="Minimum match length (default: 5).")
    ap.add_argument("--trim_k", type=int, default=5,
                    help="Instruction m-gram trimming size (default: 5).")
    ap.add_argument("--out", default=None,
                    help="Path to write JSON results.")
    args = ap.parse_args()

    if args.model_names and len(args.model_names) != len(args.generation_files):
        raise SystemExit("Number of --model_names must match --generation_files")

    model_names = args.model_names or [f.rsplit("/", 1)[-1].replace(".json", "") for f in args.generation_files]

    with open(args.test_book, "r", encoding="utf-8") as f:
        book = json.load(f)

    book_index = _build_book_index(book)
    n_words = len(book_index.words)

    # Compute coverage masks
    masks: Dict[str, np.ndarray] = {}
    for name, gen_file in tqdm(list(zip(model_names, args.generation_files)), desc="Computing coverage"):
        with open(gen_file, "r", encoding="utf-8") as f:
            examples = json.load(f)
        masks[name] = _compute_coverage_mask(book_index, examples, k=args.k, trim_k=args.trim_k)

    # Pairwise Jaccard
    print(f"\n{'=' * 60}")
    print(f"Pairwise Jaccard similarity (book: {n_words} words)")
    print(f"{'=' * 60}")

    pair_results = []
    for m_a, m_b in combinations(model_names, 2):
        if m_a in masks and m_b in masks:
            j = _jaccard(masks[m_a], masks[m_b])
            r = _random_expected_jaccard(masks[m_a], masks[m_b])
            pair_results.append({"model_a": m_a, "model_b": m_b, "jaccard": j, "random_baseline": r})
            print(f"  {m_a} vs {m_b}: jaccard={j:.4f}  random_baseline={r:.4f}")

    # Self-agreement (split generations into halves)
    print(f"\n{'=' * 60}")
    print(f"Self-agreement (first-half vs second-half generations)")
    print(f"{'=' * 60}")

    self_results = []
    for name, gen_file in zip(model_names, args.generation_files):
        with open(gen_file, "r", encoding="utf-8") as f:
            examples = json.load(f)
        max_gens = max((len(ex.get("generations", []) or []) for ex in examples), default=0)
        half = max_gens // 2
        if half < 1:
            print(f"  {name}: too few generations for self-agreement")
            continue
        mask_a = _compute_coverage_mask(book_index, examples, k=args.k, trim_k=args.trim_k,
                                        gen_indices=list(range(0, half)))
        mask_b = _compute_coverage_mask(book_index, examples, k=args.k, trim_k=args.trim_k,
                                        gen_indices=list(range(half, max_gens)))
        j = _jaccard(mask_a, mask_b)
        self_results.append({"model": name, "self_jaccard": j})
        print(f"  {name}: {j:.4f}")

    # Per-model coverage
    print(f"\n{'=' * 60}")
    print(f"Per-model BMC@{args.k} coverage")
    print(f"{'=' * 60}")

    coverage_results = []
    for name in model_names:
        if name in masks:
            cov = float(np.sum(masks[name])) / n_words
            coverage_results.append({"model": name, "bmc_score": cov})
            print(f"  {name}: {cov * 100:.2f}%")

    if args.out:
        output = {
            "config": {"test_book": args.test_book, "k": args.k, "trim_k": args.trim_k, "n_words": n_words},
            "pairwise_jaccard": pair_results,
            "self_agreement": self_results,
            "coverage": coverage_results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
