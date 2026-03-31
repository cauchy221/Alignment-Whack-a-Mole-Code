#!/usr/bin/env python3
"""
Cross-excerpt memorization analysis (Section 5.2).

For each memorized span (a contiguous word sequence found in both a model
generation and the book), this script determines:
  - Target excerpt B: the excerpt in the book where the span is located.
  - Source excerpts A: the excerpts whose generations produced this span.
  - Cross-excerpt status: whether any source A differs from target B.

A span is "cross-excerpt" when a model, prompted with the summary of
excerpt A, generates verbatim text that belongs to a *different* excerpt B
in the same book.  The cross-excerpt ratio (fraction of spans with at least
one cross-excerpt source) quantifies how often models retrieve content from
semantically related but distinct regions.

Usage:
    python analysis/cross_excerpt.py \\
        --book data/example_book.json \\
        --runs data/example_gens_gpt.json \\
        --out cross_excerpt_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from nltk.tokenize import wordpunct_tokenize

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]")


def _norm_words(tokens: List[str]) -> List[str]:
    """Lowercase word-only tokens (exclude punctuation-only tokens)."""
    return [t.lower() for t in tokens if _WORD_RE.search(t)]


def _pid_to_int(pid: str) -> int:
    try:
        if pid.startswith("p_id"):
            return int(pid[4:])
        return int(pid)
    except (ValueError, AttributeError):
        return 999999


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = [merged[-1][0], max(merged[-1][1], e)]
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _detokenize(tokens: List[str]) -> str:
    """Reconstruct readable text from tokens."""
    out: List[str] = []
    for token in tokens:
        if out and not _WORD_RE.search(token[0] if token else ""):
            out[-1] += token
        else:
            out.append(token)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Book index
# ---------------------------------------------------------------------------

class BookIndex:
    """Word-level index over the concatenated book, supporting k-gram lookup."""

    def __init__(
        self,
        book_display_tokens: List[str],
        book_word_tokens: List[str],
        para_spans_disp: List[Tuple[int, int, str]],
        para_spans_word: List[Tuple[int, int, str]],
        word_to_disp: List[int],
    ):
        self.book_display_tokens = book_display_tokens
        self.book_word_tokens = book_word_tokens
        self.para_spans_disp = para_spans_disp
        self.para_spans_word = para_spans_word
        self.word_to_disp = word_to_disp
        self.index_cache: Dict[int, Dict[Tuple[str, ...], List[int]]] = {}

        self._word_to_pid: Dict[int, str] = {}
        for start, end, pid in para_spans_word:
            for i in range(start, end):
                self._word_to_pid[i] = pid

    def get_containing_excerpt(self, start: int, end: int) -> Optional[str]:
        """Return excerpt ID if span is strictly within one excerpt, else None."""
        if end <= start:
            return None
        first_pid = self._word_to_pid.get(start)
        if first_pid is None:
            return None
        for i in range(start + 1, end):
            if self._word_to_pid.get(i) != first_pid:
                return None
        return first_pid


def build_book_index(book_examples: List[Dict[str, Any]]) -> BookIndex:
    """Build a global word-level index from sorted book excerpts."""
    exs = sorted(book_examples, key=lambda ex: _pid_to_int(str(ex.get("excerpt_id"))))

    disp_tokens: List[str] = []
    word_tokens: List[str] = []
    para_spans_disp: List[Tuple[int, int, str]] = []
    para_spans_word: List[Tuple[int, int, str]] = []
    word_to_disp: List[int] = []
    disp_offset = 0
    word_offset = 0

    for ex in exs:
        pid = str(ex.get("excerpt_id"))
        text = ex.get("excerpt_text") or ""

        ex_disp = wordpunct_tokenize(text)
        ex_words = _norm_words(ex_disp)

        ex_word_to_disp: List[int] = []
        for disp_idx, tok in enumerate(ex_disp):
            if _WORD_RE.search(tok):
                ex_word_to_disp.append(disp_offset + disp_idx)

        disp_tokens.extend(ex_disp)
        word_tokens.extend(ex_words)
        para_spans_disp.append((disp_offset, disp_offset + len(ex_disp), pid))
        para_spans_word.append((word_offset, word_offset + len(ex_words), pid))
        word_to_disp.extend(ex_word_to_disp)

        disp_offset += len(ex_disp)
        word_offset += len(ex_words)

    return BookIndex(disp_tokens, word_tokens, para_spans_disp, para_spans_word, word_to_disp)


def _get_index_for_k(book_index: BookIndex, k: int) -> Dict[Tuple[str, ...], List[int]]:
    if k in book_index.index_cache:
        return book_index.index_cache[k]
    idx: Dict[Tuple[str, ...], List[int]] = {}
    words = book_index.book_word_tokens
    for i in range(len(words) - k + 1):
        key = tuple(words[i : i + k])
        if key not in idx:
            idx[key] = []
        idx[key].append(i)
    book_index.index_cache[k] = idx
    return idx


def _word_span_to_display_slice(
    book_index: BookIndex, s_word: int, e_word: int
) -> Optional[Tuple[int, int]]:
    if s_word >= len(book_index.word_to_disp) or e_word > len(book_index.word_to_disp):
        return None
    if s_word >= e_word:
        return None
    s_disp = book_index.word_to_disp[s_word]
    e_disp = book_index.word_to_disp[e_word - 1] + 1
    while (
        e_disp < len(book_index.book_display_tokens)
        and not _WORD_RE.search(book_index.book_display_tokens[e_disp])
    ):
        e_disp += 1
    return s_disp, e_disp


# ---------------------------------------------------------------------------
# Generation matching
# ---------------------------------------------------------------------------

def _find_matching_spans(
    book_index: BookIndex,
    gen_words: List[str],
    k_match: int,
) -> List[Tuple[int, int]]:
    """Find all maximal contiguous word matches between a generation and the book."""
    if len(gen_words) < k_match:
        return []
    bw = book_index.book_word_tokens
    idx = _get_index_for_k(book_index, k_match)
    visited: Set[Tuple[int, int]] = set()
    matches: List[Tuple[int, int]] = []

    for gen_pos in range(len(gen_words) - k_match + 1):
        kgram = tuple(gen_words[gen_pos : gen_pos + k_match])
        starts = idx.get(kgram)
        if not starts:
            continue
        for book_pos in starts:
            b, g = book_pos, gen_pos
            while b > 0 and g > 0 and bw[b - 1] == gen_words[g - 1]:
                b -= 1
                g -= 1
            pair = (b, g)
            if pair in visited:
                continue
            visited.add(pair)
            length = 0
            while (
                b + length < len(bw)
                and g + length < len(gen_words)
                and bw[b + length] == gen_words[g + length]
            ):
                length += 1
            if length >= k_match:
                matches.append((b, b + length))
    return matches


def _find_spans_for_generation(
    book_index: BookIndex,
    gen_text: str,
    k_match: int,
    min_length: int,
) -> List[Tuple[int, int]]:
    """Find all book spans matching a generation (no instruction trimming)."""
    gen_disp = wordpunct_tokenize(gen_text)
    gen_words = _norm_words(gen_disp)
    if not gen_words:
        return []
    raw_matches = _find_matching_spans(book_index, gen_words, k_match)
    if not raw_matches:
        return []
    return [iv for iv in raw_matches if (iv[1] - iv[0]) >= min_length]


# ---------------------------------------------------------------------------
# Subset span filtering
# ---------------------------------------------------------------------------

def _filter_subset_spans(
    spans_with_sources: List[Tuple[Tuple[int, int], str, str, Optional[int]]]
) -> List[Tuple[Tuple[int, int], str, str, Optional[int]]]:
    """Remove spans that are strict subsets of any other span."""
    if not spans_with_sources:
        return []
    all_spans = sorted(set(span for span, _, _, _ in spans_with_sources), key=lambda x: (x[0], -x[1]))
    maximal: Set[Tuple[int, int]] = set()
    for span in all_spans:
        start, end = span
        is_subset = any(start >= ks and end <= ke for ks, ke in maximal)
        if not is_subset:
            maximal.add(span)
    return [(span, src, gen, gn) for span, src, gen, gn in spans_with_sources if span in maximal]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_cross_excerpt(
    book_index: BookIndex,
    runs: List[Dict[str, Any]],
    k_match: int,
    min_length: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Analyze cross-excerpt memorization across all generations.

    For each span found in any generation, determines which excerpts'
    generations produced it (sources) and which excerpt contains it (target).

    Returns a dict mapping (start, end) spans to their analysis results.
    """
    spans_with_sources: List[Tuple[Tuple[int, int], str, str, Optional[int]]] = []
    exs = sorted(runs, key=lambda ex: _pid_to_int(str(ex.get("excerpt_id"))))

    for ex in exs:
        pid_A = str(ex.get("excerpt_id", "?"))
        for idx, gen in enumerate(ex.get("generations", []) or []):
            gen_text = gen.get("generated_text") or gen.get("generation") or ""
            spans = _find_spans_for_generation(book_index, gen_text, k_match, min_length)
            for span in spans:
                spans_with_sources.append((span, pid_A, gen_text, idx))

    filtered = _filter_subset_spans(spans_with_sources)

    span_sources: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for span, source, gen_text, gen_num in filtered:
        if source not in span_sources[span]:
            span_sources[span][source] = {"generation": gen_text, "gen_num": gen_num}

    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for span, sources_dict in span_sources.items():
        start, end = span
        target_B = book_index.get_containing_excerpt(start, end)
        result[span] = {"sources": sources_dict, "target_B": target_B}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze cross-excerpt memorization in model generations (Section 5.2)."
    )
    ap.add_argument("--book", required=True,
                    help="Path to test book JSON.")
    ap.add_argument("--runs", required=True,
                    help="Path to generations JSON.")
    ap.add_argument("--match-gram", type=int, default=5,
                    help="Minimum match length k (default: 5).")
    ap.add_argument("--min-span", type=int, default=5,
                    help="Minimum span length to keep (default: 5).")
    ap.add_argument("--length-gt", type=int, default=None,
                    help="Only analyze spans with word length > this value.")
    ap.add_argument("--max-spans-out", type=int, default=100,
                    help="Max spans in output JSON (default: 100, -1 for no limit).")
    ap.add_argument("--max-other-pids", type=int, default=5,
                    help="Max cross-excerpt sources per span (default: 5, -1 for no limit).")
    ap.add_argument("--include-noncross", action="store_true",
                    help="Include spans with no cross-excerpt sources in output.")
    ap.add_argument("--out", default="cross_excerpt_report.json",
                    help="Output JSON path (default: cross_excerpt_report.json).")
    args = ap.parse_args()

    with open(args.book, "r", encoding="utf-8") as f:
        book = json.load(f)
    with open(args.runs, "r", encoding="utf-8") as f:
        runs = json.load(f)

    book_index = build_book_index(book)

    # Build excerpt text/instruction lookups
    pid_to_text = {str(ex.get("excerpt_id", "?")): ex.get("excerpt_text", "") for ex in book}
    pid_to_instr = {str(ex.get("excerpt_id", "?")): ex.get("instruction", "") for ex in runs}

    span_data = analyze_cross_excerpt(
        book_index=book_index,
        runs=runs,
        k_match=args.match_gram,
        min_length=args.min_span,
    )

    # Build span rows
    span_rows: List[Dict[str, Any]] = []
    for (start, end), data in span_data.items():
        sources_dict = data["sources"]
        target_B = data["target_B"]
        if target_B is None:
            continue

        sources = set(sources_dict.keys())
        is_cross = any(A != target_B for A in sources)
        other_sources = {A for A in sources if A != target_B}

        disp_slice = _word_span_to_display_slice(book_index, start, end)
        if disp_slice is None:
            continue
        s_disp, e_disp = disp_slice
        span_text = _detokenize(book_index.book_display_tokens[s_disp:e_disp])

        other_details = []
        for A in sorted(other_sources, key=_pid_to_int):
            meta = sources_dict.get(A, {})
            other_details.append({
                "A_pid": A,
                "A_excerpt_text": pid_to_text.get(A, ""),
                "A_instruction": pid_to_instr.get(A, ""),
                "A_generation": meta.get("generation", ""),
                "A_generation_num": meta.get("gen_num"),
            })
        if args.max_other_pids != -1:
            other_details = other_details[: args.max_other_pids]

        span_rows.append({
            "span_start": start,
            "span_end": end,
            "span_length": end - start,
            "span_text": span_text,
            "target_B": target_B,
            "B_excerpt_text": pid_to_text.get(target_B, ""),
            "B_instruction": pid_to_instr.get(target_B, ""),
            "is_cross": is_cross,
            "num_sources": len(sources),
            "num_other_sources": len(other_sources),
            "all_sources": sorted(sources, key=_pid_to_int),
            "other_sources": sorted(other_sources, key=_pid_to_int),
            "other_source_details": other_details,
        })

    span_rows.sort(key=lambda r: (r["span_length"], r["num_other_sources"]), reverse=True)

    # Apply filters
    analysis_rows = span_rows
    if args.length_gt is not None:
        analysis_rows = [r for r in analysis_rows if r["span_length"] > args.length_gt]

    total_spans = len(analysis_rows)
    cross_spans = sum(1 for r in analysis_rows if r["is_cross"])
    cross_ratio = cross_spans / total_spans if total_spans else 0.0

    if args.include_noncross:
        out_rows = list(analysis_rows)
    else:
        out_rows = [r for r in analysis_rows if r["is_cross"]]

    if args.max_spans_out != -1:
        out_rows = out_rows[: args.max_spans_out]

    summary = {
        "book": args.book,
        "runs": args.runs,
        "match_gram": args.match_gram,
        "min_span": args.min_span,
        "filter_length_gt": args.length_gt,
        "total_spans": total_spans,
        "cross_spans": cross_spans,
        "cross_ratio": cross_ratio,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_obj = {"summary": summary, "spans": out_rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
