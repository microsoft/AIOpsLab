#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
from pathlib import Path
import os

# -------------------------------
# Timestamp detection
# -------------------------------
DEFAULT_TIMESTAMP_REGEX = (
    r"(?:"
    # ISO-like: 2025-09-24 18:41:09 or 2025-09-24T18:41:09Z
    r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
    r"|"
    # Abbreviated month: 2025-Sep-24 18:41:09.830218
    r"\d{4}-[A-Z][a-z]{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"|"
    # Time only: 18:41:09
    r"\b\d{2}:\d{2}:\d{2}\b"
    r"|"
    # Durations: 10m5s, 30s
    r"\b\d+m(?:\d+s)?\b"
    r"|"
    r"\b\d+s\b"
    r"|"
    # syslog-like: Wed Sep 24 18:41:09 2025
    r"[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}"
    r")"
)

DEFAULT_TS_RX = re.compile(DEFAULT_TIMESTAMP_REGEX)

def find_timestamp_spans(line: str, ts_rx: re.Pattern[str]) -> list[tuple[int, int]]:
    return [m.span() for m in ts_rx.finditer(line)]

def make_blocks(lines: list[str], block_size: int) -> list[str]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return ["\n".join(lines[i:i+block_size]) for i in range(0, len(lines), block_size)]

# -------------------------------
# Greedy compressor (single pass)
# -------------------------------
def greedy_compress_pass(
    lines: list[str], 
    ts_rx: re.Pattern[str], 
    block_size: int
) -> list[str]:
    """Run greedy timestamp dedup for a single block size."""
    if not lines:
        return []

    blocks = make_blocks(lines, block_size)
    result: list[str] = [blocks[0]]
    prev_spans: list[tuple[int, int]] | None = find_timestamp_spans(blocks[0], ts_rx)

    for block in blocks[1:]:
        spans = find_timestamp_spans(block, ts_rx)

        if not prev_spans or not spans:
            result.append(block)
            prev_spans = spans
            continue

        if len(spans) != len(prev_spans):
            result.append(block)
            prev_spans = spans
            continue

        if [s[0] for s in spans] != [s[0] for s in prev_spans]:
            result.append(block)
            prev_spans = spans
            continue

        def mask_timestamps(text: str, spans: list[tuple[int, int]]) -> str:
            parts: list[str] = []
            last_end = 0
            for start, end in spans:
                parts.append(text[last_end:start])
                # Replace timestamp with spaces of equal length
                parts.append(" " * (end - start))
                last_end = end
            parts.append(text[last_end:])
            return "".join(parts)

        prev_masked = mask_timestamps(result[-1], prev_spans)
        curr_masked = mask_timestamps(block, spans)

        if prev_masked == curr_masked:
            # Replace last block if only timestamps differ
            result[-1] = block
        else:
            # print(f"prev: {prev_masked}\ncurr: {curr_masked}")
            result.append(block)

        prev_spans = spans

    return result

# -------------------------------
# Multi-pass driver
# -------------------------------
def greedy_compress_lines(
    raw_str: str, 
    ts_rx: re.Pattern[str] = DEFAULT_TS_RX, 
) -> str:
    """
    Run greedy dedup with passes from block_size=1 up to block_size=window_size.
    window_size = LOG_TRIM if LOG_TRIM is int or trimming is disabled  
    """
    log_trim = None
    try:
        value = os.environ.get("LOG_TRIM")
        log_trim = int(value) if value is not None else None
    except ValueError:
        log_trim = None
    if not log_trim or log_trim <= 0:
        return raw_str
    window_size = log_trim
    lines = raw_str.splitlines()
    result = lines[:]
    for size in range(1, window_size + 1):
        result = greedy_compress_pass(result, ts_rx, size)
    return "\n".join(result)

# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Greedy log deduplicator with multi-pass window size support."
    )
    ap.add_argument("input", help="Path to input text file")
    ap.add_argument("output", help="Path to output text file")
    ap.add_argument(
        "--timestamp-regex",
        default=DEFAULT_TIMESTAMP_REGEX,
        help="Regex for timestamps. Default matches ISO, RFC, and k8s durations.",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Maximum block size for multi-pass deduplication. Default: 2",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = in_path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text

    try:
        ts_rx = re.compile(args.timestamp_regex)
    except re.error as e:
        print(f"Invalid timestamp regex: {e}", file=sys.stderr)
        sys.exit(1)

    deduped = greedy_compress_lines(lines, ts_rx, args.window_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(deduped, encoding="utf-8")

if __name__ == "__main__":
    main()

