#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path

# -------------------------------
# Levenshtein edit distance (O(mn), 2-row DP)
# -------------------------------
def edit_distance(a: str, b: str):
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        curr = [j]
        for i, ca in enumerate(a, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[i] + 1,        # deletion
                curr[i-1] + 1,      # insertion
                prev[i-1] + cost    # substitution
            ))
        prev = curr
    return prev[-1]

# -------------------------------
# Timestamp detection
# -------------------------------
DEFAULT_TIMESTAMP_REGEX = (
    r"(?:"
    r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)"
    r"|"
    r"\b\d{2}:\d{2}:\d{2}\b"
    r"|"
    r"\b\d+m(?:\d+s)?\b"
    r"|"
    r"\b\d+s\b"
    r")"
)

try:
    LOG_DEDUP_EDIT_DISTANCE: int = int(os.environ.get("LOG_DEDUP", "-1"))
except ValueError:
    LOG_DEDUP_EDIT_DISTANCE: int = -1

def find_timestamp_span(line, rx):
    m = rx.search(line)
    return (m.start(), m.end()) if m else None

# -------------------------------
# PASS 1: line-level grouping
# -------------------------------
def group_consecutive_lines(lines, threshold):
    if not lines:
        return []
    groups = [[lines[0]]]
    for line in lines[1:]:
        d = edit_distance(line, groups[-1][-1])
        if d < threshold:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups

def compress_line_group(group, ts_rx):
    # exact duplicates -> keep one
    if len(set(group)) == 1:
        return [group[0]]

    # require a timestamp in every line at the same start index
    spans = []
    for ln in group:
        s = find_timestamp_span(ln, ts_rx)
        if s is None:
            return group[:]  # unchanged
        spans.append(s)

    starts = {s[0] for s in spans}
    if len(starts) == 1:
        # keep only first and last line
        return [group[0], group[-1]]

    return group[:]

def pass1(lines, threshold, ts_rx):
    groups = group_consecutive_lines(lines, threshold)
    out = []
    for g in groups:
        out.extend(compress_line_group(g, ts_rx))
    return out, len(groups)

# -------------------------------
# PASS 2: block-of-2-lines grouping
# -------------------------------
def make_blocks(lines):
    """Return list of blocks; each block is a list of 1 or 2 lines (last may be size 1)."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if i + 1 < n:
            blocks.append([lines[i], lines[i+1]])
            i += 2
        else:
            blocks.append([lines[i]])  # tail singleton
            i += 1
    return blocks

def block_text(block):
    """Join block lines with newline for distance calc."""
    return "\n".join(block)

def group_consecutive_blocks(blocks, threshold):
    if not blocks:
        return []
    groups = [[blocks[0]]]
    for blk in blocks[1:]:
        d = edit_distance(block_text(blk), block_text(groups[-1][-1]))
        if d < threshold:
            groups[-1].append(blk)
        else:
            groups.append([blk])
    return groups

def blocks_all_identical(block_group):
    first = block_text(block_group[0])
    for blk in block_group[1:]:
        if block_text(blk) != first:
            return False
    return True

def compress_block_group(block_group, ts_rx):
    # If all blocks identical -> keep first block only
    if blocks_all_identical(block_group):
        return [block_group[0]]

    # For timestamp-at-same-place rule, require that every block has EXACTLY 2 lines
    # and that, for each line index (0 and 1), the timestamp start index is the same across all blocks.
    for blk in block_group:
        if len(blk) != 2:
            return [b for b in block_group]  # unchanged if any singleton present

    # Collect (start index) for each line position
    pos0 = []
    pos1 = []
    for blk in block_group:
        s0 = find_timestamp_span(blk[0], ts_rx)
        s1 = find_timestamp_span(blk[1], ts_rx)
        if s0 is None or s1 is None:
            return [b for b in block_group]  # unchanged
        pos0.append(s0[0])
        pos1.append(s1[0])

    if len(set(pos0)) == 1 and len(set(pos1)) == 1:
        # keep only first and last block
        return [block_group[0], block_group[-1]]

    return [b for b in block_group]

def flatten_blocks(blocks):
    out = []
    for blk in blocks:
        out.extend(blk)
    return out

def pass2(lines_after_pass1, threshold, ts_rx):
    blocks = make_blocks(lines_after_pass1)
    groups = group_consecutive_blocks(blocks, threshold)
    kept_blocks = []
    for g in groups:
        kept_blocks.extend(compress_block_group(g, ts_rx))
    return flatten_blocks(kept_blocks), len(groups)

def dedup(text: str, threshold: int = LOG_DEDUP_EDIT_DISTANCE, timestamp_regex: str = DEFAULT_TIMESTAMP_REGEX) -> str:
    """
    Deduplicate logs using two-pass grouping.
    In the first pass, each block is one line. If multiple consecutive blocks are exactly the same, only one block is kept; if multiple consecutive blocks are within the threshold edit distance, only the first one and the last one are kept.
    In the second pass, each block is 2 lines.

    Args:
        text (str): Raw input log text.
        threshold (int): Edit distance threshold for grouping. If it is -1, the original text string is returned.
        timestamp_regex (str, optional): Regex for timestamps.

    Returns:
        str: Deduplicated log text.
    """
    if threshold == -1:
        return text
    lines = text.splitlines()

    ts_rx = None
    if timestamp_regex:
        try:
            ts_rx = re.compile(timestamp_regex)
        except re.error as e:
            print(f"Invalid timestamp regex: {e}", file=sys.stderr)
            sys.exit(1)

    # PASS 1
    pass1_out, groups1 = pass1(lines, threshold, ts_rx)

    # PASS 2
    pass2_out, groups2 = pass2(pass1_out, threshold, ts_rx)

    # Debug info to stderr (optional)
    print(
        f"Pass1 groups: {groups1} | After Pass1 lines: {len(pass1_out)} | "
        f"Pass2 groups: {groups2} | Final lines: {len(pass2_out)}",
        file=sys.stderr,
    )

    return "\n".join(pass2_out)


def main():
    ap = argparse.ArgumentParser(
        description="Two-pass log compressor: Pass1 groups by line; Pass2 groups every 2 lines."
    )
    ap.add_argument("input", help="Path to input text file")
    ap.add_argument("output", help="Path to output text file")
    ap.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Edit distance threshold for grouping (applies to both passes). Default: 8",
    )
    ap.add_argument(
        "--timestamp-regex",
        default=DEFAULT_TIMESTAMP_REGEX,
        help="Regex for timestamps; first match per line used. Default covers ISO/RFC and k8s durations.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = in_path.read_text(encoding="utf-8", errors="replace")

    deduped = dedup(raw_text, args.threshold, args.timestamp_regex)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(deduped, encoding="utf-8")

if __name__ == "__main__":
    main()
