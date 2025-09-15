# test_summary_from_source.py

import os
from pathlib import Path

from summary import (
    create_study_summary,            # one-shot (batch-backed)
    summarize_chunks_batch,          # batched map over chunks
    merge_summaries_batch_tree,      # batched tree-reduce
    write_summary,
)
from chunking import semantic_chunks
from config import (
    SUMMARY_CHUNK_MAX_CHARS,
    SUMMARY_CHUNK_OVERLAP,
    SUMMARY_REDUCE_FANIN,
)

SRC = Path("data/source.txt")
OUT = Path("data/test_summary.json")

def main():
    if not SRC.exists():
        print(f"[Test] Missing source file: {SRC.resolve()}")
        return

    text = SRC.read_text(encoding="utf-8")
    print(f"[Test] Loaded source ({len(text)} chars)")

    # Choose mode via env; default to batch map→reduce for cost/reliability
    mode = (os.getenv("SUMMARY_MODE") or "mapreduce").strip().lower()
    if mode not in {"mapreduce", "one-shot", "oneshot"}:
        print(f"[Test] Unknown SUMMARY_MODE='{mode}', defaulting to 'mapreduce'")
        mode = "mapreduce"

    try:
        if mode == "one-shot" or mode == "oneshot":
            print("[Test] Building summary (one-shot via batch)...")
            summary = create_study_summary(text)
        else:
            # MAP
            print(f"[Test] Chunking for MAP (max_chars={SUMMARY_CHUNK_MAX_CHARS}, overlap={SUMMARY_CHUNK_OVERLAP})...")
            chunks = semantic_chunks(text, max_chars=SUMMARY_CHUNK_MAX_CHARS, soft_min=max(600, SUMMARY_CHUNK_OVERLAP))
            print(f"[Test] Created {len(chunks)} chunk(s). Summarizing via batch...")
            micros = summarize_chunks_batch(chunks)
            ok = sum(1 for m in micros if isinstance(m, dict) and m)
            print(f"[Test] MAP complete: {ok}/{len(micros)} micro-summaries parsed.")

            # TREE REDUCE
            fanin = int(os.getenv("SUMMARY_REDUCE_FANIN", SUMMARY_REDUCE_FANIN))
            print(f"[Test] REDUCE (tree) with fanin={fanin} via batch...")
            summary = merge_summaries_batch_tree(micros, fanin=fanin)

        ok, msg = write_summary(summary, OUT)
        print(f"[Test] {msg}")
    except Exception as e:
        print(f"[Test] Failed to build summary: {e}")

if __name__ == "__main__":
    main()