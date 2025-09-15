# main.py

import json
from pathlib import Path
import shutil
import argparse
import re
from typing import List

from chunking import read_text, chunk_preview
from topic_map import save_topicmap_batch_input
from build_batch import build_questions_requests_balanced
from run_batch import submit_batch, wait_for_batch, download_output_or_error, cancel_batch
from assemble import parse_batch_output, extract_questions, validate_and_fix, write_final
from taxonomy import build_taxonomy, write_taxonomy
from config import (
    MAX_CHARS, OVERLAP,
    BATCH_QUESTIONS_SHARD_SIZE,
    BATCH_QUESTIONS_MAX_BYTES,
    CANCEL_ON_TIMEOUT,   # used for Ctrl-C behavior
    SUMMARY_REDUCE_FANIN,
)
from summary import summarize_chunks_batch, merge_summaries_batch_tree, write_summary

DATA = Path("data")
DATA.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "source.txt"
TOPIC_IN = DATA / "topicmap_input.jsonl"
TOPIC_OUT = DATA / "topicmap_output.jsonl"
Q_OUT = DATA / "questions_output.jsonl"
FINAL = DATA / "questions_final.json"
TAXONOMY = DATA / "taxonomy.json"
STUDY_SUMMARY = DATA / "study_summary.json"

# ---------- Redaction helpers ----------
_ID_KEYS = ("id", "request_id", "output_file_id", "error_file_id", "custom_id")

def _obf(s: str) -> str:
    """Obfuscate tokens for logs."""
    if not isinstance(s, str) or not s:
        return str(s)
    return "[redacted]"

def _redact_jsonlike_text(s: str) -> str:
    """
    Redact common ID-bearing fields in a JSONL preview string before printing.
    """
    if not isinstance(s, str) or not s:
        return s
    out = s
    # Generic key-based redaction: "key": "VALUE"
    for k in _ID_KEYS:
        out = re.sub(rf'("{k}"\s*:\s*")([^"]+)(")', rf'\1[redacted]\3', out)
    # Also redact obvious ID-like tokens: batch_*, resp_*, msg_*, rs_*, file-*, etc.
    out = re.sub(r'\b(?:batch|resp|msg|rs|file|ft|run|job)_[A-Za-z0-9\-\._]+\b', '[redacted]', out)
    return out
# --------------------------------------

def clean_data_dir():
    kept = {"source.txt", ".gitkeep", ".DS_Store"}
    removed = 0
    for f in DATA.iterdir():
        if f.name in kept:
            continue
        try:
            if f.is_symlink():
                f.unlink(missing_ok=True)
            elif f.is_file():
                f.unlink(missing_ok=True)
            elif f.is_dir():
                shutil.rmtree(f)
            removed += 1
        except Exception as e:
            print(f"[Cleanup] Could not remove {f}: {e}")
    print(f"[Cleanup] Removed {removed} item(s).")

def _print_file_head(path: Path, lines: int = 60):
    try:
        with open(path, "r", encoding="utf-8") as f:
            print("----- Begin file preview -----")
            for i, line in enumerate(f):
                if i >= lines:
                    print("... (truncated)")
                    break
                print(_redact_jsonlike_text(line.rstrip()))
            print("----- End file preview -----")
    except Exception as e:
        print(f"Could not preview file {path}: {e}")

def _inspect_output_jsonl(path: Path):
    if not path.exists():
        print(f"[inspect] File not found: {path}")
        return
    print(f"[inspect] Inspecting JSONL: {path} (size={path.stat().st_size} bytes)")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"[inspect] Line #{i} JSON error: {e}")
                    print(f"[inspect] Line #{i} preview: {_redact_jsonlike_text(line[:200])}...")
                    continue

                rid = obj.get("id") or obj.get("custom_id")
                resp = obj.get("response", {}) or {}
                status_code = resp.get("status_code") or resp.get("status")
                err = obj.get("error")

                # Drill into body for outputs (this matches how parse_batch_output reads it)
                body = resp.get("body", {}) if isinstance(resp, dict) else {}
                has_output_text = bool(body.get("output_text"))
                body_output = body.get("output", []) if isinstance(body, dict) else []
                has_output_list = isinstance(body_output, list)

                kinds = []
                if isinstance(body_output, list):
                    for it in body_output:
                        if isinstance(it, dict):
                            kinds.append(it.get("type"))
                        else:
                            kinds.append(type(it).__name__)

                print(
                    f"[inspect] Line #{i}: id={_obf(str(rid))}, status_code={status_code}, "
                    f"has_output_text={has_output_text}, has_output_list={has_output_list}, "
                    f"error_present={bool(err)}, body.output.types={kinds}"
                )
    except Exception as e:
        print(f"[inspect] Failed to inspect {path}: {e}")

def _audit_topicmap_coverage(topic_map: dict, total_chunks: int):
    """Return (missing_indices, overlap_indices)."""
    seen_counts = {}
    for unit in topic_map.get("units", []):
        for t in unit.get("topics", []):
            s, e = t["chunk_span"]
            for i in range(s, e + 1):
                seen_counts[i] = seen_counts.get(i, 0) + 1
    missing = [i for i in range(total_chunks) if i not in seen_counts]
    overlaps = [i for i, c in seen_counts.items() if c > 1]
    return missing, overlaps

# -------- Optional guardrail: ensure each shard has unique custom_ids --------
def _assert_unique_custom_ids(batch_requests: list):
    seen = {}
    for i, r in enumerate(batch_requests, start=1):
        cid = r.get("custom_id")
        if not cid:
            raise ValueError(f"Request #{i} missing custom_id")
        if cid in seen:
            raise ValueError(f"Duplicate custom_id '{cid}' between requests #{seen[cid]} and #{i}")
        seen[cid] = i
# -----------------------------------------------------------------------------

def run_topicmap_batch(chunks):
    preview = chunk_preview(chunks)  # uses first lines per chunk (good for indexing + titles)
    print(f"[TopicMap] Building request JSONL at {TOPIC_IN} with {len(preview)} preview entries.")
    save_topicmap_batch_input(preview, TOPIC_IN)

    batch_id = submit_batch(TOPIC_IN)
    print(f"[TopicMap] Submitted batch id={_obf(batch_id)}. Waiting for completion...")

    try:
        st = wait_for_batch(batch_id)
    except KeyboardInterrupt:
        if CANCEL_ON_TIMEOUT:
            cancel_batch(batch_id)
        print("[TopicMap] Interrupted by user.")
        raise

    print(f"[TopicMap] Batch status: {st.status}")

    if st.status != "completed":
        kind, path = download_output_or_error(st, TOPIC_OUT)
        if kind == "error":
            print(f"[TopicMap ERROR] Error JSONL saved to {path}")
            _print_file_head(path, 60)
            _inspect_output_jsonl(path)
        raise RuntimeError(f"Topic map batch status: {st.status}")

    kind, path = download_output_or_error(st, TOPIC_OUT)
    if kind == "error":
        print(f"[TopicMap ERROR] Error JSONL saved to {path}")
        _print_file_head(path, 60)
        _inspect_output_jsonl(path)
        raise RuntimeError("Topic map batch returned errors (no output).")

    print(f"[TopicMap] Output JSONL saved to {path}")
    _inspect_output_jsonl(path)
    print("[TopicMap] Raw output head:")
    _print_file_head(path, 40)

    payloads = parse_batch_output(TOPIC_OUT)
    if not payloads:
        print("[TopicMap] No parsed payloads from output JSONL.")
        print("[TopicMap] Showing raw file again for diagnosis:")
        _print_file_head(TOPIC_OUT.with_suffix(".errors.jsonl"), 80)
        raise RuntimeError("No topic map output parsed.")

    topic_map = payloads[0]
    if not isinstance(topic_map, dict) or "units" not in topic_map:
        print("[TopicMap] Parsed payload lacks 'units'. Full parsed object follows:")
        j = json.dumps(topic_map, ensure_ascii=False)
        print(_redact_jsonlike_text(j[:1200]) + ("..." if len(j) > 1200 else ""))
        raise RuntimeError("Topic map payload is not a valid object with 'units'.")

    # ---- Coverage check BEFORE saving, per request ----
    missing, overlaps = _audit_topicmap_coverage(topic_map, total_chunks=len(chunks))
    if missing or overlaps:
        print(f"[TopicMap] Coverage violation: missing={missing} overlaps={overlaps}")
        raise RuntimeError("Topic map violates coverage constraints (no gaps/no overlaps).")

    (DATA / "topicmap.json").write_text(
        json.dumps(topic_map, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("[TopicMap] Saved parsed topic map → data/topicmap.json")
    # ---------------------------------------------------

    print("[TopicMap] Parsed topic map successfully.")
    return topic_map

def _shard_requests(requests: list, max_per_shard: int, max_bytes: int) -> List[List[dict]]:
    if max_per_shard <= 0:
        return [requests]
    shards: List[List[dict]] = []
    cur: List[dict] = []
    cur_bytes = 0
    for req in requests:
        s = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        if (len(cur) >= max_per_shard) or (max_bytes and cur_bytes + len(s) > max_bytes):
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(req)
        cur_bytes += len(s)
    if cur:
        shards.append(cur)
    return shards

def _extract_question_type(req):
    """
    Best-effort extractor that tolerates dict/list 'body' shapes without raising.
    Prefers parsing from custom_id (e.g., "..._mcq_01") for reliability, then
    falls back to digging through the request body. Returns 'mcq'|'msq'|'tf' or None.
    """
    try:
        # --- Fast path: infer from custom_id like "q_u1_topic_mcq_01"
        cid = (req.get("custom_id") or "").lower()
        for t in ("mcq", "msq", "tf"):
            if f"_{t}_" in cid or cid.endswith(f"_{t}") or cid.startswith(f"{t}_"):
                return t

        # --- Fallbacks: inspect body shapes
        b = req.get("body")

        # Common case: body is a dict with 'input'
        if isinstance(b, dict):
            inp = b.get("input")
            if isinstance(inp, dict):
                qt = inp.get("question_type")
                if isinstance(qt, str):
                    qt = qt.lower().strip()
                    return qt if qt in {"mcq","msq","tf"} else None

            # Sometimes the input is inside a messages-like array
            msgs = b.get("messages") or b.get("input_messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        inp2 = m.get("input")
                        if isinstance(inp2, dict):
                            qt = inp2.get("question_type")
                            if isinstance(qt, str):
                                qt = qt.lower().strip()
                                return qt if qt in {"mcq","msq","tf"} else None

        # Alternate case: body is a list of items, each may carry 'input'
        elif isinstance(b, list):
            for item in b:
                if isinstance(item, dict):
                    inp = item.get("input")
                    if isinstance(inp, dict):
                        qt = inp.get("question_type")
                        if isinstance(qt, str):
                            qt = qt.lower().strip()
                            return qt if qt in {"mcq","msq","tf"} else None

        # Fallback: occasionally 'input' may be top-level
        inp_top = req.get("input")
        if isinstance(inp_top, dict):
            qt = inp_top.get("question_type")
            if isinstance(qt, str):
                qt = qt.lower().strip()
                return qt if qt in {"mcq","msq","tf"} else None

    except Exception:
        pass

    return None

def run_questions_batch(topic_map, chunks):
    # Plan all requests (not written as a single giant JSONL anymore)
    reqs = build_questions_requests_balanced(topic_map, chunks)
    print(f"[Questions] Planned {len(reqs)} requests")

    # --- Summarize question mix (safe) ---
    from collections import Counter
    qtypes = [qt for qt in (_extract_question_type(r) for r in reqs) if qt]
    mix = Counter(qtypes)
    unknown = len(reqs) - len(qtypes)
    print(f"[Questions] Planned by type: {dict(sorted(mix.items()))}  (total={len(reqs)}, unknown_shapes={unknown})")

    # Sanity check: topic count
    expected_per_topic = len([t for u in topic_map.get("units", []) for t in u.get("topics", [])])
    print(f"[Questions] Topics detected: {expected_per_topic}")

    from config import QUESTION_TYPES_PER_TOPIC, QUESTION_TYPE_MULTIPLIER
    expected_total = expected_per_topic * len(QUESTION_TYPES_PER_TOPIC) * QUESTION_TYPE_MULTIPLIER
    print(f"[Questions] Expected requests (topics × types × multiplier): "
          f"{expected_per_topic} × {len(QUESTION_TYPES_PER_TOPIC)} × {QUESTION_TYPE_MULTIPLIER} = {expected_total}")
    if expected_total != len(reqs):
        print(f"[Questions][warn] Planned count {len(reqs)} ≠ expected {expected_total}")

    shards = _shard_requests(reqs, BATCH_QUESTIONS_SHARD_SIZE, BATCH_QUESTIONS_MAX_BYTES)
    print(f"[Questions] Submitting in {len(shards)} shard(s) "
          f"(max_per_shard={BATCH_QUESTIONS_SHARD_SIZE}, byte_cap={BATCH_QUESTIONS_MAX_BYTES})")

    # Clean/prepare combined output file
    if Q_OUT.exists():
        Q_OUT.unlink()
    Q_OUT.touch()

    total = len(reqs)
    parsed_total = 0

    for si, shard in enumerate(shards, start=1):
        shard_in = DATA / f"questions_input.shard{si:02d}.jsonl"
        shard_out = DATA / f"questions_output.shard{si:02d}.jsonl"

        # Optional guardrail: assert unique IDs inside this shard before we submit it
        _assert_unique_custom_ids(shard)

        print(f"[Questions][Shard {si}/{len(shards)}] Writing {len(shard)} requests → {shard_in}")

        with shard_in.open("w", encoding="utf-8") as f:
            for line in shard:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        batch_id = submit_batch(shard_in)
        print(f"[Questions][Shard {si}] Submitted batch id={_obf(batch_id)}. Waiting for completion...")

        try:
            st = wait_for_batch(batch_id)
        except KeyboardInterrupt:
            if CANCEL_ON_TIMEOUT:
                cancel_batch(batch_id)
            print(f"[Questions][Shard {si}] Interrupted by user.")
            raise

        print(f"[Questions][Shard {si}] Batch status: {st.status}")

        if st.status != "completed":
            try:
                kind, path = download_output_or_error(st, shard_out)
                if kind == "error":
                    print(f"[Questions ERROR][Shard {si}] Error JSONL saved to {path}")
                    _print_file_head(path, 60)
                    _inspect_output_jsonl(path)
                    _summarize_error_jsonl(path)
            except Exception as e:
                print(f"[Questions][Shard {si}] Failed without output/error file: {e}")
            raise RuntimeError(f"Questions batch shard {si} status: {st.status}")

        kind, path = download_output_or_error(st, shard_out)
        if kind == "error":
            print(f"[Questions ERROR][Shard {si}] Error JSONL saved to {path}")
            _print_file_head(path, 60)
            _inspect_output_jsonl(path)
            _summarize_error_jsonl(path)
            raise RuntimeError(f"Questions batch shard {si} returned errors (no output).")

        print(f"[Questions][Shard {si}] Output JSONL saved to {path}")
        _inspect_output_jsonl(path)

        # Append shard outputs to combined Q_OUT
        parsed_shard = 0
        with Q_OUT.open("a", encoding="utf-8") as outf, shard_out.open("r", encoding="utf-8") as inf:
            for line in inf:
                outf.write(line)
                parsed_shard += 1

        parsed_total += parsed_shard
        pct = (parsed_total / total * 100.0) if total else 100.0
        print(f"[Questions][Shard {si}] Accumulated parsed: {parsed_total}/{total} ({pct:.1f}%)")

    print(f"[Questions] All shards completed. Combined output → {Q_OUT}")

    # Parse + validate
    payloads = parse_batch_output(Q_OUT)
    if not payloads:
        print("[Questions] No parsed payloads from combined output JSONL.")
        raise RuntimeError("No questions output parsed.")

    items = extract_questions(payloads)
    fixed = validate_and_fix(items)
    write_final(fixed, FINAL)
    print(f"[Questions] Final questions: {len(fixed)} → {FINAL}")

    # --- QA summary (Counter version) ---
    from collections import Counter
    print("[QA] By type:", dict(Counter(q["type"] for q in fixed)))
    print("[QA] Units covered:", sorted({q["unit_id"] for q in fixed}))
    print("[QA] Topics covered:", len({q["topic_id"] for q in fixed}), "of", sum(len(u["topics"]) for u in topic_map["units"]))
    # ------------------------------------

    # Build taxonomy.json using topic_map + generated questions
    tax = build_taxonomy(topic_map, fixed)
    write_taxonomy(tax, TAXONOMY)
    print(f"[Taxonomy] Wrote taxonomy → {TAXONOMY}")

def _summarize_error_jsonl(err_path: Path, max_lines: int = 10):
    if not err_path.exists():
        print(f"[errors] (no file) {err_path}")
        return
    print(f"[errors] Summary for {err_path}:")
    counts = {}
    samples = []
    with err_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = _obf(str(obj.get("custom_id")))
            body = ((obj.get("response") or {}).get("body") or {})
            err = (body.get("error") or {})
            msg = _redact_jsonlike_text(err.get("message") or "")
            code = err.get("code") or ""
            param = err.get("param") or ""
            key = (code, param, msg)
            counts[key] = counts.get(key, 0) + 1
            if len(samples) < max_lines:
                samples.append((rid, code, param, msg))
    # Print a compact histogram
    for (code, param, msg), n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  [{n}x] code={code or '-'} param={param or '-'} msg={msg[:160]}")
    # Show a few first examples
    if samples:
        print("  First few examples:")
        for rid, code, param, msg in samples:
            print(f"    custom_id={rid} code={code or '-'} param={param or '-'} msg={msg[:200]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topic map + question batch pipeline")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning the data directory before run",
    )
    args = parser.parse_args()

    if not args.no_clean:
        clean_data_dir()
    else:
        print("[Main] Skipping cleanup (per --no-clean)")

    print(f"[Main] Using source: {SOURCE}")
    if not SOURCE.exists():
        raise SystemExit(f"[Main] Missing source file: {SOURCE}. Put your 3,000-line text at this path.")

    text = read_text(SOURCE)
    print(f"[Main] Loaded source text ({len(text)} chars). Chunking...")

    # --- Summary: MAP → TREE REDUCE (all batch) --------------------------
    from chunking import semantic_chunks
    chunks = semantic_chunks(text, max_chars=MAX_CHARS, soft_min=max(OVERLAP, 600))
    print(f"[Summary] Map phase: summarizing {len(chunks)} chunk(s) via batch...")
    micro_summaries = summarize_chunks_batch(chunks)

    print(f"[Summary] Reduce phase (tree): fanin={SUMMARY_REDUCE_FANIN} via batch...")
    summary_dict = merge_summaries_batch_tree(micro_summaries, fanin=SUMMARY_REDUCE_FANIN)
    ok, msg = write_summary(summary_dict, STUDY_SUMMARY)
    print(f"[Summary] {msg}")
    if not ok:
        print("[Summary][warn] Study summary failed validation; continuing with topic map + questions.")
    # --------------------------------------------------------------------

    topic_map = run_topicmap_batch(chunks)
    run_questions_batch(topic_map, chunks)