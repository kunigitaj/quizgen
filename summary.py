# summary.py

import json
import os
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI  # run_batch uses this under the hood

from config import (
    OPENAI_API_KEY,
    MODEL_SUMMARY,
    SUMMARY_MAX_OUTPUT_TOKENS,
    REASONING_SUMMARY_EFFORT,
    VERBOSITY_SUMMARY,
    SUMMARY_REDUCE_FANIN,               # kept for backwards compatibility
    SUMMARY_STAGE1_MAX_OUTPUT_TOKENS,   # per micro-summary cap
    SUMMARY_REDUCE_MAX_OUTPUT_TOKENS,   # final polish cap
)
from prompts import (
    SUMMARY_SYSTEM,
    SUMMARY_USER,
    SUMMARY_MAP_SYSTEM,
    SUMMARY_MAP_USER,
    SUMMARY_REDUCE_SYSTEM,
    SUMMARY_REDUCE_USER,
    SUMMARY_POLISH_SYSTEM,
    SUMMARY_POLISH_USER,
)
from schema_models import StudySummary

# Batch helpers (already used elsewhere in the repo)
from run_batch import submit_batch, wait_for_batch, download_output_or_error

# -----------------------------------------------------------------------------
# Logging setup (lvl: DEBUG/INFO/WARN/ERROR), timestamp via env
# -----------------------------------------------------------------------------
_LOG_LEVEL = (os.getenv("SUMMARY_LOG_LEVEL") or "INFO").upper()
_LOG_TS = os.getenv("SUMMARY_LOG_TIMESTAMP")
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [summary] %(message)s" if _LOG_TS in ("1", "true", "yes")
    else "%(levelname)s [summary] %(message)s",
)
log = logging.getLogger("summary")

# -----------------------------------------------------------------------------
# OpenAI client (not directly used here; run_batch relies on it)
# -----------------------------------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Redaction + env helpers
# -----------------------------------------------------------------------------
_ID_TOKEN_RE = re.compile(r'\b(?:resp|run|msg|file|batch|job|ft|rs)_[A-Za-z0-9\-\._]+\b')
_BEARER_RE = re.compile(r'Bearer\s+[A-Za-z0-9\-\._]+')

def _obf(s: object) -> str:
    """Obfuscate IDs/tokens for logs."""
    if s is None:
        return "None"
    if not isinstance(s, str):
        s = str(s)
    s = _ID_TOKEN_RE.sub("[redacted]", s)
    s = _BEARER_RE.sub("Bearer [redacted]", s)
    s = re.sub(r"\b[a-f0-9]{16,}\b", "[hex]", s, flags=re.I)
    return s

def _preview(s: object, n: int = 400) -> str:
    """Safe preview w/ redaction and length cap."""
    if s is None:
        return "None"
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = _obf(s)
    return (s[:n] + ("..." if len(s) > n else ""))

def _len_safe(s: object) -> int:
    try:
        return len(s)  # type: ignore[arg-type]
    except Exception:
        return -1

def _format_validation_err(e: Exception) -> str:
    s = re.sub(r"\s+", " ", str(e)).strip()
    return _preview(s, 800)

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

# -----------------------------------------------------------------------------
# Env knobs for shard sizes and logging & retries
# -----------------------------------------------------------------------------
_SUMMARY_BATCH_SHARD_SIZE = int(os.getenv("SUMMARY_BATCH_SHARD_SIZE", "24"))
_SUMMARY_BATCH_MAX_BYTES = int(os.getenv("SUMMARY_BATCH_MAX_BYTES", "0"))  # 0 = ignore
_SUMMARY_LOG_COMPACT = _env_bool("SUMMARY_LOG_COMPACT", True)              # aggregate warnings
_SUMMARY_PROGRESS_LOG_EVERY = int(os.getenv("SUMMARY_PROGRESS_LOG_EVERY", "1"))  # shard cadence
# retry knobs
_SUMMARY_MAP_RETRY_MISSING = int(os.getenv("SUMMARY_MAP_RETRY_MISSING", "1"))     # retry failed chunk summaries
_SUMMARY_REDUCE_RETRY_MISSING = int(os.getenv("SUMMARY_REDUCE_RETRY_MISSING", "2"))  # retry failed merges

# -----------------------------------------------------------------------------
# JSON sanitizers / helpers (robust parsing)
# -----------------------------------------------------------------------------
_ZW_CHARS = "".join(["\u200b","\u200c","\u200d","\ufeff","\u2060","\u2028","\u2029"])
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

def _clean_jsonish_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return s
    s = _CODE_FENCE_RE.sub("", s)
    s = (s.replace("\u00a0"," ")
           .replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("‘", "'"))
    s = s.translate({ord(c): None for c in _ZW_CHARS})
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s

def _carve_largest_object(s: str) -> Optional[str]:
    stack, start, best = 0, None, None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start is not None:
                best = s[start:i+1]
    return best

# -----------------------------------------------------------------------------
# Build a /v1/responses request body → batch line
# -----------------------------------------------------------------------------
def _make_request_line(custom_id: str, sys_content: str, usr_content: str, *, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    body = {
        "model": MODEL_SUMMARY,
        "input": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content},
        ],
        # Use Responses API "text.format" (NOT response_format)
        "text": {
            "format": {"type": "json_object"},
            "verbosity": VERBOSITY_SUMMARY,
        },
        "max_output_tokens": max_tokens or SUMMARY_MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": REASONING_SUMMARY_EFFORT},  # ensure env is "minimal"
        # NOTE: GPT-5 reasoning models ignore temperature/top_p; do not set
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }

# -----------------------------------------------------------------------------
# JSONL parsing with custom_id awareness (more robust)
# -----------------------------------------------------------------------------
def _extract_json_from_body(body: dict) -> List[Any]:
    """
    Extract JSON objects from various Response API shapes, tolerating:
    - output_text with code fences / stray chars
    - content.json as list or dict
    - output_text wrapped in a one-element array
    """
    out: List[Any] = []
    if not isinstance(body, dict):
        return out

    def _try_parse_text(s: str):
        s = _clean_jsonish_text(s or "")
        if not s:
            return None
        # 1) direct parse
        try:
            obj = json.loads(s)
            return obj
        except Exception:
            pass
        # 2) carve the largest {...}
        carved = _carve_largest_object(s)
        if carved:
            carved = _clean_jsonish_text(carved)
            try:
                obj = json.loads(carved)
                return obj
            except Exception:
                pass
        # 3) array-wrapped JSON (e.g., [ { ... } ])
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, dict) and it:
                            return it
            except Exception:
                pass
        return None

    def _walk_blocks(blocks):
        hits = []
        if isinstance(blocks, list):
            for item in blocks:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            # try attribute-style objects (SDK variants)
                            cj = getattr(c, "json", None)
                            if isinstance(cj, dict):
                                hits.append(cj)
                                continue
                            txt = getattr(c, "text", None)
                            if isinstance(txt, str):
                                maybe = _try_parse_text(txt)
                                if maybe is not None:
                                    hits.append(maybe)
                            continue
                        # A) explicit json payload (dict or list)
                        if "json" in c:
                            cj = c.get("json")
                            if isinstance(cj, dict):
                                hits.append(cj)
                                continue
                            if isinstance(cj, list):
                                for it in cj:
                                    if isinstance(it, dict) and it:
                                        hits.append(it)
                                        break
                                continue
                        # B) explicit output_text-like text
                        if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                            maybe = _try_parse_text(c["text"])
                            if maybe is not None:
                                hits.append(maybe)
                                continue
                        # C) bare text (older shapes)
                        if isinstance(c.get("text"), str):
                            maybe = _try_parse_text(c["text"])
                            if maybe is not None:
                                hits.append(maybe)
        return hits

    # 1) common: body["output"]
    hits = _walk_blocks(body.get("output"))
    if hits:
        return hits

    # 2) nested: body["response"]["output"] (seen in some SDKs)
    resp_obj = body.get("response")
    if isinstance(resp_obj, dict):
        hits = _walk_blocks(resp_obj.get("output"))
        if hits:
            return hits
        ot = resp_obj.get("output_text")
        if isinstance(ot, str):
            maybe = _try_parse_text(ot)
            if maybe is not None:
                return [maybe]

    # 3) body["output_text"]
    ot = body.get("output_text")
    if isinstance(ot, str):
        maybe = _try_parse_text(ot)
        if maybe is not None:
            return [maybe]

    # 4) message/messages holders
    for k in ("message", "messages"):
        hits = _walk_blocks(body.get(k))
        if hits:
            return hits

    # 5) chat-compat: choices[0].message.content
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                maybe = _try_parse_text(content)
                if maybe is not None:
                    return [maybe]

    return out

def _parse_jsonl_with_ids(jsonl_path: Path) -> List[Tuple[str, Any]]:
    """Return list of (custom_id, parsed_payload). Emits compact warning summaries when enabled."""
    results: List[Tuple[str, Any]] = []
    if not jsonl_path.exists():
        log.error(f"[parse] Output JSONL not found: {jsonl_path}")
        return results

    no_payload_ids: List[str] = []
    error_items: List[Tuple[str, str]] = []  # (cid, message)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                log.warning(f"[parse] JSON error on line {i}: {e}")
                continue

            cid = obj.get("custom_id") or ""
            resp = obj.get("response", {}) if isinstance(obj, dict) else {}
            body = resp.get("body", {}) if isinstance(resp, dict) else {}

            payloads = _extract_json_from_body(body)
            if payloads:
                results.append((cid, payloads[0]))
                continue

            err = (body or {}).get("error")
            if err:
                msg = err.get("message") or "(no message)"
                if _SUMMARY_LOG_COMPACT:
                    error_items.append((cid, _preview(msg, 200)))
                else:
                    log.warning(f"[parse] custom_id={_obf(cid)} error: {_preview(msg, 200)}")
            else:
                if _SUMMARY_LOG_COMPACT:
                    no_payload_ids.append(cid)
                else:
                    log.warning(f"[parse] custom_id={_obf(cid)} had no parseable payloads.")

    # Emit compact summaries if enabled
    if _SUMMARY_LOG_COMPACT:
        if no_payload_ids:
            log.warning(
                f"[parse] {len(no_payload_ids)} request(s) had no parseable payloads. "
                f"Examples: {', '.join(_obf(x) for x in no_payload_ids[:5])}"
                + (" ..." if len(no_payload_ids) > 5 else "")
            )
        if error_items:
            log.warning(
                f"[parse] {len(error_items)} request(s) returned API error bodies. "
                f"Examples: "
                + "; ".join(f"{_obf(cid)}: {msg}" for cid, msg in error_items[:3])
                + (" ..." if len(error_items) > 3 else "")
            )
    return results

# -----------------------------------------------------------------------------
# Coercion / normalization to StudySummary schema
# -----------------------------------------------------------------------------
_ALLOWED_COLORS = {"blue.600", "green.600", "amber.600", "red.600", "purple.600"}

def _ensure_color(c: Optional[str]) -> str:
    if isinstance(c, str) and c in _ALLOWED_COLORS:
        return c
    return "blue.600"

def _as_str_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if isinstance(t, (str, int, float)) and str(t).strip()]
    if isinstance(x, (str, int, float)) and str(x).strip():
        return [str(x).strip()]
    return []

def _normalize_section(s: dict) -> Optional[dict]:
    if not isinstance(s, dict):
        return None
    title = s.get("title") or s.get("sectionTitle") or s.get("heading") or s.get("label")
    if not isinstance(title, str) or not title.strip():
        return None
    bullets = _as_str_list(s.get("bullets") or s.get("points") or s.get("content") or s.get("items"))
    subs_in = s.get("subsections") or []
    subsections: List[dict] = []
    if isinstance(subs_in, list):
        for sub in subs_in:
            st = sub.get("title") or sub.get("heading") or sub.get("label") if isinstance(sub, dict) else None
            if isinstance(st, str) and st.strip():
                subsections.append({
                    "title": st.strip(),
                    "bullets": _as_str_list(
                        (sub.get("bullets") if isinstance(sub, dict) else None) or
                        (sub.get("points") if isinstance(sub, dict) else None) or
                        (sub.get("items") if isinstance(sub, dict) else None)
                    )
                })
    return {"title": title.strip(), "bullets": bullets, "subsections": subsections}

def _normalize_subheading(h: dict) -> Optional[dict]:
    if not isinstance(h, dict):
        return None
    heading = h.get("heading") or h.get("label") or h.get("title")
    content = _as_str_list(h.get("content") or h.get("bullets") or h.get("points") or h.get("items"))
    color = _ensure_color(h.get("color"))
    if not isinstance(heading, str) or not heading.strip():
        return None
    return {"heading": heading.strip(), "color": color, "content": content}

def _normalize_slide(sl: dict) -> Optional[dict]:
    if not isinstance(sl, dict):
        return None
    title = sl.get("title") or sl.get("heading")
    if not isinstance(title, str) or not title.strip():
        return None
    subtitle = sl.get("subtitle")
    if subtitle is not None and not isinstance(subtitle, str):
        subtitle = None

    subs_raw = sl.get("subheadings")
    subs: List[dict] = []
    if isinstance(subs_raw, list):
        for h in subs_raw:
            nh = _normalize_subheading(h) if isinstance(h, dict) else None
            if nh:
                subs.append(nh)

    # If no subheadings, synthesize one from any bullets/points present
    if not subs:
        synthesized = _as_str_list(sl.get("bullets") or sl.get("points") or sl.get("items") or sl.get("content"))
        if synthesized:
            subs = [{"heading": "Key points", "color": "blue.600", "content": synthesized}]

    # Still empty? Provide a minimal valid shell
    if not subs:
        subs = [{"heading": "Notes", "color": "blue.600", "content": []}]

    return {"title": title.strip(), "subtitle": (subtitle.strip() if isinstance(subtitle, str) and subtitle.strip() else None), "subheadings": subs}

def _normalize_summary_shape(obj: dict) -> dict:
    """Coerce loosely-shaped payloads into valid StudySummary structure."""
    if not isinstance(obj, dict):
        return {"schema_version": "1.0", "narrativeSections": [], "slides": []}

    out = {
        "schema_version": "1.0",
        "narrativeSections": [],
        "slides": [],
    }

    # Accept alternate keys from micros
    sections = []
    if isinstance(obj.get("narrativeSections"), list):
        sections.extend(obj["narrativeSections"])
    if isinstance(obj.get("sections"), list):
        sections.extend(obj["sections"])
    if isinstance(obj.get("section"), dict):
        sections.append(obj["section"])

    slides = []
    if isinstance(obj.get("slides"), list):
        slides.extend(obj["slides"])
    if isinstance(obj.get("slide"), dict):
        slides.append(obj["slide"])

    # Normalize sections
    for s in sections:
        ns = _normalize_section(s) if isinstance(s, dict) else None
        if ns:
            out["narrativeSections"].append(ns)

    # Normalize slides
    for sl in slides:
        nsl = _normalize_slide(sl) if isinstance(sl, dict) else None
        if nsl:
            out["slides"].append(nsl)

    # Defensive: ensure lists present
    if not isinstance(out["narrativeSections"], list):
        out["narrativeSections"] = []
    if not isinstance(out["slides"], list):
        out["slides"] = []

    return out

# -----------------------------------------------------------------------------
# Simple local merge (fallback) to guarantee coverage when the API drops groups
# -----------------------------------------------------------------------------
def _local_merge(a: dict, b: dict) -> dict:
    """Best-effort deterministic merge of two StudySummary-like dicts."""
    a = _normalize_summary_shape(a or {})
    b = _normalize_summary_shape(b or {})
    out = {"schema_version": "1.0", "narrativeSections": [], "slides": []}

    # Merge narrativeSections by title (preserve order, dedupe bullets)
    def _merge_sections(sa: List[dict], sb: List[dict]) -> List[dict]:
        by_title: Dict[str, dict] = {}
        order: List[str] = []
        for s in sa + sb:
            t = (s.get("title") or "").strip()
            if not t:
                continue
            if t not in by_title:
                by_title[t] = {"title": t, "bullets": [], "subsections": []}
                order.append(t)
            # bullets (dedupe, keep <= 6)
            seen = set(by_title[t]["bullets"])
            for blt in s.get("bullets") or []:
                blt = blt.strip()
                if blt and blt not in seen and len(by_title[t]["bullets"]) < 6:
                    by_title[t]["bullets"].append(blt)
                    seen.add(blt)
            # subsections (append up to 4, each up to 5 bullets)
            for sub in s.get("subsections") or []:
                if len(by_title[t]["subsections"]) >= 4:
                    break
                sub_t = (sub.get("title") or "").strip()
                if not sub_t:
                    continue
                norm_sub = {"title": sub_t, "bullets": []}
                for blt in sub.get("bullets") or []:
                    if len(norm_sub["bullets"]) < 5:
                        norm_sub["bullets"].append(blt.strip())
                by_title[t]["subsections"].append(norm_sub)
        return [by_title[k] for k in order]

    # Merge slides by title (preserve order, ensure subheadings exist)
    def _merge_slides(sa: List[dict], sb: List[dict]) -> List[dict]:
        by_title: Dict[str, dict] = {}
        order: List[str] = []
        for s in sa + sb:
            t = (s.get("title") or "").strip()
            if not t:
                continue
            if t not in by_title:
                by_title[t] = {"title": t, "subtitle": s.get("subtitle") if isinstance(s.get("subtitle"), str) else None, "subheadings": []}
                order.append(t)
            # collect subheadings
            for sh in s.get("subheadings") or []:
                if len(by_title[t]["subheadings"]) >= 3:
                    break
                content = [c.strip() for c in (sh.get("content") or []) if isinstance(c, str) and c.strip()]
                if not content:
                    continue
                by_title[t]["subheadings"].append({
                    "heading": (sh.get("heading") or "Key points").strip(),
                    "color": _ensure_color(sh.get("color")),
                    "content": content[:5]
                })
        # ensure every slide has at least one subheading
        for t in order:
            if not by_title[t]["subheadings"]:
                by_title[t]["subheadings"].append({"heading": "Key points", "color": "blue.600", "content": []})
        return [by_title[k] for k in order]

    out["narrativeSections"] = _merge_sections(a.get("narrativeSections") or [], b.get("narrativeSections") or [])
    out["slides"] = _merge_slides(a.get("slides") or [], b.get("slides") or [])
    return out

# -----------------------------------------------------------------------------
# Sharding helpers + multi-line batch runner with progress
# -----------------------------------------------------------------------------
def _shard_lines(lines: List[Dict[str, Any]], max_per: int, max_bytes: int) -> List[List[Dict[str, Any]]]:
    if max_per <= 0 and max_bytes <= 0:
        return [lines]
    shards: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_bytes = 0
    for req in lines:
        s = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        if (max_per > 0 and len(cur) >= max_per) or (max_bytes > 0 and cur_bytes + len(s) > max_bytes):
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(req)
        cur_bytes += len(s)
    if cur:
        shards.append(cur)
    return shards

def _run_batch_multi(request_lines: List[Dict[str, Any]], tmp_prefix: str) -> List[Tuple[str, Any]]:
    """
    Submit many request lines via one or more shards, with detailed logs.
    Returns a list of (custom_id, payload) across all shards, preserving shard order.
    """
    total = len(request_lines)
    if total == 0:
        return []

    shards = _shard_lines(request_lines, _SUMMARY_BATCH_SHARD_SIZE, _SUMMARY_BATCH_MAX_BYTES)
    log.info(
        f"[batch] Submitting {total} request(s) in {len(shards)} shard(s) "
        f"(max_per_shard={_SUMMARY_BATCH_SHARD_SIZE}, byte_cap={_SUMMARY_BATCH_MAX_BYTES})"
    )

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Tuple[str, Any]] = []
    parsed_so_far = 0

    for si, shard in enumerate(shards, start=1):
        shard_in = data_dir / f"{tmp_prefix}.shard{si:02d}.input.jsonl"
        shard_out = data_dir / f"{tmp_prefix}.shard{si:02d}.output.jsonl"

        # Write shard file
        with shard_in.open("w", encoding="utf-8") as f:
            for line in shard:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        log.info(f"[batch] Shard {si}/{len(shards)}: wrote {len(shard)} request(s) → {shard_in}")

        batch_id = submit_batch(shard_in)
        log.info(f"[batch] Shard {si}: submitted id={_obf(batch_id)}. Waiting...")

        st = wait_for_batch(batch_id)
        log.info(f"[batch] Shard {si}: status={st.status}")

        if st.status != "completed":
            kind, path = download_output_or_error(st, shard_out)
            if kind == "error":
                try:
                    err_head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
                except Exception as e:
                    err_head = f"(could not read error file: {e})"
                log.error(f"[batch] Shard {si}: error file saved to {path}\n{_preview(err_head, 1200)}")
            raise RuntimeError(f"[batch] Shard {si} did not complete successfully: status={st.status}")

        kind, path = download_output_or_error(st, shard_out)
        if kind == "error":
            try:
                err_head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
            except Exception as e:
                err_head = f"(could not read error file: {e})"
            log.error(f"[batch] Shard {si}: error file saved to {path}\n{_preview(err_head, 1200)}")
            raise RuntimeError(f"[batch] Shard {si} returned errors (no output).")

        # Parse with IDs
        results = _parse_jsonl_with_ids(shard_out)
        parsed_this = len(results)
        parsed_so_far += parsed_this
        all_results.extend(results)

        # Per-shard + overall progress
        pct = (parsed_so_far / total * 100.0) if total else 100.0
        log.info(
            f"[batch] Shard {si}: parsed {parsed_this}/{len(shard)} payload(s). "
            f"Overall: {parsed_so_far}/{total} ({pct:.1f}%)"
        )

        # Optional cadence
        if _SUMMARY_PROGRESS_LOG_EVERY > 1 and (si % _SUMMARY_PROGRESS_LOG_EVERY == 0 or si == len(shards)):
            log.info(f"[batch] Progress checkpoint: {parsed_so_far}/{total} ({pct:.1f}%) parsed")

    log.info(f"[batch] Finished all shards: parsed {parsed_so_far}/{total} payload(s) total.")
    return all_results

# -----------------------------------------------------------------------------
# Public APIs
# -----------------------------------------------------------------------------
class SummaryValidationError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type(SummaryValidationError),
)
def create_study_summary(full_text: str) -> dict:
    """
    One-shot summary of the entire source via a single batched request.
    Use this when the full source comfortably fits in context.
    """
    text_len = _len_safe(full_text)
    log.info(
        "[one-shot/batch] create_study_summary: "
        f"source_len={text_len}, model={MODEL_SUMMARY}, "
        f"max_output_tokens={SUMMARY_MAX_OUTPUT_TOKENS}, verbosity={VERBOSITY_SUMMARY}, "
        f"reasoning={REASONING_SUMMARY_EFFORT}"
    )

    usr_content = SUMMARY_USER.format(full_text=full_text)
    line = _make_request_line(
        custom_id="summary_one_shot_0001",
        sys_content=SUMMARY_SYSTEM,
        usr_content=usr_content,
    )

    results = _run_batch_multi([line], tmp_prefix="summary_one_shot")
    if not results or not isinstance(results[0][1], dict):
        raise SummaryValidationError("No valid StudySummary payload returned.")

    parsed = _normalize_summary_shape(results[0][1])
    try:
        StudySummary(**parsed)
    except Exception as ve:
        log.warning("[one-shot/batch] StudySummary validation failed: " + _format_validation_err(ve))
        raise SummaryValidationError("Validation failed for StudySummary (batch)")
    log.info("[one-shot/batch] StudySummary validated successfully.")
    return parsed

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def summarize_chunk(chunk_text: str) -> dict:
    """
    Backward-compatible single-chunk map call (batch under the hood).
    Prefer summarize_chunks_batch for better efficiency/logs.
    """
    clen = _len_safe(chunk_text)
    log.debug(f"[map/batch] summarize_chunk len={clen}")
    usr_content = SUMMARY_MAP_USER.format(chunk_text=chunk_text)
    line = _make_request_line(
        "summary_map_chunk_0001",
        SUMMARY_MAP_SYSTEM,
        usr_content,
        max_tokens=SUMMARY_STAGE1_MAX_OUTPUT_TOKENS,
    )
    results = _run_batch_multi([line], tmp_prefix="summary_map_chunk")
    obj = (results[0][1] if results else {}) or {}
    if isinstance(obj, list):
        obj = next((x for x in obj if isinstance(x, dict) and x), {}) or {}
    return _normalize_summary_shape(obj) if obj else {}

def summarize_chunks_batch(chunks: List[str]) -> List[dict]:
    """
    Submit many chunk summaries at once with detailed shard logs.
    Returns a list of parsed+normalized dicts aligned to input order; invalid/missing → {}.
    Retries any missing items up to _SUMMARY_MAP_RETRY_MISSING times.
    """
    if not chunks:
        log.warning("[map/batch] No chunks provided.")
        return []

    # initial submit
    def _submit(ids_and_texts: List[Tuple[int, str]]) -> Dict[int, dict]:
        lines: List[Dict[str, Any]] = []
        for i, ch in ids_and_texts:
            usr = SUMMARY_MAP_USER.format(chunk_text=ch)
            cid = f"summary_map_chunk_{i:04d}"
            lines.append(_make_request_line(cid, SUMMARY_MAP_SYSTEM, usr, max_tokens=SUMMARY_STAGE1_MAX_OUTPUT_TOKENS))
        results = _run_batch_multi(lines, tmp_prefix="summary_map_batch")
        by_idx: Dict[int, dict] = {}
        for cid, payload in results:
            try:
                idx = int(cid.split("_")[-1])
            except Exception:
                continue
            by_idx[idx] = payload if isinstance(payload, dict) else (payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else {})
        return by_idx

    ids_and_texts = [(i, ch) for i, ch in enumerate(chunks, start=1)]
    by_idx = _submit(ids_and_texts)

    # retry loop for missing
    attempts = 0
    while attempts < _SUMMARY_MAP_RETRY_MISSING:
        missing = [(i, ch) for i, ch in ids_and_texts if i not in by_idx or not isinstance(by_idx[i], dict) or not by_idx[i]]
        if not missing:
            break
        attempts += 1
        log.info(f"[map/batch] Retrying {len(missing)} missing chunk(s), attempt {attempts}/{_SUMMARY_MAP_RETRY_MISSING}")
        retr = _submit(missing)
        by_idx.update(retr)

    # build output in order
    out: List[dict] = []
    ok_count = 0
    for i, _ in ids_and_texts:
        obj = by_idx.get(i) or {}
        if isinstance(obj, list):
            obj = next((x for x in obj if isinstance(x, dict) and x), {}) or {}
        if isinstance(obj, dict) and obj:
            norm = _normalize_summary_shape(obj)
            out.append(norm)
            ok_count += 1
        else:
            out.append({})
    log.info(f"[map/batch] Completed chunk summaries: {ok_count}/{len(chunks)} successful after retries.")
    return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def merge_summaries(micro_json_list: list) -> dict:
    """
    Single-call reduce via batch. For maximum batch utilization, prefer tree reduce:
    merge_summaries_batch_tree(...).
    """
    log.debug(f"[reduce/batch] merge_summaries with {len(micro_json_list)} inputs")
    # Ensure inputs are normalized before sending to the model (helps it keep shape)
    safe_inputs = [_normalize_summary_shape(m) for m in micro_json_list if isinstance(m, dict)]
    usr_content = SUMMARY_REDUCE_USER.format(
        micro_json_list=json.dumps(safe_inputs, ensure_ascii=False)
    )
    line = _make_request_line("summary_reduce_0001", SUMMARY_REDUCE_SYSTEM, usr_content, max_tokens=SUMMARY_REDUCE_MAX_OUTPUT_TOKENS)
    results = _run_batch_multi([line], tmp_prefix="summary_reduce")
    if not results or not isinstance(results[0][1], dict):
        return {}
    parsed = _normalize_summary_shape(results[0][1])
    try:
        StudySummary(**parsed)
    except Exception as ve:
        log.warning("[reduce/batch] StudySummary validation failed: " + _format_validation_err(ve))
        raise
    log.info("[reduce/batch] Merged StudySummary validated successfully.")
    return parsed

def merge_summaries_batch_tree(micros: List[dict], fanin: Optional[int] = None) -> dict:
    """
    Multi-level batched reduce. Submits many merge requests per level (sharded),
    logs progress at each level, and repeats until one summary remains.
    Retries failed groups; if still missing, falls back to deterministic local merge
    so coverage is never lost.
    """
    if not micros:
        log.warning("[reduce-tree] No micro-summaries provided.")
        return {}

    # Normalize inputs up front
    current = [_normalize_summary_shape(m) for m in micros if isinstance(m, dict) and m]
    fanin = fanin or SUMMARY_REDUCE_FANIN
    level = 0
    log.info(f"[reduce-tree] Start with {len(current)} micro-summaries; fanin={fanin}")

    while len(current) > 1:
        level += 1
        # Build grouped requests for this level
        lines: List[Dict[str, Any]] = []
        groups: List[List[dict]] = []
        group: List[dict] = []
        for i, m in enumerate(current, start=1):
            group.append(m)
            if len(group) == fanin or i == len(current):
                safe_group = [_normalize_summary_shape(g) for g in group]
                usr = SUMMARY_REDUCE_USER.format(
                    micro_json_list=json.dumps(safe_group, ensure_ascii=False)
                )
                cid = f"summary_reduce_L{level:02d}_{len(groups)+1:04d}"
                lines.append(_make_request_line(cid, SUMMARY_REDUCE_SYSTEM, usr, max_tokens=SUMMARY_REDUCE_MAX_OUTPUT_TOKENS))
                groups.append(safe_group)
                group = []

        log.info(f"[reduce-tree] Level {level}: submitting {len(lines)} merge request(s).")
        results = _run_batch_multi(lines, tmp_prefix=f"summary_reduce_L{level:02d}")
        by_id = {cid: payload for cid, payload in results}

        # Detect missing and retry
        attempts = 0
        missing_ids = [f"summary_reduce_L{level:02d}_{i+1:04d}" for i in range(len(groups)) if f"summary_reduce_L{level:02d}_{i+1:04d}" not in by_id or not by_id[f"summary_reduce_L{level:02d}_{i+1:04d}"]]
        while missing_ids and attempts < _SUMMARY_REDUCE_RETRY_MISSING:
            attempts += 1
            log.info(f"[reduce-tree] Level {level}: retrying {len(missing_ids)} missing group(s), attempt {attempts}/{_SUMMARY_REDUCE_RETRY_MISSING}")
            retry_lines = []
            for mid in missing_ids:
                idx = int(mid.split("_")[-1])
                safe_group = groups[idx-1]
                usr = SUMMARY_REDUCE_USER.format(
                    micro_json_list=json.dumps(safe_group, ensure_ascii=False)
                )
                retry_lines.append(_make_request_line(mid, SUMMARY_REDUCE_SYSTEM, usr, max_tokens=SUMMARY_REDUCE_MAX_OUTPUT_TOKENS))
            retry_results = _run_batch_multi(retry_lines, tmp_prefix=f"summary_reduce_L{level:02d}_retry{attempts}")
            for cid, payload in retry_results:
                by_id[cid] = payload
            missing_ids = [mid for mid in missing_ids if mid not in by_id or not by_id[mid]]

        # Collect parsed dicts in original group order, local-merge fallback if still missing
        merged: List[dict] = []
        ok = 0
        for gi in range(len(groups)):
            cid = f"summary_reduce_L{level:02d}_{gi+1:04d}"
            payload = by_id.get(cid)
            normalized_payload: Optional[dict] = None
            if isinstance(payload, list):
                payload = next((x for x in payload if isinstance(x, dict) and x), {}) or {}
            if isinstance(payload, dict) and payload:
                normalized_payload = _normalize_summary_shape(payload)
                try:
                    StudySummary(**normalized_payload)
                except Exception as ve:
                    log.warning(f"[reduce-tree] Level {level} group {gi+1}: validation failed: {_format_validation_err(ve)}; using local merge fallback")
                    normalized_payload = None
            if normalized_payload:
                merged.append(normalized_payload)
                ok += 1
            else:
                # local merge fallback to avoid coverage loss
                gm = groups[gi]
                local = {}
                for part in gm:
                    local = _local_merge(local, part)
                merged.append(_normalize_summary_shape(local))

        log.info(f"[reduce-tree] Level {level}: merged {ok}/{len(groups)} groups (fanin={fanin}).")
        if not merged:
            log.error("[reduce-tree] No merges succeeded at this level; aborting.")
            return {}

        current = merged

    final = _normalize_summary_shape(current[0])
    try:
        StudySummary(**final)
    except Exception as ve:
        log.warning("[reduce-tree] Final StudySummary validation failed: " + _format_validation_err(ve))
        raise
    log.info("[reduce-tree] Final StudySummary validated.")
    return final

# -----------------------------------------------------------------------------
# New: Single-call POLISH for local-merge result
# -----------------------------------------------------------------------------
def polish_summary(merged: dict) -> dict:
    """
    Send the merged (local) StudySummary once to the model to:
    - enforce schema/constraints,
    - dedupe titles and bullets,
    - trim bullet length and ensure subheadings.
    If the polish call fails, return the normalized local merge.
    """
    safe = _normalize_summary_shape(merged or {})
    usr = SUMMARY_POLISH_USER.format(merged_json=json.dumps(safe, ensure_ascii=False))
    line = _make_request_line(
        "summary_polish_0001",
        SUMMARY_POLISH_SYSTEM,
        usr,
        max_tokens=SUMMARY_REDUCE_MAX_OUTPUT_TOKENS,
    )
    results = _run_batch_multi([line], tmp_prefix="summary_polish")
    if not results or not isinstance(results[0][1], dict):
        log.warning("[polish] No valid payload from polish call; returning local-merged result.")
        return safe
    polished = _normalize_summary_shape(results[0][1])
    try:
        StudySummary(**polished)
    except Exception as ve:
        log.warning("[polish] Validation failed on polished payload: " + _format_validation_err(ve))
        return safe
    return polished

def summarize_large_text_simple(chunks: List[str]) -> dict:
    """
    Preferred pipeline for very large sources when single-pass is not feasible:
      1) MAP: batch summarize chunks → micros
      2) LOCAL MERGE: deterministic fold of micros → merged
      3) POLISH: single LLM call to tidy and enforce constraints → final
    """
    micros = summarize_chunks_batch(chunks)
    merged: dict = {}
    for m in micros:
        merged = _local_merge(merged, m)
    final = polish_summary(merged)
    return final

# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------
def save_summary_batch_input(text: str, out_path: Path, custom_id: str = "summary_0001"):
    """
    Prepare a single JSONL line suitable for OpenAI Batch /v1/responses.
    """
    log.debug(f"[batch-prepare] Saving single-line batch input to {out_path}")
    usr_content = SUMMARY_USER.format(full_text=text)
    line = _make_request_line(custom_id=custom_id, sys_content=SUMMARY_SYSTEM, usr_content=usr_content)
    out_path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(f"[batch-prepare] Wrote batch input → {out_path}")

def write_summary(summary: dict, out_path: Path) -> Tuple[bool, str]:
    """
    Validate against StudySummary and write pretty JSON.
    """
    # Normalize before validating (heals minor shape drift)
    summary = _normalize_summary_shape(summary if isinstance(summary, dict) else {})
    try:
        validated = StudySummary(**summary).model_dump()
    except Exception as e:
        msg = f"StudySummary validation failed: {_format_validation_err(e)}"
        log.warning("[write] " + msg)
        return False, msg
    try:
        out_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"[write] Wrote study summary → {out_path}")
        return True, f"Wrote study summary → {out_path}"
    except Exception as e:
        msg = f"Failed to write summary file: {e}"
        log.error("[write] " + msg)
        return False, msg