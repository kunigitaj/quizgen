# assemble.py

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from schema_models import Question, QuestionFile
from config import MAX_CONTEXT_SCAN_LEN


def parse_batch_output(jsonl_path: Path) -> List[Any]:
    """
    Parse OpenAI Batch /v1/responses JSONL.

    Priority:
      1) body.output[*].content[*].json   (structured blocks)
      2) body.output_text                 (convenience field when present)
      3) body.output[*].content[*].text   (fallback; fences allowed)

    Notes:
    - We parse each content block independently (avoids concatenating multiple JSON objects).
    - If a block is wrapped in fences or extra prose, we carve out the largest {...} object.
    """

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        # remove a single leading fence (``` or ```json)
        if s.startswith("```"):
            nl = s.find("\n")
            if nl != -1:
                s = s[nl + 1 :]
        # remove a single trailing fence
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _extract_json_object(s: str) -> str | None:
        """Return the largest top-level {...} span (tolerates prefix/suffix junk)."""
        stack = 0
        start = None
        best = None
        for i, ch in enumerate(s):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start is not None:
                        best = s[start : i + 1]  # keep the last (largest) object
        return best

    def _try_parse_json_string(s: str) -> Any | None:
        """Try direct JSON parse, then carve largest object."""
        if not isinstance(s, str) or not s.strip():
            return None
        s0 = _strip_code_fences(s)
        # 1) direct
        try:
            return json.loads(s0)
        except Exception:
            pass
        # 2) carved
        cand = _extract_json_object(s0)
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                return None
        return None

    out: List[Any] = []
    if not jsonl_path.exists():
        print(f"[parse_batch_output] File not found: {jsonl_path}")
        return out

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[parse_batch_output] JSON error on line {i}: {e}")
                continue

            # Responses Batch record shape:
            # { "custom_id": "...", "response": { "status_code": 200, "body": {...} }, ... }
            resp = obj.get("response", {}) if isinstance(obj, dict) else {}
            body = resp.get("body", {}) if isinstance(resp, dict) else {}

            parsed_any = False

            # --- (1) Structured JSON blocks inside body.output[*].content[*] ---
            output = body.get("output")
            if isinstance(output, list):
                for item in output:
                    # Be permissive: any item that has "content" list
                    if isinstance(item, dict):
                        content = item.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if not isinstance(c, dict):
                                    continue
                                ctype = (c.get("type") or "").lower()

                                # Known patterns:
                                # - {"type":"json","json":{...}}
                                # - {"type":"object","json":{...}}  (rare)
                                # Prefer explicit JSON payloads first.
                                if "json" in c and isinstance(c.get("json"), (dict, list)):
                                    out.append(c["json"])
                                    parsed_any = True
                                elif ctype == "json" and isinstance(c.get("data"), (dict, list)):
                                    # Some SDKs use "data" for JSON payload
                                    out.append(c["data"])
                                    parsed_any = True

            if parsed_any:
                continue  # done with this line

            # --- (2) Convenience synthesized text (present when text.format used) ---
            if isinstance(body.get("output_text"), str) and body["output_text"].strip():
                js = _try_parse_json_string(body["output_text"])
                if js is not None:
                    out.append(js)
                    continue
                # If output_text exists but isn't pure JSON, fall through to try content blocks as text.

            # --- (3) Fallback: scan body.output[*].content[*] for text blocks and parse individually ---
            if isinstance(output, list):
                found_from_text_blocks = False
                for item in output:
                    if isinstance(item, dict):
                        content = item.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if not isinstance(c, dict):
                                    continue
                                # Accept anything that exposes a 'text' field, regardless of c['type']
                                if isinstance(c.get("text"), str) and c["text"].strip():
                                    js = _try_parse_json_string(c["text"])
                                    if js is not None:
                                        out.append(js)
                                        found_from_text_blocks = True
                if found_from_text_blocks:
                    continue

            # If we reach here, we failed to extract any usable text/JSON for this record.
            status = (resp or {}).get("status_code") or (resp or {}).get("status")
            body_keys = list(body.keys()) if isinstance(body, dict) else []
            err = (body or {}).get("error") if isinstance(body, dict) else None

            # Extra diagnostics: enumerate the content types we saw (if any)
            content_kinds = []
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and isinstance(item.get("content"), list):
                        kinds = []
                        for c in item["content"]:
                            if isinstance(c, dict):
                                kinds.append((c.get("type"), list(c.keys())))
                        if kinds:
                            content_kinds.append(kinds)

            print(
                f"[parse_batch_output] No parseable JSON/text on line {i}. "
                f"status={status} keys={body_keys} error={err} "
                f"contentKinds={content_kinds}"
            )

    print(f"[parse_batch_output] Parsed {len(out)} payload(s).")
    return out


def extract_questions(payloads: List[Any]) -> List[Dict]:
    """Flatten parsed payloads into a list of question dicts."""
    items: List[Dict] = []
    for i, data in enumerate(payloads):
        if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
            items.extend([d for d in data["questions"] if isinstance(d, dict)])
        elif isinstance(data, list):
            items.extend([d for d in data if isinstance(d, dict)])
        elif isinstance(data, dict):
            if {"id", "type", "unit_id", "topic_id", "choices", "question_rich", "context_rich"}.issubset(data.keys()):
                items.append(data)
            else:
                print(f"[extract_questions] Payload #{i+1} dict doesn't match question shape. Keys={list(data.keys())[:10]}")
        else:
            print(f"[extract_questions] Payload #{i+1} unsupported type: {type(data)}")
    print(f"[extract_questions] Extracted {len(items)} question object(s).")
    return items


def ensure_ids_unique(items: List[Dict]):
    seen = set()
    for q in items:
        qid = q.get("id") or f"q_{uuid.uuid4().hex[:6]}"
        while qid in seen:
            qid = f"q_{uuid.uuid4().hex[:6]}"
        q["id"] = qid
        seen.add(qid)


def _gather_text(rich_list):
    out = []
    for block in rich_list or []:
        if isinstance(block, dict):
            if block.get("type") == "paragraph":
                for ch in (block.get("children") or []):
                    t = ch.get("text")
                    if t:
                        out.append(t)
            else:
                for ch in (block.get("children") or []):
                    if isinstance(ch, dict):
                        t = ch.get("text")
                        if t:
                            out.append(t)
    return out


def _rich_to_plain_text(rich_list: List[Dict], max_chars: int = 1000) -> str:
    """Flatten a rich block list to plaintext (soft cap length)."""
    txt = " ".join(_gather_text(rich_list)).strip()
    return (txt[:max_chars] + "…") if len(txt) > max_chars else txt


def context_leak_check(question: Dict) -> Tuple[bool, List[str]]:
    leaks: List[str] = []
    context_text = json.dumps(question.get("context_rich", ""))[:MAX_CONTEXT_SCAN_LEN].lower()
    for c in question.get("choices", []):
        if c.get("is_correct"):
            plain = " ".join(_gather_text(c.get("text_rich", []))).lower().strip()
            if plain and plain in context_text:
                leaks.append(plain)
    return (len(leaks) == 0, leaks)


def soften_context(question: Dict):
    ok, leaks = context_leak_check(question)
    if ok:
        return
    ctx = question.get("context_rich", [])
    ctx_json = json.dumps(ctx)
    for leak in leaks:
        pattern = re.escape(leak)
        ctx_json = re.sub(pattern, "this concept", ctx_json, flags=re.I)
    question["context_rich"] = json.loads(ctx_json)


# -----------------------
# Tag hygiene helpers
# -----------------------
_TAG_RE = re.compile(r"[^a-z0-9]+")

def _slugify_tag(s: str) -> str:
    """
    snake_case: lowercase, non-alnum -> '_', collapse repeats, strip edges.
    """
    if not isinstance(s, str):
        s = str(s or "")
    s = s.strip().lower()
    s = _TAG_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _normalize_tag_list(values) -> List[str]:
    """
    Normalize to snake_case and dedupe while preserving first occurrence order.
    """
    out = []
    seen = set()
    for v in (values or []):
        t = _slugify_tag(v)
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out
# -----------------------


def _normalize_choices_and_meta(item: Dict):
    """
    Normalize per spec:
    - MCQ/MSQ: exactly A–E choices, ensure rationales
    - MSQ: 2–3 correct
    - MCQ: exactly 1 correct
    - TF: keep shuffle False
    - Difficulty: clamp to 1..3 (prompt requirement)
    - Hints: cap to 3 later
    """
    t = (item.get("type") or "").lower()

    # Clamp difficulty 1..3 (schema allows up to 5, but prompt requires 1–3)
    try:
        d = int(item.get("difficulty", 2))
    except Exception:
        d = 2
    item["difficulty"] = max(1, min(3, d))

    if t in ("mcq", "msq"):
        choices = [c for c in (item.get("choices") or []) if isinstance(c, dict)]

        # Ensure exactly 5 choices; pad or trim as needed
        while len(choices) < 5:
            nxt = chr(ord("A") + len(choices))
            choices.append(
                {
                    "id": nxt,
                    "text_rich": [
                        {"type": "paragraph", "children": [{"text": f"Option {nxt}"}]}
                    ],
                    "is_correct": False,
                    "rationale_rich": [
                        {
                            "type": "paragraph",
                            "children": [
                                {
                                    "text": "Placeholder choice added to meet schema."
                                }
                            ],
                        }
                    ],
                }
            )
        if len(choices) > 5:
            choices = choices[:5]

        # Relabel to A..E in order and ensure rationales
        for i, c in enumerate(choices[:5]):
            c["id"] = chr(ord("A") + i)
            if c.get("rationale_rich") in (None, [], ""):
                c["rationale_rich"] = [
                    {
                        "type": "paragraph",
                        "children": [
                            {
                                "text": "This option is (in)correct based on the supplied context."
                            }
                        ],
                    }
                ]

        # MSQ: ensure 2–3 correct
        if t == "msq":
            correct_idx = [i for i, c in enumerate(choices) if c.get("is_correct")]
            if len(correct_idx) < 2:
                for i, c in enumerate(choices):
                    if not c.get("is_correct"):
                        c["is_correct"] = True
                        correct_idx.append(i)
                        if len(correct_idx) >= 2:
                            break
            if len(correct_idx) > 3:
                for i in correct_idx[3:]:
                    choices[i]["is_correct"] = False

        # MCQ: ensure exactly 1 correct
        if t == "mcq":
            correct_idx = [i for i, c in enumerate(choices) if c.get("is_correct")]
            if not correct_idx:
                choices[0]["is_correct"] = True
            elif len(correct_idx) > 1:
                for i in correct_idx[1:]:
                    choices[i]["is_correct"] = False

        item["choices"] = choices

    if t == "tf":
        # Keep canonical order and no shuffling for TF
        item["shuffle"] = False


def _enforce_id_convention(items: List[Dict]):
    """
    Force IDs to the pattern: q_<topic_id>_<type>_<nn>
    where nn is 01-based, per (topic_id, type).
    """
    counters: Dict[Tuple[str, str], int] = {}
    for q in items:
        topic = (q.get("topic_id") or "").strip()
        qtype = (q.get("type") or "").strip().lower()
        if not topic or qtype not in {"mcq", "msq", "tf"}:
            # Fall back to existing id if we can't build one
            q.setdefault("id", f"q_{uuid.uuid4().hex[:6]}")
            continue
        key = (topic, qtype)
        counters[key] = counters.get(key, 0) + 1
        seq = counters[key]
        q["id"] = f"q_{topic}_{qtype}_{seq:02d}"


def validate_and_fix(items: List[Dict]) -> List[Dict]:
    fixed: List[Dict] = []

    # First, enforce the canonical ID convention across all items
    _enforce_id_convention(items)

    # Safety: still guarantee uniqueness if any collisions slip through
    ensure_ids_unique(items)

    for it in items:
        # --- Tag hygiene (snake_case + dedupe) ---
        it["tags"] = _normalize_tag_list(it.get("tags"))
        it["concept_tags"] = _normalize_tag_list(it.get("concept_tags"))
        it["context_tags"] = _normalize_tag_list(it.get("context_tags"))

        if it.get("type") == "msq":
            it.setdefault(
                "grading",
                {
                    "mode": "msq",
                    "partial_credit": True,
                    "penalty": 0,
                    "require_all_correct": False,
                },
            )
        elif it.get("type") in ("mcq", "tf"):
            it.setdefault(
                "grading",
                {
                    "mode": "mcq",
                    "partial_credit": False,
                    "penalty": 0,
                    "require_all_correct": False,
                },
            )

        it.setdefault("shuffle", True)

        # Then soften to prevent leaks (covers any accidental answer echoes)
        soften_context(it)

        # Normalize per-type constraints
        _normalize_choices_and_meta(it)

        # Keep TF wording canonical
        if it.get("type") == "tf":
            for c in it.get("choices", []):
                txts = _gather_text(c.get("text_rich", []))
                joined = " ".join(txts).strip().lower()
                if "true" in joined:
                    c["text_rich"] = [
                        {"type": "paragraph", "children": [{"text": "True"}]}
                    ]
                elif "false" in joined:
                    c["text_rich"] = [
                        {"type": "paragraph", "children": [{"text": "False"}]}
                    ]

        # Ensure 2–3 hints (cap at 3)
        hints = it.get("hints_rich") or []
        if len(hints) < 2:
            default_tips = [
                {
                    "type": "callout",
                    "variant": "tip",
                    "children": [
                        {
                            "type": "paragraph",
                            "children": [
                                {"text": "Re-read the context and focus on key terms."}
                            ],
                        }
                    ],
                },
                {
                    "type": "callout",
                    "variant": "tip",
                    "children": [
                        {
                            "type": "paragraph",
                            "children": [
                                {
                                    "text": "Eliminate distractors that contradict definitions in the text."
                                }
                            ],
                        }
                    ],
                },
            ]
            hints = (hints + default_tips)[:2]
        it["hints_rich"] = hints[:3]

        try:
            Question(**it)
            fixed.append(it)
        except Exception as e:
            print(
                f"[validate_and_fix] Dropping invalid item (reason: {e}). "
                f"Item preview: {json.dumps(it)[:300]}..."
            )
            continue

    print(f"[validate_and_fix] Final valid questions: {len(fixed)}")
    return fixed


def write_final(questions: List[Dict], out_path: Path):
    payload = QuestionFile(schema_version="1.0", questions=questions).model_dump()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write_final] Wrote final questions JSON → {out_path} (count={len(questions)})")