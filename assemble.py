# assemble.py

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from schema_models import Question, QuestionFile
from config import MAX_CONTEXT_SCAN_LEN


def parse_batch_output(jsonl_path: Path) -> List[Any]:
    """Parse OpenAI Batch /v1/responses output JSONL assuming text.format={type:'json_object'}."""
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

            resp = obj.get("response", {})
            body = resp.get("body", {}) if isinstance(resp, dict) else {}

            txt = body.get("output_text")
            if not txt:
                output = body.get("output", [])
                if isinstance(output, list):
                    parts = []
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            for c in (item.get("content") or []):
                                t = c.get("text") if isinstance(c, dict) else None
                                if isinstance(t, str):
                                    parts.append(t)
                    txt = "\n".join(parts) if parts else None

            if not txt:
                print(f"[parse_batch_output] No text on line {i}. Keys in body: {list(body.keys())}")
                continue

            # Trim common code fences defensively
            s = txt.strip()
            if s.startswith("```"):
                s = s[3:]
                if "\n" in s:
                    s = s.split("\n", 1)[1]
                if s.endswith("```"):
                    s = s[:-3]
                s = s.strip()

            try:
                out.append(json.loads(s))
            except Exception as e:
                print(f"[parse_batch_output] Could not parse JSON on line {i}: {e}\nPreview: {s[:400]}...")
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


def validate_and_fix(items: List[Dict]) -> List[Dict]:
    fixed: List[Dict] = []
    ensure_ids_unique(items)
    for it in items:
        if it.get("type") == "msq":
            it.setdefault("grading", {
                "mode": "msq",
                "partial_credit": True,
                "penalty": 0,
                "require_all_correct": False
            })
        elif it.get("type") in ("mcq", "tf"):
            it.setdefault("grading", {
                "mode": "mcq",
                "partial_credit": False,
                "penalty": 0,
                "require_all_correct": False
            })

        it.setdefault("shuffle", True)
        soften_context(it)

        if it.get("type") == "tf":
            for c in it.get("choices", []):
                txts = _gather_text(c.get("text_rich", []))
                joined = " ".join(txts).strip().lower()
                if "true" in joined:
                    c["text_rich"] = [{"type": "paragraph", "children": [{"text": "True"}]}]
                elif "false" in joined:
                    c["text_rich"] = [{"type": "paragraph", "children": [{"text": "False"}]}]

        if it.get("type") in ("mcq", "msq"):
            for c in it.get("choices", []):
                if c.get("rationale_rich") in (None, [], ""):
                    c["rationale_rich"] = [
                        {"type": "paragraph", "children": [{"text": "This option is (in)correct based on the supplied context."}]}
                    ]

        # Ensure at least 2 hints
        hints = it.get("hints_rich") or []
        if len(hints) < 2:
            default_tips = [
                {"type": "callout", "variant": "tip",
                 "children": [{"type": "paragraph", "children": [{"text": "Re-read the context and focus on key terms."}]}]},
                {"type": "callout", "variant": "tip",
                 "children": [{"type": "paragraph", "children": [{"text": "Eliminate distractors that contradict definitions in the text."}]}]}
            ]
            hints = (hints + default_tips)[:2]
            it["hints_rich"] = hints

        try:
            Question(**it)
            fixed.append(it)
        except Exception as e:
            print(f"[validate_and_fix] Dropping invalid item (reason: {e}). Item preview: {json.dumps(it)[:300]}...")
            continue

    print(f"[validate_and_fix] Final valid questions: {len(fixed)}")
    return fixed


def write_final(questions: List[Dict], out_path: Path):
    payload = QuestionFile(schema_version="1.0", questions=questions).model_dump()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write_final] Wrote final questions JSON → {out_path} (count={len(questions)})")