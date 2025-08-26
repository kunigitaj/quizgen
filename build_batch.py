# build_batch.py

import json
from pathlib import Path
from typing import Dict, List
from prompts import QUESTIONS_SYSTEM, SCHEMA_ITEM_SHAPE, TYPE_EXAMPLES
from config import (
    MODEL_QUESTIONS,
    QUESTION_TYPES_PER_TOPIC,
    MAX_TOPIC_CONTEXT_CHARS,
    CONTEXT_SAMPLE_WINDOWS,
    SAMPLE_HEAD_CHARS,
    SAMPLE_TAIL_CHARS,
)

def write_jsonl(lines: List[Dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

FORCE_TYPE_USER = """Create EXACTLY 1 question of type: {force_type}
unit_id: {unit_id}
topic_id: {topic_id}
title: {title}
summary: {summary}

CONTEXT:
{context_text}

SCHEMA ITEM SHAPE:
{schema_item_shape}

TYPE EXAMPLES (use structure only, not content):
{type_examples}
"""

def _sample_topic_context(chunks: List[str], start: int, end: int) -> str:
    span = chunks[start : end + 1]
    if not span:
        return ""

    W = max(1, CONTEXT_SAMPLE_WINDOWS)
    total = len(span)
    step = max(1, total // W)

    parts: List[str] = []
    for i in range(0, total, step):
        block = span[i]
        head = block[:SAMPLE_HEAD_CHARS]
        tail = block[-SAMPLE_TAIL_CHARS:] if len(block) > SAMPLE_TAIL_CHARS else ""
        mid_idx = len(block) // 2
        mid_len = 300
        mid_start = max(0, mid_idx - (mid_len // 2))
        mid = block[mid_start : mid_start + mid_len]
        parts.extend([head, mid, tail, "\n\n"])

    ctx = "".join(parts)
    if len(ctx) > MAX_TOPIC_CONTEXT_CHARS:
        ctx = ctx[:MAX_TOPIC_CONTEXT_CHARS]
    return ctx

def build_questions_requests_balanced(topic_map: Dict, chunks: List[str]) -> List[Dict]:
    reqs: List[Dict] = []
    per_topic_type_counts: Dict[tuple, int] = {}

    for unit in topic_map["units"]:
        u = unit["unit_id"]
        for t in unit["topics"]:
            start, end = t["chunk_span"]
            ctx = _sample_topic_context(chunks, start, end)

            for force_type in QUESTION_TYPES_PER_TOPIC:
                ft = force_type.strip().lower()
                key = (t["topic_id"], ft)
                per_topic_type_counts[key] = per_topic_type_counts.get(key, 0) + 1
                seq = per_topic_type_counts[key]

                user = FORCE_TYPE_USER.format(
                    force_type=ft,
                    unit_id=u,
                    topic_id=t["topic_id"],
                    title=t["title"],
                    summary=t["summary"],
                    context_text=ctx,
                    schema_item_shape=json.dumps(SCHEMA_ITEM_SHAPE, ensure_ascii=False),
                    type_examples=json.dumps(TYPE_EXAMPLES, ensure_ascii=False),
                )

                custom_id = f"q_{u}_{t['topic_id']}_{ft}_{seq:02d}"

                reqs.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": MODEL_QUESTIONS,
                        "input": [
                            {"role": "system", "content": QUESTIONS_SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "text": {"format": {"type": "json_object"}},
                        # omit temperature for compatibility with some models in /v1/responses
                        "max_output_tokens": 1600,
                    },
                })
    return reqs