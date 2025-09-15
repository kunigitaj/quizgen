# topic_map.py

import json
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    OPENAI_API_KEY,
    MODEL_TOPIC,
    TOPIC_MAX_OUTPUT_TOKENS,
    REASONING_TOPIC_EFFORT,
    VERBOSITY_TOPIC,
)
from prompts import TOPIC_MAP_SYSTEM, TOPIC_MAP_USER

client = OpenAI(api_key=OPENAI_API_KEY)


def _response_text_or_first_json(r) -> str:
    """
    Prefer the SDK convenience field output_text when present.
    If missing, try to serialize the first json/object content block.
    Returns a JSON string (not parsed) so callers can decide what to do.
    """
    if getattr(r, "output_text", None):
        return r.output_text

    # Fallback: scan output content for a json/object block and dump it
    for item in (getattr(r, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            ctype = (getattr(c, "type", "") or "").lower()
            if ctype in ("json", "object") and getattr(c, "json", None) is not None:
                try:
                    return json.dumps(c.json, ensure_ascii=False)
                except Exception:
                    pass

    # Last resort: represent empty object
    return "{}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def create_topic_map(chunks_preview):
    """
    One-shot (non-batch) topic-map creation using the Responses API.
    Returns a JSON string.
    """
    total = len(chunks_preview)
    usr_content = TOPIC_MAP_USER.format(
        total_chunks=total,
        last_index=total - 1,
        all_indices=", ".join(str(i) for i in range(total)),
        chunks_preview=json.dumps(chunks_preview, ensure_ascii=False),
    )

    sys = {"role": "system", "content": TOPIC_MAP_SYSTEM}
    usr = {"role": "user", "content": usr_content}

    kwargs = {
        "model": MODEL_TOPIC,
        "input": [sys, usr],
        "text": {
            "format": {"type": "json_object"},
            "verbosity": VERBOSITY_TOPIC,
        },
        "max_output_tokens": TOPIC_MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": REASONING_TOPIC_EFFORT},
        # NOTE: Do not pass temperature/top_p for GPT-5 reasoning models.
    }

    r = client.responses.create(**kwargs)
    return _response_text_or_first_json(r)


def save_topicmap_batch_input(preview, out_path: Path):
    """
    Write a single JSONL line suitable for OpenAI Batch /v1/responses.
    The created item requests a topic map for the provided preview.
    """
    total = len(preview)
    usr_content = TOPIC_MAP_USER.format(
        total_chunks=total,
        last_index=total - 1,
        all_indices=", ".join(str(i) for i in range(total)),
        chunks_preview=json.dumps(preview, ensure_ascii=False),
    )

    body = {
        "model": MODEL_TOPIC,
        "input": [
            {"role": "system", "content": TOPIC_MAP_SYSTEM},
            {"role": "user", "content": usr_content},
        ],
        "text": {
            "format": {"type": "json_object"},
            "verbosity": VERBOSITY_TOPIC,
        },
        "max_output_tokens": TOPIC_MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": REASONING_TOPIC_EFFORT},
        # No temperature/top_p here either.
    }

    line = {
        "custom_id": "topicmap_0001",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }

    out_path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")