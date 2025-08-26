# topic_map.py

import json
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENAI_API_KEY, MODEL_TOPIC
from prompts import TOPIC_MAP_SYSTEM, TOPIC_MAP_USER

client = OpenAI(api_key=OPENAI_API_KEY)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def create_topic_map(chunks_preview):
    """
    Optional synchronous helper to sanity-check prompt shape.
    Forces a single JSON message via text.format={type:"json_object"}.
    """
    total = len(chunks_preview)
    usr_content = TOPIC_MAP_USER.format(
        total_chunks=total,
        last_index=total - 1,
        all_indices=", ".join(str(i) for i in range(total)),
        chunks_preview=json.dumps(chunks_preview, ensure_ascii=False)
    )

    sys = {"role": "system", "content": TOPIC_MAP_SYSTEM}
    usr = {"role": "user", "content": usr_content}

    r = client.responses.create(
        model=MODEL_TOPIC,
        input=[sys, usr],
        text={"format": {"type": "json_object"}},
        # temperature omitted for compatibility with some models in /v1/responses
        max_output_tokens=3000,
    )
    return r.output_text

def save_topicmap_batch_input(preview, out_path: Path):
    """
    Build a single-line JSONL for OpenAI Batch (/v1/responses) that requests the topic map.
    Forces JSON output via text.format={type:"json_object"}.
    """
    total = len(preview)
    usr_content = TOPIC_MAP_USER.format(
        total_chunks=total,
        last_index=total - 1,
        all_indices=", ".join(str(i) for i in range(total)),
        chunks_preview=json.dumps(preview, ensure_ascii=False)
    )

    body = {
        "model": MODEL_TOPIC,
        "input": [
            {"role": "system", "content": TOPIC_MAP_SYSTEM},
            {"role": "user", "content": usr_content},
        ],
        "text": {"format": {"type": "json_object"}},
        # no `reasoning` here – only send it to reasoning models
        "max_output_tokens": 3000,
    }

    line = {
        "custom_id": "topicmap_0001",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }

    out_path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")