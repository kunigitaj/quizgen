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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def create_topic_map(chunks_preview):
    total = len(chunks_preview)
    usr_content = TOPIC_MAP_USER.format(
        total_chunks=total,
        last_index=total - 1,
        all_indices=", ".join(str(i) for i in range(total)),
        chunks_preview=json.dumps(chunks_preview, ensure_ascii=False)
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
        # NOTE: GPT-5 reasoning models do not support temperature/top_p.
    }

    r = client.responses.create(**kwargs)
    return r.output_text


def save_topicmap_batch_input(preview, out_path: Path):
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