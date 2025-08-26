# config.py

import os

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MODEL_TOPIC = os.getenv("MODEL_TOPIC", "gpt-5")
MODEL_QUESTIONS = os.getenv("MODEL_QUESTIONS", "gpt-5")

BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")

MAX_CHARS = int(os.getenv("MAX_CHARS", "7000"))
OVERLAP = int(os.getenv("OVERLAP", "700"))

MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "6"))

MAX_CONTEXT_SCAN_LEN = 8000

# --- Question volume / distribution ---
# Comma-separated types generated per topic (default: 4 per topic → msq + 2x mcq + tf)
QUESTION_TYPES_PER_TOPIC = os.getenv("QUESTION_TYPES_PER_TOPIC", "msq,mcq,mcq,tf").split(",")

# --- Taxonomy ---
# Optional version label for taxonomy.json; defaults to today's date if empty
TAXONOMY_VERSION = os.getenv("TAXONOMY_VERSION", "")

# --- Batch sharding (prevents huge JSONL uploads from failing validation) ---
# Max requests per batch file (tune down if you still see 'failed' with no files)
BATCH_QUESTIONS_SHARD_SIZE = int(os.getenv("BATCH_QUESTIONS_SHARD_SIZE", "24"))

# Optional maximum combined bytes per shard file (0 = ignore). If both are set,
# the shard will break as soon as either the request count or byte budget is hit.
BATCH_QUESTIONS_MAX_BYTES = int(os.getenv("BATCH_QUESTIONS_MAX_BYTES", "0"))

# --- Context sampling for each topic (keeps requests small & representative) ---
# Hard character cap per question's CONTEXT payload after sampling
MAX_TOPIC_CONTEXT_CHARS = int(os.getenv("MAX_TOPIC_CONTEXT_CHARS", "45000"))

# How many equal "windows" to sample across a topic's chunk span (>=1)
CONTEXT_SAMPLE_WINDOWS = int(os.getenv("CONTEXT_SAMPLE_WINDOWS", "5"))

# Head/Tail characters per window (rest is mid-sample)
SAMPLE_HEAD_CHARS = int(os.getenv("SAMPLE_HEAD_CHARS", "500"))
SAMPLE_TAIL_CHARS = int(os.getenv("SAMPLE_TAIL_CHARS", "500"))