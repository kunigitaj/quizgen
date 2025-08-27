# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- Auth ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Models (gpt-5 only) ---
MODEL_TOPIC = os.getenv("MODEL_TOPIC", "gpt-5")
MODEL_QUESTIONS = os.getenv("MODEL_QUESTIONS", "gpt-5")

# --- Batch API ---
BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")

# --- Chunking ---
MAX_CHARS = int(os.getenv("MAX_CHARS", "7000"))
OVERLAP = int(os.getenv("OVERLAP", "700"))
MAX_CONTEXT_SCAN_LEN = int(os.getenv("MAX_CONTEXT_SCAN_LEN", "8000"))

# --- Parallelism (used by your driver; not directly by API) ---
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "6"))

# --- Question volume / distribution ---
# Order matters; this list is repeated QUESTION_TYPE_MULTIPLIER times per topic.
QUESTION_TYPES_PER_TOPIC = os.getenv(
    "QUESTION_TYPES_PER_TOPIC",
    "msq,mcq,mcq,tf"
).split(",")
# Multiplier: repeat the above mix N times per topic
QUESTION_TYPE_MULTIPLIER = int(os.getenv("QUESTION_TYPE_MULTIPLIER", "3"))

# --- Taxonomy ---
TAXONOMY_VERSION = os.getenv("TAXONOMY_VERSION", "")

# --- Batch sharding for questions ---
BATCH_QUESTIONS_SHARD_SIZE = int(os.getenv("BATCH_QUESTIONS_SHARD_SIZE", "24"))
BATCH_QUESTIONS_MAX_BYTES = int(os.getenv("BATCH_QUESTIONS_MAX_BYTES", "0"))

# --- Context sampling per topic ---
MAX_TOPIC_CONTEXT_CHARS = int(os.getenv("MAX_TOPIC_CONTEXT_CHARS", "45000"))
CONTEXT_SAMPLE_WINDOWS = int(os.getenv("CONTEXT_SAMPLE_WINDOWS", "5"))
SAMPLE_HEAD_CHARS = int(os.getenv("SAMPLE_HEAD_CHARS", "500"))
SAMPLE_TAIL_CHARS = int(os.getenv("SAMPLE_TAIL_CHARS", "500"))

# --- Reasoning controls ---
# GPT-5 valid: minimal | low | medium | high
REASONING_AUTO = False  # keep question generation deterministic wrt message outputs
REASONING_TOPIC_EFFORT = os.getenv("REASONING_TOPIC_EFFORT", "minimal")
REASONING_QUESTIONS_EFFORT = os.getenv("REASONING_QUESTIONS_EFFORT", "minimal")

# --- Verbosity controls (new) ---
# GPT-5 valid: low | medium | high
VERBOSITY_TOPIC = os.getenv("VERBOSITY_TOPIC", "low")
VERBOSITY_QUESTIONS = os.getenv("VERBOSITY_QUESTIONS", "low")

# --- Max output tokens per stage ---
TOPIC_MAX_OUTPUT_TOKENS = int(os.getenv("TOPIC_MAX_OUTPUT_TOKENS", "6000"))
QUESTIONS_MAX_OUTPUT_TOKENS = int(os.getenv("QUESTIONS_MAX_OUTPUT_TOKENS", "1600"))