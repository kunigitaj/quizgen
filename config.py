# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- helpers ---
def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# --- Auth ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Models (gpt-5 only) ---
MODEL_TOPIC = os.getenv("MODEL_TOPIC", "gpt-5")
MODEL_QUESTIONS = os.getenv("MODEL_QUESTIONS", "gpt-5")
MODEL_SUMMARY = os.getenv("MODEL_SUMMARY", "gpt-5")

# --- Batch API ---
BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")

# --- Client-side polling/timeout controls ---
BATCH_POLL_SECONDS       = _get_int("BATCH_POLL_SECONDS", 15)        # how often to poll
BATCH_TIMEOUT_SECONDS    = _get_int("BATCH_TIMEOUT_SECONDS", 0)      # 0 = wait indefinitely
CANCEL_ON_TIMEOUT        = _get_bool("CANCEL_ON_TIMEOUT", False)     # cancel server job if we time out?
BATCH_STATUS_LOG_SECONDS = _get_int("BATCH_STATUS_LOG_SECONDS", 60)  # status print cadence

# --- Chunking ---
MAX_CHARS = _get_int("MAX_CHARS", 7000)
OVERLAP = _get_int("OVERLAP", 700)
MAX_CONTEXT_SCAN_LEN = _get_int("MAX_CONTEXT_SCAN_LEN", 8000)

# --- Question volume / distribution ---
QUESTION_TYPES_PER_TOPIC = os.getenv(
    "QUESTION_TYPES_PER_TOPIC",
    "msq,mcq,mcq,tf"
).split(",")
QUESTION_TYPE_MULTIPLIER = _get_int("QUESTION_TYPE_MULTIPLIER", 3)

# --- Taxonomy ---
TAXONOMY_VERSION = os.getenv("TAXONOMY_VERSION", "")

# --- Batch sharding for questions ---
BATCH_QUESTIONS_SHARD_SIZE = _get_int("BATCH_QUESTIONS_SHARD_SIZE", 24)
BATCH_QUESTIONS_MAX_BYTES  = _get_int("BATCH_QUESTIONS_MAX_BYTES", 0)  # 0 = ignore

# --- Context sampling per topic ---
MAX_TOPIC_CONTEXT_CHARS = _get_int("MAX_TOPIC_CONTEXT_CHARS", 45000)
CONTEXT_SAMPLE_WINDOWS = _get_int("CONTEXT_SAMPLE_WINDOWS", 5)
SAMPLE_HEAD_CHARS = _get_int("SAMPLE_HEAD_CHARS", 500)
SAMPLE_TAIL_CHARS = _get_int("SAMPLE_TAIL_CHARS", 500)

# --- Reasoning controls (GPT-5: minimal | low | medium | high) ---
REASONING_TOPIC_EFFORT     = os.getenv("REASONING_TOPIC_EFFORT", "minimal")
REASONING_QUESTIONS_EFFORT = os.getenv("REASONING_QUESTIONS_EFFORT", "minimal")
REASONING_SUMMARY_EFFORT   = os.getenv("REASONING_SUMMARY_EFFORT", "minimal")

# --- Verbosity controls (GPT-5: low | medium | high) ---
VERBOSITY_TOPIC     = os.getenv("VERBOSITY_TOPIC", "low")
VERBOSITY_QUESTIONS = os.getenv("VERBOSITY_QUESTIONS", "low")
VERBOSITY_SUMMARY   = os.getenv("VERBOSITY_SUMMARY", "low")

# --- Max output tokens per stage ---
TOPIC_MAX_OUTPUT_TOKENS     = _get_int("TOPIC_MAX_OUTPUT_TOKENS", 6000)
QUESTIONS_MAX_OUTPUT_TOKENS = _get_int("QUESTIONS_MAX_OUTPUT_TOKENS", 1600)
SUMMARY_MAX_OUTPUT_TOKENS   = _get_int("SUMMARY_MAX_OUTPUT_TOKENS", 2000)

# --- Hierarchical / staged summary caps ---
SUMMARY_CHUNK_MAX_CHARS           = _get_int("SUMMARY_CHUNK_MAX_CHARS", 7000)   # per-map chunk size
SUMMARY_CHUNK_OVERLAP             = _get_int("SUMMARY_CHUNK_OVERLAP", 600)      # semantic overlap
# Stage-1 (micro summaries) — slightly higher than before for safety
SUMMARY_STAGE1_MAX_OUTPUT_TOKENS  = _get_int("SUMMARY_STAGE1_MAX_OUTPUT_TOKENS", 1100)
# Final polish cap
SUMMARY_REDUCE_MAX_OUTPUT_TOKENS  = _get_int("SUMMARY_REDUCE_MAX_OUTPUT_TOKENS", 5000)
# Only used by tree reduce (kept for backwards compatibility)
SUMMARY_REDUCE_FANIN              = _get_int("SUMMARY_REDUCE_FANIN", 12)

# --- Summary batching knobs (used by summary.py’s multi-item shards) ---
SUMMARY_BATCH_SHARD_SIZE = _get_int("SUMMARY_BATCH_SHARD_SIZE", 24)   # requests per shard
SUMMARY_BATCH_MAX_BYTES  = _get_int("SUMMARY_BATCH_MAX_BYTES", 0)     # 0 = ignore byte limit

# --- Summary logging/debug flags ---
SUMMARY_LOG_LEVEL     = os.getenv("SUMMARY_LOG_LEVEL", "INFO")
SUMMARY_LOG_TIMESTAMP = "1" if _get_bool("SUMMARY_LOG_TIMESTAMP", False) else "0"
# Optional debug/convenience toggles (read by other modules if desired)
SUMMARY_USE_BATCH      = _get_bool("SUMMARY_USE_BATCH", True)
SUMMARY_DEBUG_MAX_CHUNKS = os.getenv("SUMMARY_DEBUG_MAX_CHUNKS")  # leave as string if provided
# Modes: "", "local_merge_polish", "concat" etc.
SUMMARY_MODE             = os.getenv("SUMMARY_MODE", "local_merge_polish")

# --- Summary retries (optional; read directly by summary.py if set) ---
SUMMARY_MAP_RETRY_MISSING    = _get_int("SUMMARY_MAP_RETRY_MISSING", 1)
SUMMARY_REDUCE_RETRY_MISSING = _get_int("SUMMARY_REDUCE_RETRY_MISSING", 2)

# (Also optional, for completeness if you want them surfaced here)
SUMMARY_LOG_COMPACT          = _get_bool("SUMMARY_LOG_COMPACT", True)
SUMMARY_PROGRESS_LOG_EVERY   = _get_int("SUMMARY_PROGRESS_LOG_EVERY", 1)