# run_batch.py

import time
from pathlib import Path
from typing import Optional
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    BATCH_COMPLETION_WINDOW,
    BATCH_POLL_SECONDS,
    BATCH_TIMEOUT_SECONDS,
    BATCH_STATUS_LOG_SECONDS,
    CANCEL_ON_TIMEOUT,
)

client = OpenAI(api_key=OPENAI_API_KEY)

def submit_batch(input_jsonl: Path):
    uploaded = client.files.create(file=open(input_jsonl, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=BATCH_COMPLETION_WINDOW,
    )
    return batch.id

def retrieve_batch(batch_id: str):
    return client.batches.retrieve(batch_id)

def cancel_batch(batch_id: str):
    """Explicit helper to cancel a running batch."""
    try:
        client.batches.cancel(batch_id)
        print(f"[Batch] Sent cancel for {batch_id}.")
    except Exception as e:
        print(f"[Batch] Cancel attempt failed for {batch_id}: {e}")

def wait_for_batch(batch_id: str):
    """
    Poll until the batch hits a terminal state.
    Honors config:
      - BATCH_POLL_SECONDS: seconds between polls
      - BATCH_STATUS_LOG_SECONDS: print status every N seconds
      - BATCH_TIMEOUT_SECONDS: 0 == wait indefinitely
      - CANCEL_ON_TIMEOUT: if True and we do time out (>0), attempt server-side cancel
    """
    start = time.time()
    last_log = 0.0

    while True:
        st = retrieve_batch(batch_id)
        if st.status in ("completed", "failed", "cancelled", "expired", "cancelling"):
            return st

        now = time.time()
        if now - last_log > max(5, BATCH_STATUS_LOG_SECONDS):
            rc = getattr(st, "request_counts", None)
            print(f"[Batch] status={st.status} request_counts={rc}")
            last_log = now

        # If timeout is 0, wait forever
        if BATCH_TIMEOUT_SECONDS > 0 and (now - start) > BATCH_TIMEOUT_SECONDS:
            msg = f"Batch {batch_id} did not complete within {BATCH_TIMEOUT_SECONDS}s"
            if CANCEL_ON_TIMEOUT:
                cancel_batch(batch_id)
            raise TimeoutError(msg)

        time.sleep(max(1, BATCH_POLL_SECONDS))

def _download_file(file_id: str, out_path: Path):
    content = client.files.content(file_id).read().decode("utf-8")
    out_path.write_text(content, encoding="utf-8")
    return out_path

def download_output_or_error(batch_obj, out_path: Path, err_path: Optional[Path] = None):
    """
    If output_file_id exists, write it to out_path and return ("output", path).
    Else if error_file_id exists, write it to err_path (or out_path.with_suffix('.errors.jsonl')) and return ("error", path).
    Else raise with details.
    """
    if getattr(batch_obj, "output_file_id", None):
        return ("output", _download_file(batch_obj.output_file_id, out_path))

    if getattr(batch_obj, "error_file_id", None):
        if err_path is None:
            err_path = out_path.with_suffix(".errors.jsonl")
        return ("error", _download_file(batch_obj.error_file_id, err_path))

    rc = getattr(batch_obj, "request_counts", None)
    raise RuntimeError(
        f"Batch finished without output or error file. Status={batch_obj.status}, request_counts={rc}"
    )