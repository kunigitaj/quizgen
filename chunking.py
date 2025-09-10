# chunking.py
import re
from typing import List
from pathlib import Path

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def chunk_preview(chunks, head_lines=10):
    preview = []
    for idx, ch in enumerate(chunks):
        head = "\n".join(ch.strip().splitlines()[:head_lines])
        preview.append({"index": idx, "text": head})
    return preview

# Visual separators preserved in your source
SEP_THREE = re.compile(r"^\s*\.{3}\s*$")       # '...'
SEP_SIX = re.compile(r"^\s*\.{6,}\s*$")        # '......' or more
UNDERLINE_ROW = re.compile(r"^_{5,}\s*$")

# Headings that should start new sections
# NOTE: We intentionally do NOT split on plain 'Objective' anymore.
HEADING = re.compile(
    r"^\s*("
    r"Unit\s+\d+|Knowledge quiz$|"
    r"Getting Started.*|Working with Development Spaces$|"
    r"Creating a New Project.*|Importing an Existing Project.*|"
    r"Navigating SAP Business Application Studio$|"
    r"Development Spaces in SAP Business Application Studio$|"
    r"Projects in SAP Business Application Studio$|"
    r"Interface of SAP Business Application Studio$|"
    r"SAP Business Application Studio$"
    r")\s*$",
    re.IGNORECASE
)

def semantic_chunks(text: str, max_chars: int = 7000, soft_min: int = 1000) -> List[str]:
    """
    Three-phase chunking:
      1) Strong splits on long dot rows ('......') and underline rows.
      2) Sub-split on '...' and known headings.
         - Headings are *kept with* the content that follows (no micro-chunks).
         - '...' lines are dropped (they just act as soft separators).
      3) Pack blocks into chunks up to max_chars, avoiding tiny fragments
         by ensuring a soft minimum length (soft_min) before flushing.

    Tip: We pass soft_min = max(OVERLAP, 600) from the caller to avoid tiny chunks.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]

    # ---------- Phase 1: strong splits (unit/end markers) ----------
    sections, cur = [], []
    for ln in lines:
        if SEP_SIX.match(ln) or UNDERLINE_ROW.match(ln):
            if cur:
                sections.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(ln)
    if cur:
        sections.append("\n".join(cur).strip())

    # ---------- Phase 2: sub-split by '...' and headings ----------
    def sub_split(sec: str) -> List[str]:
        out: List[str] = []
        buf: List[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                out.append("\n".join(buf).strip())
                buf = []

        for ln in sec.splitlines():
            if SEP_THREE.match(ln):
                # Soft separator: end current buffer, but do NOT keep the '...' line
                flush_buf()
                continue

            if HEADING.match(ln):
                # Start a new section at heading: attach heading to the content that follows
                flush_buf()
                buf = [ln.strip()]
                continue

            buf.append(ln)

        # Tail
        flush_buf()
        # Remove any accidental empty blocks
        return [b for b in out if b and not SEP_THREE.match(b)]

    blocks: List[str] = []
    for sec in sections:
        blocks.extend(sub_split(sec))

    # ---------- Phase 3: pack into chunks (avoid tiny or overlong) ----------
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len, chunks
        if buf:
            chunks.append("\n".join(buf).strip())
            buf, buf_len = [], 0

    for b in blocks:
        bl = len(b)
        if buf_len == 0:
            buf.append(b)
            buf_len += bl + 1
            continue

        if buf_len + bl + 1 <= max_chars:
            buf.append(b)
            buf_len += bl + 1
        else:
            # If current buffer is still too small, flush it early (except if it only has the last block)
            if buf_len < soft_min and len(buf) > 1:
                last = buf.pop()
                buf_len -= (len(last) + 1)
                flush()
                buf.append(last)
                buf_len = len(last) + 1
            else:
                flush()
                buf.append(b)
                buf_len = bl + 1

    flush()
    return chunks