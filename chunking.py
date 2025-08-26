# chunking.py

from pathlib import Path
from typing import List

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def char_chunks(text: str, max_chars=7000, overlap=700) -> List[str]:
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j])
        i = max(j - overlap, j)
    return out

def chunk_preview(chunks: List[str], head_lines=10):
    preview = []
    for idx, ch in enumerate(chunks):
        head = "\n".join(ch.strip().splitlines()[:head_lines])
        preview.append({"index": idx, "text": head})
    return preview