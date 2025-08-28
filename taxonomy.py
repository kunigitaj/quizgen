# taxonomy.py

import json
import re
from datetime import date
from typing import Dict, List, Any
from pathlib import Path
from config import TAXONOMY_VERSION
from schema_models import Taxonomy, TaxonomyUnit, TaxonomyTopic, TaxonomyTagEntry


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "tag"


def _dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _smart_title(s: str) -> str:
    """
    Humanize labels so they look like normal text, not code-y slugs.
    Examples:
      "customer-collaboration" -> "Customer Collaboration"
      "instruction guides" -> "Instruction Guides"
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return s
    # Replace separators with spaces
    s = re.sub(r"[_\-\/]+", " ", s)
    # Collapse multi-space
    s = re.sub(r"\s{2,}", " ", s)
    # Lowercase common small words but keep leading word capitalized by title()
    titled = s.title()
    # Optionally protect specific all-caps acronyms (simple heuristic)
    for acro in ("API", "HTTP", "SQL", "GPU", "CPU", "UI", "UX"):
        titled = re.sub(rf"\b{acro.title()}\b", acro, titled)
    return titled


def _make_tag_entries(values: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for v in _dedupe_preserve_order([x for x in values if isinstance(x, str) and x.strip()]):
        label_human = _smart_title(v)
        slug = _slug(v)
        aliases = _dedupe_preserve_order(
            [
                v,
                v.lower(),
                v.title(),
                label_human,
                slug.replace("_", " "),
                slug.replace("_", "-"),
            ]
        )
        entries.append(
            {
                "id": slug,
                "label": label_human,
                "aliases": aliases,
                "description": "",
            }
        )
    return entries


def build_taxonomy(topic_map: Dict, questions: List[Dict]) -> Dict[str, Any]:
    # Version: env override or today
    version = TAXONOMY_VERSION.strip() or date.today().isoformat()

    # Units
    units: List[Dict[str, Any]] = []
    for u in topic_map.get("units", []):
        uid = u.get("unit_id")
        label = _smart_title(u.get("title", uid))
        topic_titles = [_smart_title(t.get("title", "")) for t in u.get("topics", [])]
        desc = "Includes topics: " + ", ".join(topic_titles[:6]) + ("..." if len(topic_titles) > 6 else "")
        units.append(TaxonomyUnit(id=uid, label=label, description=desc).model_dump())

    # Topics
    topics: List[Dict[str, Any]] = []
    for u in topic_map.get("units", []):
        for t in u.get("topics", []):
            tid = t.get("topic_id")
            topics.append(
                TaxonomyTopic(
                    id=tid,
                    label=_smart_title(t.get("title", tid)),
                    description=t.get("summary", ""),
                ).model_dump()
            )

    # Gather tag vocabularies from questions
    tags_raw, concept_raw, context_raw = [], [], []
    for q in questions:
        tags_raw.extend(q.get("tags", []) or [])
        concept_raw.extend(q.get("concept_tags", []) or [])
        context_raw.extend(q.get("context_tags", []) or [])

    taxonomy_dict = {
        "version": version,
        "units": units,
        "topics": topics,
        "tags": _make_tag_entries(tags_raw),
        "concept_tags": _make_tag_entries(concept_raw),
        "context_tags": _make_tag_entries(context_raw),
    }

    # Validate via Pydantic
    Taxonomy(**taxonomy_dict)
    return taxonomy_dict


def write_taxonomy(tax: Dict[str, Any], out_path: Path):
    out_path.write_text(json.dumps(tax, ensure_ascii=False, indent=2), encoding="utf-8")