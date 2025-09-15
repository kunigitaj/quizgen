# prompts.py

TOPIC_MAP_SYSTEM = """You are an expert instructional designer.
From ordered chunks of a single document, produce a compact topic map that fully covers the material.

OUTPUT CONTRACT (STRICT):
- Return ONLY a single JSON object. No prose, no code fences, no markdown.
- The FIRST character must be '{' and the LAST character must be '}'.

SCHEMA:
{
  "schema_version": "1.0",
  "units": [
    {
      "unit_id": "u1",
      "title": "Unit Title",
      "topics": [
        {"topic_id": "u1_t1_slug", "title": "Topic Title", "summary": "1-2 sentences",
         "chunk_span": [start_idx, end_idx]}
      ]
    }
  ]
}

HARD CONSTRAINTS (must all be satisfied):
- You will be given N (total_chunks) and the valid chunk indices 0..N-1 in the user message.
- The UNION of all topic chunk_span ranges MUST cover EVERY index in 0..N-1 with no gaps and no overlaps.
- chunk_span uses inclusive integer indices and MUST stay within [0, N-1].
- Enforce start_idx <= end_idx for every chunk_span.
- Respect semantic boundaries: headings and ellipsis separators have been preserved in chunking; avoid splitting a topic across unrelated sections.
- Titles must be concise and specific; prefer 6–15 topics for a ~3,000-line text; add more units if needed.
- If a chunk contains only boilerplate (e.g., “Objective”, “Note”, “Demo”), roll it into the nearest relevant topic so coverage remains continuous.
- If resolving a boundary ambiguity would cause either a gap or an overlap, EXPAND the earlier topic’s end by one index to absorb the boundary so coverage remains continuous and non-overlapping.
"""

TOPIC_MAP_USER = """Create the topic map from these CHUNKS (index + first lines shown).

TOTAL_CHUNKS (N): {total_chunks}
VALID INDICES: 0..{last_index}
EXPLICIT INDEX LIST: [{all_indices}]

Guidance:
- These chunks are aligned to lesson/section boundaries using '...' (minor breaks) and '......' (unit ends).
- Group related chunks into coherent topics; keep each topic's span contiguous.
- Prefer unit titles like "Unit 1 – Preparing the Modeling Environment" and topic titles like "Getting Started with BAS", "Development Spaces", etc.

CHUNKS PREVIEW:
{chunks_preview}

Return ONLY the JSON object described by the system message (no extra text)."""

# Keep your compact shape as a schema anchor:
# (Show 5 options A–E to bias MCQ/MSQ generations to five choices.)
SCHEMA_ITEM_SHAPE = {
  "id": "q_XXXX",
  "type": "mcq|msq|tf",
  "unit_id": "uX",
  "topic_id": "uX_tY_slug",
  "question_rich": [{"type":"paragraph","children":[{"text":"..."}]}],
  "context_rich": [{"type":"callout","variant":"info","children":[{"type":"paragraph","children":[{"text":"..."}]}]}],
  "choices": [
    {"id":"A","text_rich":[{"type":"paragraph","children":[{"text":"..."}]}], "is_correct": True,
     "rationale_rich":[{"type":"paragraph","children":[{"text":"..."}]}]},
    {"id":"B","text_rich":[{"type":"paragraph","children":[{"text":"..."}]}], "is_correct": False,
     "rationale_rich":[{"type":"paragraph","children":[{"text":"..."}]}]},
    {"id":"C","text_rich":[{"type":"paragraph","children":[{"text":"..."}]}], "is_correct": False,
     "rationale_rich":[{"type":"paragraph","children":[{"text":"..."}]}]},
    {"id":"D","text_rich":[{"type":"paragraph","children":[{"text":"..."}]}], "is_correct": False,
     "rationale_rich":[{"type":"paragraph","children":[{"text":"..."}]}]},
    {"id":"E","text_rich":[{"type":"paragraph","children":[{"text":"..."}]}], "is_correct": False,
     "rationale_rich":[{"type":"paragraph","children":[{"text":"..."}]}]}
  ],
  "difficulty": 2,
  "tags": ["..."],
  "concept_tags": ["..."],
  "context_tags": ["..."],
  "hints_rich": [{"type":"callout","variant":"tip","children":[{"type":"paragraph","children":[{"text":"..."}]}]}],
  "mnemonic_rich": [{"type":"paragraph","children":[{"text":"..."}]}],
  "explanation_rich": [{"type":"paragraph","children":[{"text":"..."}]}],
  "elaboration_prompts_rich": [{"type":"paragraph","children":[{"text":"..."}]}],
  "shuffle": True,
  "grading": {"mode":"mcq","partial_credit": False, "penalty": 0, "require_all_correct": False},
  "example_rich": [{"type":"paragraph","children":[{"text":"..."}]}]
}

TYPE_EXAMPLES = {
  "msq": {
    "id":"q_EX_msq",
    "type":"msq",
    "unit_id":"uX",
    "topic_id":"uX_tY_slug",
    "question_rich":[{"type":"paragraph","children":[{"text":"Select all statements that align with the concept described."}]}],
    "context_rich":[{"type":"callout","variant":"info","children":[{"type":"paragraph","children":[{"text":"Background framing that does not reveal any specific correct statement."}]}]}],
    "choices":[
      {"id":"A","text_rich":[{"type":"paragraph","children":[{"text":"Correct facet 1"}]}],"is_correct":True,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Facet 1 is supported by the text."}]}]},
      {"id":"B","text_rich":[{"type":"paragraph","children":[{"text":"Correct facet 2"}]}],"is_correct":True,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Facet 2 is also supported by the text."}]}]},
      {"id":"C","text_rich":[{"type":"paragraph","children":[{"text":"Plausible but incorrect"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Conflicts with or is absent from the text."}]}]},
      {"id":"D","text_rich":[{"type":"paragraph","children":[{"text":"Irrelevant distractor"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Irrelevant to the concept."}]}]},
      {"id":"E","text_rich":[{"type":"paragraph","children":[{"text":"Overgeneralized claim"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Too broad; not supported by the text."}]}]}
    ],
    "difficulty":2,
    "tags":["example","msq"],
    "concept_tags":["example_concept"],
    "context_tags":["example_context"],
    "hints_rich":[
      {"type":"callout","variant":"tip","children":[{"type":"paragraph","children":[{"text":"Start by eliminating what is clearly unsupported."}]}]},
      {"type":"callout","variant":"tip","children":[{"type":"paragraph","children":[{"text":"Multiple correct facets may be present."}]}]}
    ],
    "mnemonic_rich":[{"type":"paragraph","children":[{"text":"Remember: multiple truths can coexist."}]}],
    "explanation_rich":[{"type":"paragraph","children":[{"text":"The concept includes facets 1 and 2; other options conflict with or are unsupported by the text."}]}],
    "elaboration_prompts_rich":[{"type":"paragraph","children":[{"text":"Which parts of the text support facet 1?"}]}],
    "shuffle":True,
    "grading":{"mode":"msq","partial_credit":True,"penalty":0,"require_all_correct":False},
    "example_rich":[{"type":"paragraph","children":[{"text":"A scenario where both facets apply."}]}]
  },
  "mcq": {
    "id":"q_EX_mcq",
    "type":"mcq",
    "unit_id":"uX",
    "topic_id":"uX_tY_slug",
    "question_rich":[{"type":"paragraph","children":[{"text":"Which option best completes the idea?"}]}],
    "context_rich":[{"type":"callout","variant":"info","children":[{"type":"paragraph","children":[{"text":"Neutral context framing without the actual answer."}]}]}],
    "choices":[
      {"id":"A","text_rich":[{"type":"paragraph","children":[{"text":"Correct answer"}]}],"is_correct":True,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Directly matches the text."}]}]},
      {"id":"B","text_rich":[{"type":"paragraph","children":[{"text":"Plausible but wrong"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Not supported by the text."}]}]},
      {"id":"C","text_rich":[{"type":"paragraph","children":[{"text":"Common misconception"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Contradicted by the text."}]}]},
      {"id":"D","text_rich":[{"type":"paragraph","children":[{"text":"Irrelevant detail"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Not relevant to the concept."}]}]},
      {"id":"E","text_rich":[{"type":"paragraph","children":[{"text":"Overly specific edge case"}]}],"is_correct":False,
       "rationale_rich":[{"type":"paragraph","children":[{"text":"Too narrow; not the best completion."}]}]}
    ],
    "difficulty":1,
    "tags":["example","mcq"],
    "concept_tags":["example_concept"],
    "context_tags":["example_context"],
    "hints_rich":[{"type":"callout","variant":"tip","children":[{"type":"paragraph","children":[{"text":"Focus on the precise phrasing used."}]}]}],
    "mnemonic_rich":[{"type":"paragraph","children":[{"text":"One best answer stands out when aligned with definitions."}]}],
    "explanation_rich":[{"type":"paragraph","children":[{"text":"The correct option aligns with the key definition in the text."}]}],
    "elaboration_prompts_rich":[{"type":"paragraph","children":[{"text":"What makes the correct option better than the plausible one?"}]}],
    "shuffle":True,
    "grading":{"mode":"mcq","partial_credit":False,"penalty":0,"require_all_correct":False},
    "example_rich":[{"type":"paragraph","children":[{"text":"A brief application example."}]}]
  },
  "tf": {
    "id":"q_EX_tf",
    "type":"tf",
    "unit_id":"uX",
    "topic_id":"uX_tY_slug",
    "question_rich":[{"type":"paragraph","children":[{"text":"The statement below is accurate according to the text."}]}],
    "context_rich":[{"type":"callout","variant":"info","children":[{"type":"paragraph","children":[{"text":"Provide context but do not reveal whether the statement is true or false."}]}]}],
    "choices":[
      {"id":"A","text_rich":[{"type":"paragraph","children":[{"text":"True"}]}],"is_correct":True},
      {"id":"B","text_rich":[{"type":"paragraph","children":[{"text":"False"}]}],"is_correct":False}
    ],
    "difficulty":1,
    "tags":["example","tf"],
    "concept_tags":["example_concept"],
    "context_tags":["example_context"],
    "hints_rich":[{"type":"callout","variant":"tip","children":[{"type":"paragraph","children":[{"text":"Re-read the relevant definition carefully."}]}]}],
    "mnemonic_rich":[{"type":"paragraph","children":[{"text":"Check the exact wording vs. the text."}]}],
    "explanation_rich":[{"type":"paragraph","children":[{"text":"Why the statement is true (or false) per the text."}]}],
    "elaboration_prompts_rich":[{"type":"paragraph","children":[{"text":"Rephrase the statement to make it true if it is false (or vice versa)."}]}],
    "shuffle":False,
    "grading":{"mode":"mcq","partial_credit":False,"penalty":0,"require_all_correct":False},
    "example_rich":[{"type":"paragraph","children":[{"text":"Short example tied to the statement."}]}]
  }
}

QUESTIONS_SYSTEM = """You are a senior assessment author for college-level learners.
Produce EXACTLY ONE quiz question that STRICTLY follows the provided JSON schema (a single question object).
Requirements:
- The question TYPE must match the user instruction (msq, mcq, or tf).
- Use ONLY the supplied context text; do not invent facts.
- context_rich MUST be neutral and MUST NOT reveal the answer (no verbatim answer strings).
- For MCQ/MSQ provide EXACTLY FIVE choices labeled A–E.
- For MCQ mark EXACTLY ONE option as correct.
- For MSQ mark TWO to THREE options as correct; the rest must be plausible distractors.
- Provide a brief rationale for EVERY choice (MSQ/MCQ).
- Difficulty is 1–3.
- Fill EVERY field for the question (no empty lists).
- Use “True” and “False” labels for TF choices.
- Provide 2–3 hints in hints_rich (each a 'tip' callout). Do not provide fewer than 2 hints.
- Output ONLY a JSON OBJECT for the single question (no array, no wrapper, no prose).

Follow these structural anchors:
1) SCHEMA ITEM SHAPE (fields and nesting).
2) TYPE EXAMPLES (minimal patterns for msq/mcq/tf). Use their structure, not their content.
"""

# =========================
# Study Summary (MAP / POLISH)
# =========================
SUMMARY_SYSTEM = """You are a senior instructional designer creating a concise, mobile-first study companion from ONE large source text.

STRICT SOURCING RULE:
- Use ONLY the content provided in the user message. Do NOT introduce topics, brands, or examples not present in the source. If a section header appears without body text, summarize only what is present or omit that subsection.

OUTPUT CONTRACT (STRICT):
- Return ONLY a single JSON object. No prose, no code fences, no markdown.
- The FIRST character must be '{' and the LAST character must be '}'.

SHAPE:
{
  "schema_version": "1.0",
  "narrativeSections": [
    {
      "title": "Section Title",
      "bullets": ["bullet 1", "bullet 2"],
      "subsections": [{"title": "Subsection Title", "bullets": ["point 1", "point 2"]}]
    }
  ],
  "slides": [
    {
      "title": "Slide Title",
      "subtitle": "Optional subtitle or null",
      "subheadings": [{"heading": "Label", "color": "blue.600", "content": ["point 1", "point 2"]}]
    }
  ]
}

REQUIREMENTS:
- schema_version must equal "1.0".
- Allowed colors: blue.600, green.600, amber.600, red.600, purple.600.
- Bullets: ≤ 18 words, sentence case, no trailing period.
- No empty arrays or empty strings.
- narrativeSections: target 4–12 items; each 2–6 bullets (subsections optional, each 2–5 bullets).
- slides: target 4–12; EACH slide MUST include subheadings (2–5 bullets per subheading).
"""

SUMMARY_USER = """Create study_summary.json from the following FULL SOURCE TEXT.
Return ONLY the JSON object (no prose, no code fences). Use ONLY the content below:

{full_text}
"""

SUMMARY_MAP_SYSTEM = """You are an instructional designer. Summarize ONE chunk into a StudySummary JSON object (same SHAPE and REQUIREMENTS as SUMMARY_SYSTEM).

STRICT SOURCING RULE:
- Use ONLY this chunk. Do NOT add external examples or unrelated technologies. If the chunk is mostly headings or boilerplate, synthesize brief, faithful bullets (2–4) without inventing new concepts.

OUTPUT CONTRACT (STRICT):
- Single valid JSON OBJECT.
- No prose or code fences.
- The FIRST character must be '{' and the LAST character must be '}'.
- Return ONLY these top-level keys: schema_version, narrativeSections, slides.
- schema_version MUST be "1.0".
- Ensure every slide has at least one subheading; if needed, synthesize "Key points" with 2–5 bullets drawn ONLY from the chunk.
"""

SUMMARY_MAP_USER = """Summarize this CHUNK into the StudySummary JSON object.
Return ONLY the JSON object (no prose; no code fences). Use ONLY this chunk:

{chunk_text}
"""

# (Kept for backwards compatibility with tree-reduce; not used in the simple flow)
SUMMARY_REDUCE_SYSTEM = """You are an instructional designer. Merge multiple StudySummary JSON objects into ONE final StudySummary (same SHAPE and REQUIREMENTS as SUMMARY_SYSTEM).

GOALS:
- De-duplicate and merge logically; preserve coverage across ALL input micros.
- Preserve clarity and mobile-first bullets.
- Keep allowed colors and bullet limits.
- No empty arrays/strings; schema_version = "1.0".
- Ensure EVERY slide includes subheadings (synthesize "Key points" if a source item lacked them).
- DO NOT introduce any content that is not present across the inputs.

OUTPUT CONTRACT (STRICT):
- Single valid JSON OBJECT. No prose or code fences.
- The FIRST character must be '{' and the LAST character must be '}'.
"""

SUMMARY_REDUCE_USER = """Merge these micro-summaries (JSON list) into the final StudySummary.
Return ONLY a single JSON object (no prose; no code fences). Use ONLY the content of the list:

{micro_json_list}
"""

# --- New: single-call polish prompt for the local-merge output ---
SUMMARY_POLISH_SYSTEM = """You are an instructional designer.
INPUT is ONE StudySummary JSON (schema_version "1.0" with narrativeSections[] and slides[]).

TASK: Return the SAME SHAPE, improved:
- Keep ONLY allowed keys and colors (blue.600, green.600, amber.600, red.600, purple.600).
- De-duplicate sections/slides by title; merge bullets (2–6 per section; 2–5 per subheading).
- Trim bullets to ≤ 18 words, sentence case, no trailing period.
- No empty strings/arrays; ensure every slide has ≥ 1 subheading with 2–5 bullets.
- Do NOT add content not already present; only consolidate, trim, and rephrase minimally.

OUTPUT CONTRACT (STRICT):
- Return ONLY a single JSON object. No prose, no code fences.
- The FIRST character must be '{' and the LAST character must be '}'.
"""

SUMMARY_POLISH_USER = """Polish this StudySummary JSON and enforce all constraints:

{merged_json}
"""