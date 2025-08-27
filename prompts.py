# prompts.py

TOPIC_MAP_SYSTEM = """You are an expert instructional designer.
From ordered chunks of a single document, produce a compact topic map that fully covers the material.

Return ONLY valid JSON with:
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
- The UNION of all topic chunk_span ranges MUST cover EVERY index in 0..N-1 with **no gaps and no overlaps**.
- chunk_span uses inclusive integer indices and MUST stay within [0, N-1].
- If some chunks are minor/boilerplate, still include them in a small topic (e.g., “Misc/Glue”) so nothing is uncovered.
- Prefer 6–15 topics per unit for a ~3,000-line text; add more units if needed to achieve complete coverage.
- Titles should be concise and specific.
"""

TOPIC_MAP_USER = """Create the topic map from these CHUNKS (index + first lines shown).

TOTAL_CHUNKS (N): {total_chunks}
VALID INDICES: 0..{last_index}
EXPLICIT INDEX LIST: [{all_indices}]

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

QUESTIONS_USER = """Create questions for:
unit_id: {unit_id}
topic_id: {topic_id}
title: {title}
summary: {summary}

CONTEXT (concatenated chunks for this topic):
{context_text}

SCHEMA ITEM SHAPE:
{schema_item_shape}

TYPE EXAMPLES (for structure only, NOT for content):
{type_examples}
"""