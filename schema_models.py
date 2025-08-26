# schema_models.py

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

Rich = dict  # treat rich blocks opaquely

# --------------------------
# Question schema
# --------------------------
class Choice(BaseModel):
    id: str
    text_rich: List[Rich]
    is_correct: Optional[bool] = None
    rationale_rich: Optional[List[Rich]] = None

class Grading(BaseModel):
    mode: Literal["msq","mcq"]
    partial_credit: bool
    penalty: int
    require_all_correct: bool

class Question(BaseModel):
    id: str
    type: Literal["msq","mcq","tf"]
    unit_id: str
    topic_id: str
    question_rich: List[Rich]
    context_rich: List[Rich]
    choices: List[Choice]
    difficulty: int = Field(ge=1, le=5)
    tags: List[str]
    concept_tags: List[str]
    context_tags: List[str]
    hints_rich: List[Rich]
    mnemonic_rich: List[Rich]
    explanation_rich: List[Rich]
    elaboration_prompts_rich: List[Rich]
    shuffle: bool
    grading: Optional[Grading] = None
    example_rich: List[Rich]

    @model_validator(mode="after")
    def check_by_type(self):
        if self.type == "tf":
            assert len(self.choices) == 2, "TF must have 2 choices"
            assert sum(1 for c in self.choices if c.is_correct) == 1, "TF must have 1 correct"
        if self.type == "mcq":
            assert sum(1 for c in self.choices if c.is_correct) == 1, "MCQ must have exactly 1 correct"
            assert self.grading and self.grading.mode == "mcq", "Grading.mode must be 'mcq' for MCQ"
        if self.type == "msq":
            assert sum(1 for c in self.choices if c.is_correct) >= 2, "MSQ should have 2+ correct"
            assert self.grading and self.grading.mode == "msq", "Grading.mode must be 'msq' for MSQ"
        for name, value in self.__dict__.items():
            assert value not in (None, [], "", {}), f"Field {name} is empty"
        return self

class QuestionFile(BaseModel):
    schema_version: Literal["1.0"]
    questions: List[Question]

# --------------------------
# Taxonomy schema (NEW)
# --------------------------
class TaxonomyTagEntry(BaseModel):
    id: str
    label: str
    aliases: List[str]
    description: str

class TaxonomyUnit(BaseModel):
    id: str
    label: str
    description: str

class TaxonomyTopic(BaseModel):
    id: str
    label: str
    description: str

class Taxonomy(BaseModel):
    version: str
    units: List[TaxonomyUnit]
    topics: List[TaxonomyTopic]
    tags: List[TaxonomyTagEntry]
    concept_tags: List[TaxonomyTagEntry]
    context_tags: List[TaxonomyTagEntry]