"""Pydantic domain models shared by parsing, matching, and the UI."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CriterionKind(str, Enum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class Operator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    WITHIN_DAYS = "within_days"
    EXISTS = "exists"
    HUMAN_REVIEW = "human_review"


class ExecutionStatus(str, Enum):
    AUTOMATED = "automated"
    HUMAN_REVIEW = "human_review"


class Criterion(BaseModel):
    """A traceable, machine-readable eligibility criterion."""

    model_config = ConfigDict(use_enum_values=True, validate_default=True)

    criterion_id: str
    kind: CriterionKind
    source_text: str = Field(min_length=1)
    source_reference: str
    field: str | None = None
    operator: Operator
    value: Any = None
    unit: str | None = None
    time_window_days: int | None = Field(default=None, ge=0)
    logic_group: str | None = None
    applicability: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_status: ExecutionStatus = ExecutionStatus.AUTOMATED
    review_status: Literal[
        "pending", "confirmed", "changes_requested", "expert_review"
    ] = "pending"
    reviewer: str = ""
    review_comment: str = ""
    note: str = ""

    @field_validator("criterion_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip().upper()


class TrialSource(BaseModel):
    source_type: Literal["clinicaltrials", "pdf", "text", "demo"]
    identifier: str
    title: str
    source_reference: str
    criteria_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CriterionEvidence(BaseModel):
    criterion_id: str
    criterion_kind: CriterionKind
    status: Literal["pass", "fail", "missing", "review", "not_applicable"]
    field: str | None
    patient_value: Any = None
    expected: Any = None
    source_text: str
    message: str


class MatchResult(BaseModel):
    patient_id: str
    overall_status: Literal["eligible", "ineligible", "missing_data", "needs_review"]
    evidences: list[CriterionEvidence]

    @property
    def failed_criteria(self) -> list[str]:
        return [item.criterion_id for item in self.evidences if item.status == "fail"]

    @property
    def missing_criteria(self) -> list[str]:
        return [item.criterion_id for item in self.evidences if item.status == "missing"]

    @property
    def review_criteria(self) -> list[str]:
        return [item.criterion_id for item in self.evidences if item.status == "review"]
