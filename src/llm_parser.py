"""DeepSeek-backed eligibility parsing with schema validation and cost guards."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from pydantic import ValidationError

from .config import (
    DATA_DIR,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_TIMEOUT_SECONDS,
    MAX_LIVE_CALLS_PER_HOUR,
    MAX_LLM_CHUNK_CHARS,
    MAX_LLM_TOTAL_CHARS,
)
from .models import Criterion
from .trial_sources import normalize_text


class LLMParseError(RuntimeError):
    """A safe error surfaced when a live parse cannot be completed."""


class LLMRateLimitError(LLMParseError):
    pass


@dataclass(frozen=True)
class ParseOutcome:
    criteria: list[Criterion]
    from_cache: bool
    model: str
    chunk_count: int


class HourlyBudget:
    def __init__(self, limit: int = MAX_LIVE_CALLS_PER_HOUR) -> None:
        self.limit = limit
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def consume(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self._lock:
            while self._timestamps and current - self._timestamps[0] >= 3600:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.limit:
                raise LLMRateLimitError("实时解析已达到小时调用上限，请使用缓存案例或稍后再试。")
            self._timestamps.append(current)

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()


_BUDGET = HourlyBudget()
_CACHE: dict[str, list[Criterion]] = {}


ALLOWED_FIELDS = [
    "age",
    "sex",
    "copd_diagnosis",
    "smoking_pack_years",
    "post_bd_fev1_pct_predicted",
    "post_bd_fev1_liters",
    "post_bd_fev1_fvc",
    "spirometry_reproducible",
    "contraception_confirmed",
    "informed_consent_confirmed",
    "visit_adherence_confirmed",
    "severe_comorbidity_concern",
    "other_significant_respiratory_disease",
    "days_since_copd_exacerbation",
    "oxygen_hours_per_day",
    "days_since_respiratory_infection",
    "days_since_systemic_steroids",
    "malignancy_within_5y",
    "qtc_ms",
    "bladder_outflow_obstruction_within_6m",
    "narrow_angle_glaucoma",
    "aerosol_medication_hypersensitivity",
    "substance_abuse_within_3m",
    "psychiatric_completion_concern",
    "investigational_drug_within_30d",
    "prior_sun101",
    "study_drug_class_hypersensitivity",
]


def text_hash(text: str, model: str = DEEPSEEK_DEFAULT_MODEL) -> str:
    payload = f"{model}\n{normalize_text(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paragraph_chunks(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in normalize_text(text).split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > limit:
            if current:
                chunks.append("\n\n".join(current))
                current, current_size = [], 0
            chunks.extend(paragraph[i : i + limit] for i in range(0, len(paragraph), limit))
            continue
        needed = len(paragraph) + (2 if current else 0)
        if current and current_size + needed > limit:
            chunks.append("\n\n".join(current))
            current, current_size = [], 0
        current.append(paragraph)
        current_size += needed
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_for_llm(text: str) -> list[str]:
    normalized = normalize_text(text)[:MAX_LLM_TOTAL_CHARS]
    if len(normalized) <= MAX_LLM_CHUNK_CHARS:
        return [normalized]
    return _paragraph_chunks(normalized, MAX_LLM_CHUNK_CHARS)


def _system_prompt(source_reference: str, chunk_index: int) -> str:
    schema_example = {
        "criteria": [
            {
                "criterion_id": f"C{chunk_index:02d}-001",
                "kind": "inclusion",
                "source_text": "Age 40 years or older",
                "source_reference": source_reference,
                "field": "age",
                "operator": "gte",
                "value": 40,
                "unit": "year",
                "time_window_days": None,
                "logic_group": None,
                "applicability": {},
                "confidence": 0.95,
                "execution_status": "automated",
                "note": "",
            }
        ]
    }
    return (
        "You are a clinical-trial eligibility structuring assistant. Output JSON only. "
        "Extract every inclusion and exclusion criterion without deciding whether a patient qualifies. "
        "Preserve each original criterion in source_text. Use only these operators: "
        "eq, neq, lt, lte, gt, gte, between, in, not_in, is_true, is_false, within_days, exists, human_review. "
        "Use execution_status=human_review whenever the rule is subjective, depends on investigator judgment, "
        "or cannot be represented with the available fields. For unsupported concepts, keep field null and "
        "operator human_review. Use applicability for simple exact conditions such as sex. "
        f"Preferred executable fields: {', '.join(ALLOWED_FIELDS)}. "
        f"The exact JSON shape is: {json.dumps(schema_example, ensure_ascii=False)}"
    )


def _validate_payload(payload: object, chunk_index: int, source_reference: str) -> list[Criterion]:
    if not isinstance(payload, dict) or not isinstance(payload.get("criteria"), list):
        raise LLMParseError("模型返回内容缺少 criteria 数组。")
    validated: list[Criterion] = []
    for position, item in enumerate(payload["criteria"], start=1):
        if not isinstance(item, dict):
            raise LLMParseError("模型返回了无法识别的标准条目。")
        candidate = dict(item)
        candidate.setdefault("criterion_id", f"C{chunk_index:02d}-{position:03d}")
        candidate["source_reference"] = source_reference
        try:
            validated.append(Criterion.model_validate(candidate))
        except ValidationError as exc:
            raise LLMParseError(f"标准 {position} 未通过结构校验。") from exc
    if not validated:
        raise LLMParseError("模型没有提取出任何标准。")
    return validated


def _deduplicate(criteria: Iterable[Criterion]) -> list[Criterion]:
    output: list[Criterion] = []
    seen: set[tuple[str, str]] = set()
    for criterion in criteria:
        key = (criterion.kind, normalize_text(criterion.source_text).lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(criterion.model_copy(update={"criterion_id": f"C{len(output) + 1:03d}"}))
    return output


def parse_with_deepseek(
    text: str,
    api_key: str,
    source_reference: str,
    model: str = DEEPSEEK_DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> ParseOutcome:
    if not api_key.strip():
        raise LLMParseError("未配置 DEEPSEEK_API_KEY，无法进行实时解析。")
    chunks = split_for_llm(text)
    if not chunks or not chunks[0]:
        raise LLMParseError("没有可发送给模型的标准文本。")

    cache_key = text_hash(text, model)
    if cache_key in _CACHE:
        return ParseOutcome(deepcopy(_CACHE[cache_key]), True, model, len(chunks))

    sdk_client = client or OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
        max_retries=0,
    )
    extracted: list[Criterion] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        last_error: Exception | None = None
        for attempt in range(DEEPSEEK_MAX_RETRIES + 1):
            try:
                _BUDGET.consume()
                response = sdk_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _system_prompt(source_reference, chunk_index)},
                        {"role": "user", "content": f"Parse this eligibility text into JSON:\n\n{chunk}"},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=8192,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise LLMParseError("模型返回了空内容。")
                payload = json.loads(content)
                extracted.extend(_validate_payload(payload, chunk_index, source_reference))
                last_error = None
                break
            except (json.JSONDecodeError, ValidationError, LLMParseError, Exception) as exc:
                last_error = exc
                if attempt < DEEPSEEK_MAX_RETRIES:
                    time.sleep(0.25 * (2**attempt))
        if last_error is not None:
            if isinstance(last_error, LLMRateLimitError):
                raise last_error
            raise LLMParseError("DeepSeek 解析失败，原文已保留，请稍后重试。") from last_error

    merged = _deduplicate(extracted)
    _CACHE[cache_key] = deepcopy(merged)
    return ParseOutcome(merged, False, model, len(chunks))


def load_cached_demo_criteria(path: Path | None = None) -> list[Criterion]:
    criteria_path = path or DATA_DIR / "golden4_criteria.json"
    payload = json.loads(criteria_path.read_text(encoding="utf-8"))
    return [Criterion.model_validate(item) for item in payload]


def reset_runtime_guards() -> None:
    """Test helper: clear in-process cache and budget."""

    _CACHE.clear()
    _BUDGET.reset()
