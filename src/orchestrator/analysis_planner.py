from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.domain.models import DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_text
from src.llm.structured_response import extract_structured_json
from src.orchestrator.contracts.models import AnalysisPlan, analysis_plan_contract_metadata, allowed_analysis_task_types
from src.services.data.profile.data_profile_prompt_formatter import DataProfilePromptFormatter
from src.services.planning.planning_guidance import PlanningGuidanceService

logger = logging.getLogger(__name__)

_PROMPT_CHAR_BUDGET = 12000
_PLANNING_GUIDANCE_CHAR_BUDGET = 1800
_DATA_PROFILE_CHAR_BUDGET = 4200
_USER_CONTEXT_CHAR_BUDGET = 800


@dataclass(frozen=True)
class _ParsedAnalysisPlan:
    plan: AnalysisPlan
    parse_mode: str


class AnalysisPlanner:
    """Create an LLM-grounded, schema-validated analysis plan for one dataset.

    The planner intentionally has no rules/regex branch. If the model cannot
    produce a valid plan after retries, planning fails instead of silently
    substituting heuristic tasks.
    """

    def __init__(
            self,
            *,
            max_charts: int = 3,
            reasoning_llm: Any | None = None,
            max_attempts: int = 2,
    ) -> None:
        if max_charts < 1 or max_charts > 3:
            raise ValueError("max_charts must be between 1 and 3.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.max_charts = max_charts
        self.reasoning_llm = reasoning_llm
        self.max_attempts = max_attempts

    def plan(
            self,
            *,
            user_query: str,
            data_path: str,
            data_profile: DataProfile,
            runtime: RuntimeContext | None = None,
            user_context: dict[str, Any] | None = None,
            input_type: str = "table",
            original_input_path: str | None = None,
            preprocessing_report_path: str | None = None,
    ) -> AnalysisPlan:
        llm = self.reasoning_llm or (runtime.reasoning_llm if runtime is not None else None)
        if llm is None:
            raise RuntimeError("AnalysisPlanner requires a reasoning LLM; rules planner has been removed.")

        planning_guidance = self._planning_guidance(
            user_query=user_query,
            data_profile=data_profile,
            runtime=runtime,
            input_type=input_type,
        )
        prompt = self._prompt(
            user_query=user_query,
            data_path=data_path,
            data_profile=data_profile,
            user_context=user_context or {},
            input_type=input_type,
            original_input_path=original_input_path,
            preprocessing_report_path=preprocessing_report_path,
            planning_guidance=planning_guidance,
        )
        available_fields = {column.name for column in data_profile.columns}
        errors: list[str] = []
        current_prompt = prompt + "\n\n" + self._strict_json_schema_block(data_profile, user_query, data_path, input_type)
        for attempt_number in range(1, self.max_attempts + 1):
            raw_response = invoke_text(
                llm,
                current_prompt,
                runtime=runtime,
                stage="analysis_planner",
                role="reasoning",
            )
            try:
                parsed = self._parse_plan_json(raw_response)
                self._log_parse_mode(parsed.parse_mode)
                plan_payload = parsed.plan.model_dump()
                plan_payload.update(
                    {
                        "user_query": user_query.strip(),
                        "data_path": data_path,
                        "input_type": input_type,
                        "original_input_path": original_input_path,
                        "preprocessing_report_path": preprocessing_report_path,
                        "max_charts": self.max_charts,
                    }
                )
                plan = AnalysisPlan.model_validate(plan_payload)
                plan.validate_against_available_fields(available_fields)
                return plan
            except (ValidationError, ValueError) as exc:
                errors.append(f"attempt {attempt_number}: {type(exc).__name__}: {exc}")
                if attempt_number >= self.max_attempts:
                    break
                current_prompt = (
                    prompt
                    + "\n\nThe previous response violated the strict AnalysisPlan contract. "
                    + "Return one raw JSON object only. Do not use markdown fences. Do not invent fields.\n"
                    + f"Validation/parsing error:\n{exc}\n"
                    + f"Previous response:\n{raw_response}\n"
                    + self._strict_json_schema_block(data_profile, user_query, data_path, input_type)
                )
        raise RuntimeError(
            f"Failed to parse AnalysisPlan after {self.max_attempts} attempts. "
            f"Last error: {errors[-1] if errors else 'unknown'}"
        )

    def _planning_guidance(
            self,
            *,
            user_query: str,
            data_profile: DataProfile,
            runtime: RuntimeContext | None,
            input_type: str,
    ) -> str:
        return PlanningGuidanceService().invoke(
            user_query=user_query,
            data_profile=data_profile,
            runtime=runtime,
            input_type=input_type,
        ).text

    @staticmethod
    def _parse_plan_json(raw_response: str) -> _ParsedAnalysisPlan:
        text = (raw_response or "").strip()
        if not text:
            raise ValueError("Empty analysis planner response.")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as strict_error:
            return AnalysisPlanner._parse_extracted_plan(text, strict_error)
        return _ParsedAnalysisPlan(
            plan=AnalysisPlanner._validate_plan_payload(payload),
            parse_mode="raw",
        )

    @staticmethod
    def _parse_extracted_plan(text: str, strict_error: json.JSONDecodeError) -> _ParsedAnalysisPlan:
        extraction = extract_structured_json(text)
        if extraction.error is not None:
            raise ValueError(
                "Analysis planner response is not raw JSON and no valid JSON object could be extracted. "
                f"Strict JSON error: {strict_error}. Extraction error: {extraction.error}"
            )
        return _ParsedAnalysisPlan(
            plan=AnalysisPlanner._validate_plan_payload(extraction.payload),
            parse_mode=extraction.mode,
        )

    @staticmethod
    def _validate_plan_payload(payload: Any) -> AnalysisPlan:
        if not isinstance(payload, dict):
            raise ValueError("Analysis planner response must be one JSON object.")
        return AnalysisPlan.model_validate(payload)

    @staticmethod
    def _log_parse_mode(parse_mode: str) -> None:
        if parse_mode == "raw":
            return
        logger.warning(
            "Analysis planner response violated the raw JSON contract; parsed via %s.",
            parse_mode,
        )

    def _strict_json_schema_block(
            self,
            data_profile: DataProfile,
            user_query: str,
            data_path: str,
            input_type: str,
    ) -> str:
        example = self._example_payload(
            user_query=user_query,
            data_path=data_path,
            data_profile=data_profile,
            input_type=input_type,
        )
        compact_schema = {
            "required_top_level_keys": ["subtasks", "skipped_candidates", "rationale"],
            "subtask_required_keys": [
                "id", "task_type", "query", "purpose", "required_fields", "optional_fields",
                "priority", "constraints", "rationale",
            ],
            "allowed_task_types": allowed_analysis_task_types(),
            "field_rule": "required_fields and optional_fields must use exact available_fields values only",
            "max_charts": self.max_charts,
        }
        return (
            "Strict AnalysisPlan JSON contract:\n"
            "Return one raw JSON object. No markdown, XML, comments, or prose.\n"
            f"{json.dumps(compact_schema, ensure_ascii=False, separators=(',', ':'))}\n"
            "Example using available fields only:\n"
            f"{json.dumps(example, ensure_ascii=False, separators=(',', ':'), default=str)}\n"
        )

    def _prompt(
            self,
            *,
            user_query: str,
            data_path: str,
            data_profile: DataProfile,
            user_context: dict[str, Any],
            input_type: str,
            original_input_path: str | None,
            preprocessing_report_path: str | None,
            planning_guidance: str,
    ) -> str:
        context_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(user_context.items())) or "- none"
        payload = {
            "data_path": data_path,
            "input_type": input_type,
            "original_input_path": original_input_path,
            "preprocessing_report_path": preprocessing_report_path,
            "max_charts": self.max_charts,
            "allowed_task_types": allowed_analysis_task_types(),
            "allowed_controlled_values": analysis_plan_contract_metadata(),
            "available_fields": [column.name for column in data_profile.columns],
        }
        prompt = (
            "Role: scientific data analysis planner. Select 1-3 executable analytical subtasks for one dataset.\n"
            "Output: raw AnalysisPlan JSON only. No markdown fences, no comments, no prose.\n"
            "Use only listed fields. Do not infer missing columns. Do not choose chart families.\n\n"
            "Rules:\n"
            f"1. Produce 1..{self.max_charts} subtasks.\n"
            "2. required_fields must be non-empty and must use exact available_fields values.\n"
            "3. optional_fields must also use exact available_fields values.\n"
            "4. task_type must be allowed. skipped_candidates is mandatory.\n"
            "5. Prefer complementary subtasks; avoid duplicate views.\n"
            "6. For quality/problem requests, plan ranking or normalized severity instead of raw mixed-scale axes.\n\n"
            f"Metadata JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}\n\n"
            f"User request:\n{user_query.strip()}\n\n"
            f"User context:\n{_truncate_text(context_lines, _USER_CONTEXT_CHAR_BUDGET)}\n\n"
            f"Planning guidance:\n{_truncate_text(planning_guidance or 'No planning guidance retrieved.', _PLANNING_GUIDANCE_CHAR_BUDGET)}\n\n"
            "Compact DataProfile:\n"
            f"{_truncate_text(DataProfilePromptFormatter.for_query_analysis(data_profile, max_columns=24), _DATA_PROFILE_CHAR_BUDGET)}\n"
        )
        return _truncate_text(prompt, _PROMPT_CHAR_BUDGET)

    def _example_payload(
            self,
            *,
            user_query: str,
            data_path: str,
            data_profile: DataProfile,
            input_type: str,
    ) -> dict[str, Any]:
        fields = [column.name for column in data_profile.columns]
        first = fields[0] if fields else "field"
        second = fields[1] if len(fields) > 1 else first
        return {
            "user_query": user_query.strip() or "Analyze the dataset.",
            "data_path": data_path,
            "input_type": input_type,
            "original_input_path": None,
            "preprocessing_report_path": None,
            "max_charts": self.max_charts,
            "subtasks": [
                {
                    "id": "analysis_001",
                    "task_type": "group_comparison" if len(fields) > 1 else "overview",
                    "query": f"Analyze {second} by {first}.",
                    "purpose": "Answer the highest-priority analytical part of the user request.",
                    "required_fields": [first, second] if len(fields) > 1 else [first],
                    "optional_fields": [],
                    "priority": 1,
                    "constraints": {"output_target": "scientific_figure"},
                    "metric_semantics": {},
                    "ranking_strategy": None,
                    "scale_strategy": None,
                    "visual_constraints": [],
                    "comparison_group_id": None,
                    "rationale": "The selected fields exist in the provided DataProfile and match the request.",
                }
            ],
            "skipped_candidates": [
                {
                    "task_type": "correlation",
                    "reason": "The request did not require an additional numeric relationship view.",
                    "required_fields": [],
                }
            ],
            "rationale": ["The plan uses exact DataProfile fields and stays within max_charts."],
        }


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = text or ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 24)].rstrip() + "\n[truncated]"
