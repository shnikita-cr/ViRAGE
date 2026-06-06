from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.domain.models import DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_text
from src.orchestrator.contracts.models import AnalysisPlan, analysis_plan_contract_metadata, allowed_analysis_task_types
from src.services.data.profile.data_profile_prompt_formatter import DataProfilePromptFormatter
from src.services.planning.planning_guidance import PlanningGuidanceService


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
                parsed = self._parse_strict_plan_json(raw_response)
                plan_payload = parsed.model_dump()
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
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
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
    def _parse_strict_plan_json(raw_response: str) -> AnalysisPlan:
        text = (raw_response or "").strip()
        if not text:
            raise ValueError("Empty analysis planner response.")
        if "```" in text:
            raise ValueError("Analysis planner response must be raw JSON without markdown fences.")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Analysis planner response must be one JSON object.")
        return AnalysisPlan.model_validate(payload)

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
        return (
            "\nStrict JSON schema requirements for AnalysisPlan:\n"
            "Return one raw JSON object matching this Pydantic schema. No markdown. No XML. No comments.\n"
            f"{json.dumps(AnalysisPlan.model_json_schema(), ensure_ascii=False, indent=2)}\n\n"
            "Example shape using available fields only:\n"
            f"{json.dumps(example, ensure_ascii=False, indent=2, default=str)}\n"
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
        return (
            "You are the ViRAGE LLM analysis orchestrator. Build one strict AnalysisPlan JSON object for one dataset.\n"
            "The plan will be executed directly, so do not output markdown, code fences, comments, or explanatory text.\n"
            "Use the user request and compact DataProfile only. Do not infer fields that are not listed.\n\n"
            "Planning contract:\n"
            f"1. Produce between 1 and {self.max_charts} subtasks. Never produce more than max_charts.\n"
            "2. Every subtask.required_fields must be non-empty and must contain only exact field names from available_fields.\n"
            "3. Every optional_fields entry must also be an exact available field name.\n"
            "4. task_type must be one of the allowed task types.\n"
            "5. skipped_candidates is mandatory. Use it to document relevant task types considered but not selected.\n"
            "6. skipped_candidates.required_fields may be empty when no concrete existing field combination supports the skipped task.\n"
            "7. Do not choose chart families. Do not recommend chart types. Select analytical subtasks and fields only.\n"
            "8. For image_folder input, use only columns generated in the image metrics table. If every IQA metric column is unusable or absent, do not select it.\n"
            "9. If the request is broad EDA, select the most useful complementary subtasks, not duplicate views.\n"
            "10. If the request is specific, prioritize the requested analytical task and include other subtasks only when they are clearly useful.\n"
            "11. Use retrieved planning guidance to define metric_semantics, ranking_strategy, scale_strategy and visual_constraints when relevant. metric_semantics keys must be exact available field names.\n"
            "12. For problematic-item or quality requests, plan ranking/severity views instead of raw all-items multi-metric charts.\n"
            "13. Do not put different-scale metrics on one shared quantitative axis. Plan normalized severity or independent panels.\n\n"
            f"Authoritative metadata JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n\n"
            f"User request:\n{user_query.strip()}\n\n"
            f"User context:\n{context_lines}\n\n"
            f"Retrieved planning guidance:\n{planning_guidance or 'No planning guidance retrieved.'}\n\n"
            f"Compact DataProfile:\n{DataProfilePromptFormatter.for_query_analysis(data_profile, max_columns=40)}\n"
        )

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
