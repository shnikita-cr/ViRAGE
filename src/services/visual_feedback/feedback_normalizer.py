from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from src.domain.feedback_models import NormalizedFeedbackRecord
from src.domain.visual_feedback_models import VisualFeedbackExample
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured

_FIELD_KEYS = {"field", "fieldName", "field_name"}


class _LLMNormalizedFeedbackSchema(BaseModel):
    feedback_type: str = Field(default="general_visual_feedback")
    severity: str = Field(default="medium")
    task_type: str = ""
    chart_family: str = ""
    problem: str = ""
    recommendation: str = ""
    avoid: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    fields_used: list[str] = Field(default_factory=list)
    priority: float = Field(default=1.0, ge=0.0, le=4.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackNormalizerService:
    """Normalize raw user/VLM chart feedback into reviewed RAG-facing records.

    The service has two explicit modes:
    - mode="llm" requires runtime.reasoning_llm and structures feedback with the project model.
    - mode="rules" is a deterministic, explicit utility mode for tests/offline bootstrapping.

    There is no hidden fallback from LLM to rules.
    """

    def normalize(
        self,
        raw: dict[str, Any] | VisualFeedbackExample,
        *,
        mode: str = "rules",
        runtime: RuntimeContext | None = None,
        approved_for_rag: bool = False,
    ) -> NormalizedFeedbackRecord:
        payload = raw.model_dump() if isinstance(raw, VisualFeedbackExample) else dict(raw)
        if mode == "llm":
            if runtime is None or runtime.reasoning_llm is None:
                raise RuntimeError("LLM feedback normalization requires runtime.reasoning_llm.")
            parsed = self._normalize_with_llm(payload, runtime)
        elif mode == "rules":
            parsed = self._normalize_with_rules(payload)
        else:
            raise ValueError("mode must be one of: rules, llm")

        source_kind = "manual_feedback" if str(payload.get("source") or "").lower().find("user") >= 0 else "vlm_feedback"
        feedback_id = self._feedback_id(payload)
        base_priority = self._base_priority(payload, source_kind)
        priority = max(base_priority, float(parsed.get("priority") or 0.0))
        severity = str(parsed.get("severity") or "medium").strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"

        return NormalizedFeedbackRecord(
            feedback_id=feedback_id,
            source_record_id=str(payload.get("run_id") or "") + f":{payload.get('attempt_number', 1)}",
            source_kind=source_kind,  # type: ignore[arg-type]
            source=str(payload.get("source") or ""),
            created_at=str(payload.get("created_at") or ""),
            run_id=str(payload.get("run_id") or ""),
            attempt_number=int(payload.get("attempt_number") or 1),
            user_query=str(payload.get("user_query") or ""),
            feedback_type=str(parsed.get("feedback_type") or "general_visual_feedback"),
            severity=severity,  # type: ignore[arg-type]
            task_type=str(parsed.get("task_type") or self._task_type(payload)),
            chart_family=str(parsed.get("chart_family") or self._chart_family(payload)),
            problem=str(parsed.get("problem") or self._feedback_text(payload)),
            recommendation=str(parsed.get("recommendation") or self._feedback_text(payload)),
            avoid=self._as_text_list(parsed.get("avoid")),
            quality_checks=self._as_text_list(parsed.get("quality_checks")),
            fields_used=self._dedupe([*self._as_text_list(parsed.get("fields_used")), *self._fields_used(payload)]),
            data_profile_summary=dict(payload.get("data_profile_summary") or {}),
            priority=priority,
            approved_for_rag=bool(approved_for_rag),
            approval_notes="approved_by_export_flag" if approved_for_rag else "manual_review_required",
            metadata={
                **dict(parsed.get("metadata") or {}),
                "status": payload.get("status"),
                "requested_regeneration": bool(payload.get("requested_regeneration", False)),
                "feedback_weight": payload.get("feedback_weight"),
            },
        )

    def _normalize_with_llm(self, payload: dict[str, Any], runtime: RuntimeContext) -> dict[str, Any]:
        parsed = invoke_structured(
            runtime.reasoning_llm,
            self._prompt(payload),
            _LLMNormalizedFeedbackSchema,
            runtime=runtime,
            stage="feedback_normalization",
            role="reasoning",
            max_attempts=2,
        )
        return parsed.model_dump()

    def _normalize_with_rules(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = self._feedback_text(payload)
        lowered = text.lower()
        feedback_type = "general_visual_feedback"
        if any(token in lowered for token in ["axis", "ось", "domain", "zero", "ноль", "0"]):
            feedback_type = "axis_domain_issue"
        if any(token in lowered for token in ["empty", "space", "whitespace", "пуст", "мест", "margin"]):
            feedback_type = "plot_area_issue"
        if any(token in lowered for token in ["compact", "width", "wide", "4:3", "boxplot", "боксплот", "шир"]):
            feedback_type = "compact_categorical_layout_issue"
        if any(token in lowered for token in ["repeat", "facet", "panel", "панел", "value", "metric", "подпис"]):
            feedback_type = "repeat_axis_label_issue"
        if any(token in lowered for token in ["publication", "article", "paper", "стат", "публикац"]):
            feedback_type = "publication_layout_issue"
        if any(token in lowered for token in ["label", "legend", "readable", "читаб", "легенд"]):
            feedback_type = "readability_issue" if feedback_type == "general_visual_feedback" else feedback_type
        if any(token in lowered for token in ["wrong field", "wrong encoding", "не то поле", "кодиров"]):
            feedback_type = "wrong_encoding_issue"
        severity = "high" if bool(payload.get("requested_regeneration")) or self._retry_recommended(payload) else "medium"
        return {
            "feedback_type": feedback_type,
            "severity": severity,
            "task_type": self._task_type(payload),
            "chart_family": self._chart_family(payload),
            "problem": text,
            "recommendation": text,
            "avoid": self._avoid_for_type(feedback_type),
            "quality_checks": self._checks_for_type(feedback_type),
            "fields_used": self._fields_used(payload),
            "priority": self._base_priority(payload, "manual_feedback" if str(payload.get("source") or "").lower().find("user") >= 0 else "vlm_feedback"),
        }

    @staticmethod
    def _prompt(payload: dict[str, Any]) -> str:
        minimal = {
            "source": payload.get("source"),
            "status": payload.get("status"),
            "user_query": payload.get("user_query"),
            "user_comment": payload.get("user_comment"),
            "requested_regeneration": payload.get("requested_regeneration"),
            "feedback_for_next_generation": payload.get("feedback_for_next_generation"),
            "request_analysis_summary": payload.get("request_analysis_summary"),
            "generated_spec": payload.get("generated_spec"),
            "chart_fact_summary": payload.get("chart_fact_summary"),
            "judge_result": payload.get("judge_result"),
            "data_profile_summary": payload.get("data_profile_summary"),
        }
        return (
            "Normalize one ViRAGE chart feedback record for a RAG guidance corpus.\n"
            "Do not invent new facts. Use only the provided feedback, chart spec metadata, and profile summary.\n"
            "Return concise structured guidance. approved_for_rag is not decided here.\n"
            "Prefer feedback_type values such as axis_domain_issue, plot_area_issue, "
            "compact_categorical_layout_issue, repeat_axis_label_issue, publication_layout_issue, "
            "readability_issue, wrong_encoding_issue, general_visual_feedback.\n\n"
            f"Raw feedback JSON:\n{json.dumps(minimal, ensure_ascii=False, default=str)}"
        )

    @staticmethod
    def _feedback_id(payload: dict[str, Any]) -> str:
        stable = json.dumps(
            {
                "run_id": payload.get("run_id"),
                "attempt_number": payload.get("attempt_number"),
                "created_at": payload.get("created_at"),
                "source": payload.get("source"),
                "feedback": payload.get("feedback_for_next_generation") or payload.get("user_comment"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return "feedback_" + hashlib.sha1(stable.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _feedback_text(payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("user_comment", "feedback_for_next_generation"):
            value = str(payload.get(key) or "").strip()
            if value:
                parts.append(value)
        judge = payload.get("judge_result") or {}
        if isinstance(judge, dict):
            for key in ("feedback_for_next_generation", "missing_requirements", "wrong_or_suspicious_parts", "improvement_comments"):
                value = judge.get(key)
                if isinstance(value, list):
                    parts.extend(str(item).strip() for item in value if str(item).strip())
                elif isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        return " ".join(dict.fromkeys(parts)).strip()

    @staticmethod
    def _retry_recommended(payload: dict[str, Any]) -> bool:
        judge = payload.get("judge_result") or {}
        return isinstance(judge, dict) and str(judge.get("retry_recommendation") or "").lower() in {"retry", "reject"}

    @staticmethod
    def _base_priority(payload: dict[str, Any], source_kind: str) -> float:
        if source_kind == "manual_feedback" and bool(payload.get("requested_regeneration")):
            return 3.0
        if source_kind == "manual_feedback":
            return 2.0
        if FeedbackNormalizerService._retry_recommended(payload):
            return 1.5
        return 1.0

    @staticmethod
    def _task_type(payload: dict[str, Any]) -> str:
        analysis = payload.get("request_analysis_summary") or {}
        if isinstance(analysis, dict):
            return str(analysis.get("analysis_task") or "")
        return ""

    @staticmethod
    def _chart_family(payload: dict[str, Any]) -> str:
        spec = payload.get("generated_spec") or {}
        if not isinstance(spec, dict):
            return ""
        mark = spec.get("mark")
        if isinstance(mark, dict):
            return str(mark.get("type") or "")
        if isinstance(mark, str):
            return mark
        return ""

    @classmethod
    def _fields_used(cls, payload: dict[str, Any]) -> list[str]:
        fields: list[str] = []
        analysis = payload.get("request_analysis_summary") or {}
        if isinstance(analysis, dict):
            fields.extend(cls._as_text_list(analysis.get("selected_fields")))
        spec = payload.get("generated_spec") or {}
        fields.extend(cls._extract_spec_fields(spec))
        return cls._dedupe(fields)

    @classmethod
    def _extract_spec_fields(cls, value: Any) -> list[str]:
        fields: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _FIELD_KEYS and isinstance(item, str) and item.strip():
                    fields.append(item.strip())
                else:
                    fields.extend(cls._extract_spec_fields(item))
        elif isinstance(value, list):
            for item in value:
                fields.extend(cls._extract_spec_fields(item))
        return fields

    @staticmethod
    def _as_text_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys([value for value in values if value]))

    @staticmethod
    def _avoid_for_type(feedback_type: str) -> list[str]:
        mapping = {
            "axis_domain_issue": ["Do not force an axis to start at zero when zero is not analytically meaningful and the data occupy a narrow non-zero range."],
            "plot_area_issue": ["Do not export charts where the visible data occupy only a small fraction of the plotting area."],
            "compact_categorical_layout_issue": ["Do not use a wide 4:3 canvas for two to four narrow categorical marks such as boxplots."],
            "repeat_axis_label_issue": ["Do not leave repeat/facet panels with only generic axis labels such as Value or Metric when panel titles are unclear."],
            "publication_layout_issue": ["Do not rely on tooltips or excessive whitespace for a static article figure."],
        }
        return mapping.get(feedback_type, [])

    @staticmethod
    def _checks_for_type(feedback_type: str) -> list[str]:
        mapping = {
            "axis_domain_issue": ["Check that the axis domain makes variation visible without clipping data."],
            "plot_area_issue": ["Check that data marks use the plotting area efficiently."],
            "compact_categorical_layout_issue": ["Check that figure width matches the number of categories."],
            "repeat_axis_label_issue": ["Check that each repeat/facet panel has a visible metric or variable label."],
            "publication_layout_issue": ["Check that the static figure is readable and suitable for insertion into an article."],
        }
        return mapping.get(feedback_type, ["Check that the feedback recommendation is reflected in the next chart generation."])
