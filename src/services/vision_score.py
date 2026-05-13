from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import PlotImageArtifact, QueryUnderstandingResult, VisualQualityMetric
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import (
    ainvoke_structured_multimodal,
    ainvoke_structured_multimodal_many,
    invoke_structured_multimodal,
    invoke_structured_multimodal_many,
)
from src.services.base import BaseService


class _VisionScoreSchema(BaseModel):
    visualization_type: int = Field(default=0, ge=0, le=2)
    data_encoding: int = Field(default=0, ge=0, le=2)
    data_transformation: int = Field(default=0, ge=0, le=2)
    aesthetics: int = Field(default=0, ge=0, le=2)
    prompt_compliance: int = Field(default=0, ge=0, le=2)
    insight_supportiveness: int = Field(default=0, ge=0, le=2)
    is_blank: bool = False
    rationales: dict[str, str] = Field(default_factory=dict)
    details: list[str] = Field(default_factory=list)


class VisionScoreService(BaseService):
    """VegaChat-compatible visual metric.

    Two modes are supported:
    1. reference mode: compares a generated chart image against a reference image and the user prompt;
    2. self mode: evaluates one generated image against the user prompt for normal ViRAGE UI runs.

    Reference mode follows the VegaChat metric structure: a VLM returns discrete 0/1/2 dimension scores for
    visualization type, data encoding, data transformation, aesthetics, prompt compliance, and blank-chart detection.
    The final score is a deterministic weighted aggregation of the normalized dimensions.
    """

    VEGACHAT_WEIGHTS = {
        "visualization_type": 1.0,
        "data_encoding": 2.0,
        "data_transformation": 1.0,
        "aesthetics": 0.75,
        "prompt_compliance": 1.5,
    }
    SELF_WEIGHTS = {
        "visualization_type": 1.0,
        "data_encoding": 2.0,
        "data_transformation": 1.0,
        "aesthetics": 0.75,
        "prompt_compliance": 1.5,
        "insight_supportiveness": 1.0,
    }

    def invoke(
            self,
            plot_image: PlotImageArtifact,
            runtime: RuntimeContext,
            query_understanding: QueryUnderstandingResult | None = None,
            *,
            user_prompt: str | None = None,
            reference_image_path: str | None = None,
    ) -> VisualQualityMetric:
        if runtime.vision_judge_llm is None:
            raise RuntimeError("Vision scoring requires runtime.vision_judge_llm. No vision-judge model was provided.")
        if reference_image_path:
            prompt = self._build_reference_prompt(query_understanding, user_prompt)
            parsed = invoke_structured_multimodal_many(
                runtime.vision_judge_llm,
                prompt,
                [plot_image.image_path, reference_image_path],
                _VisionScoreSchema,
                runtime=runtime,
                stage="vision_score",
                role="vision_judge",
                examples=[self._reference_example()],
                max_attempts=2,
            )
            return self._to_metric(parsed, weights=self.VEGACHAT_WEIGHTS, mode="reference")

        prompt = self._build_self_prompt(query_understanding, user_prompt)
        parsed = invoke_structured_multimodal(
            runtime.vision_judge_llm,
            prompt,
            plot_image.image_path,
            _VisionScoreSchema,
            runtime=runtime,
            stage="vision_score",
            role="vision_judge",
            examples=[self._self_example()],
            max_attempts=2,
        )
        return self._to_metric(parsed, weights=self.SELF_WEIGHTS, mode="self")

    async def ainvoke(
            self,
            plot_image: PlotImageArtifact,
            runtime: RuntimeContext,
            query_understanding: QueryUnderstandingResult | None = None,
            *,
            user_prompt: str | None = None,
            reference_image_path: str | None = None,
    ) -> VisualQualityMetric:
        if runtime.vision_judge_llm is None:
            raise RuntimeError("Vision scoring requires runtime.vision_judge_llm. No vision-judge model was provided.")
        if reference_image_path:
            prompt = self._build_reference_prompt(query_understanding, user_prompt)
            parsed = await ainvoke_structured_multimodal_many(
                runtime.vision_judge_llm,
                prompt,
                [plot_image.image_path, reference_image_path],
                _VisionScoreSchema,
                runtime=runtime,
                stage="vision_score",
                role="vision_judge",
                examples=[self._reference_example()],
                max_attempts=2,
            )
            return self._to_metric(parsed, weights=self.VEGACHAT_WEIGHTS, mode="reference")

        prompt = self._build_self_prompt(query_understanding, user_prompt)
        parsed = await ainvoke_structured_multimodal(
            runtime.vision_judge_llm,
            prompt,
            plot_image.image_path,
            _VisionScoreSchema,
            runtime=runtime,
            stage="vision_score",
            role="vision_judge",
            examples=[self._self_example()],
            max_attempts=2,
        )
        return self._to_metric(parsed, weights=self.SELF_WEIGHTS, mode="self")

    @staticmethod
    def _build_reference_prompt(
            query_understanding: QueryUnderstandingResult | None,
            user_prompt: str | None,
    ) -> str:
        intent = query_understanding.intent if query_understanding else "unknown"
        goal = query_understanding.analysis_goal if query_understanding else "unknown"
        prompt = user_prompt or "unknown"
        return (
            "You are an excellent judge at evaluating visualizations between a model-generated plot and "
            "the ground truth plot. You will be given the generated plot as the first image, the "
            "ground_truth plot as the second image, and the user's request to generate the first image.\n\n"
            "Scoring should be carried out based on the following criteria. Use only integer scores.\n"
            "- visualization_type: 2 if the visualization type is the same as the ground_truth; "
            "1 if it is different but does not significantly affect interpretation; 0 otherwise.\n"
            "- data_encoding: 2 if the generated plot encodes the same data as the ground_truth; "
            "1 if it is missing an encoding channel or has flipped x/y axes; "
            "0 if it encodes completely different data.\n"
            "- data_transformation: 2 if transformations match or neither image has transformations; "
            "1 if transformations differ but interpretation remains similar; 0 otherwise.\n"
            "- aesthetics: 2 if style is similar to the ground_truth; "
            "1 if style differs but interpretation is not affected; 0 if style has no resemblance.\n"
            "- prompt_compliance: 2 if the generated plot is relevant to the user query and contains all "
            "requested information; 1 if partially relevant but missing some requested information; 0 otherwise. "
            "This score is independent of the ground_truth.\n"
            "- is_blank: true if the generated plot is blank, empty, or shows no data except axes; false otherwise.\n\n"
            "Return only structured values for the requested schema. The generated image is first; "
            "the reference image is second.\n"
            f"User request: {prompt}\n"
            f"Detected intent: {intent}\n"
            f"Analysis goal: {goal}\n"
        )

    @staticmethod
    def _build_self_prompt(
            query_understanding: QueryUnderstandingResult | None,
            user_prompt: str | None,
    ) -> str:
        intent = query_understanding.intent if query_understanding else "unknown"
        goal = query_understanding.analysis_goal if query_understanding else "unknown"
        prompt = user_prompt or "unknown"
        return (
            "Judge the generated chart image using a VegaChat-style 0,1,2 scale.\n"
            "No reference image is available, so judge against the user request and the apparent chart quality.\n"
            "Criteria: visualization_type, data_encoding, data_transformation, aesthetics, prompt_compliance, "
            "insight_supportiveness.\n"
            "Set is_blank=true if the image is effectively blank or useless.\n"
            f"User request: {prompt}\n"
            f"Detected intent: {intent}\n"
            f"Analysis goal: {goal}\n"
        )

    @staticmethod
    def _reference_example() -> dict[str, object]:
        return {
            "visualization_type": 2,
            "data_encoding": 2,
            "data_transformation": 1,
            "aesthetics": 2,
            "prompt_compliance": 2,
            "insight_supportiveness": 0,
            "is_blank": False,
            "rationales": {
                "visualization_type": "Both charts use the same visual mark family.",
                "data_encoding": "The generated chart maps the same data fields to equivalent visual channels.",
            },
            "details": ["reference-image comparison"],
        }

    @staticmethod
    def _self_example() -> dict[str, object]:
        return {
            "visualization_type": 2,
            "data_encoding": 2,
            "data_transformation": 1,
            "aesthetics": 2,
            "prompt_compliance": 2,
            "insight_supportiveness": 2,
            "is_blank": False,
            "rationales": {"prompt_compliance": "The chart answers the user request."},
            "details": ["single-image quality judgment"],
        }

    def _to_metric(self, parsed: _VisionScoreSchema, *, weights: dict[str, float], mode: str) -> VisualQualityMetric:
        rationales = dict(parsed.rationales)
        details = [f"mode={mode}", *parsed.details]
        weighted_sum = 0.0
        max_sum = 0.0
        for name, weight in weights.items():
            value = getattr(parsed, name)
            weighted_sum += (float(value) / 2.0) * weight
            max_sum += weight
            details.append(f"{name}={value}/2")
        if parsed.is_blank:
            # VegaChat treats blank charts as a huge denominator penalty instead of a hard early return.
            max_sum += 1000.0
            details.append("is_blank=1/1")
            details.append("blank_weight=1000")
        else:
            details.append("is_blank=0/1")
        score = weighted_sum / max_sum if max_sum else 0.0
        return VisualQualityMetric(
            score=round(max(0.0, min(1.0, score)), 6),
            prompt_compliance=round(parsed.prompt_compliance / 2.0, 6),
            readability=round(parsed.aesthetics / 2.0, 6),
            insight_supportiveness=round(parsed.insight_supportiveness / 2.0, 6),
            visualization_type=round(parsed.visualization_type / 2.0, 6),
            data_encoding=round(parsed.data_encoding / 2.0, 6),
            data_transformation=round(parsed.data_transformation / 2.0, 6),
            aesthetics=round(parsed.aesthetics / 2.0, 6),
            is_blank=bool(parsed.is_blank),
            weights={**weights, **({"is_blank": 1000.0} if parsed.is_blank else {})},
            rationales=rationales,
            details=details,
        )
