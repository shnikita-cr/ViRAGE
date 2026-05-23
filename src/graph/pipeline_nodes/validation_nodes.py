from __future__ import annotations

from .common import *  # noqa: F401,F403


class ValidationPipelineNodesMixin:
    @staticmethod
    def _validation_attempt_payload(attempt_number: int, vega_spec, validation_result) -> dict:
        return {
            "generation_attempt_number": attempt_number,
            "is_valid": validation_result.is_valid,
            "spec_json": vega_spec.spec_json,
            "validated_spec": validation_result.validated_spec,
            "validation_errors": validation_result.validation_errors,
            "repair_hints": validation_result.repair_hints,
        }

    @staticmethod
    def _validation_attempt_report(attempt_number: int, payload: dict) -> str:
        spec_json = json.dumps(payload.get("spec_json", {}), ensure_ascii=False, indent=2, default=str)
        validated_spec = json.dumps(payload.get("validated_spec", {}), ensure_ascii=False, indent=2, default=str)
        errors = payload.get("validation_errors") or []
        hints = payload.get("repair_hints") or []
        error_block = "\n".join(f"- {item}" for item in errors) or "- none"
        hint_block = "\n".join(f"- {item}" for item in hints) or "- none"
        return (
            f"# Spec validation attempt {attempt_number:03d}\n\n"
            f"## Status\n\nvalid = `{payload.get('is_valid')}`\n\n"
            f"## Vega-Lite spec sent to validator\n\n```json\n{spec_json}\n```\n\n"
            f"## Validator normalized spec\n\n```json\n{validated_spec}\n```\n\n"
            f"## Errors\n\n{error_block}\n\n"
            f"## Repair hints\n\n{hint_block}\n"
        )

    @traceable(name="virage.spec_validator")
    def spec_validator_node(self, state: PipelineState) -> dict:
        attempt_number = max(1, int(state.get("technical_attempt_number", 1) or 1))
        current_spec = state["vega_spec"]
        try:
            validation_result = self.spec_validator.invoke(current_spec, runtime=self.runtime)
        except TypeError as exc:
            if "runtime" not in str(exc):
                raise
            validation_result = self.spec_validator.invoke(current_spec)
        payload = self._validation_attempt_payload(attempt_number, current_spec, validation_result)
        artifact_paths = dict(state.get("artifact_paths", {}))
        artifact_paths = self._save_attempt_into(artifact_paths, state, "spec_validation", payload)

        return {
            "spec_validation": validation_result,
            "technical_status": "ok" if validation_result.is_valid else "validation_failed",
            "stage": PipelineStage.SPEC_VALIDATION,
            "trace": self._trace(state, "spec_validator"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="spec_validator",
                title="Specification validation",
                summary=f"valid={validation_result.is_valid}; technical attempt {attempt_number}",
                inputs=["vega_spec"],
                outputs=["validated_spec" if validation_result.is_valid else "validation_errors"],
                details={
                    "artifact": artifact_paths.get("spec_validation"),
                    "attempt_count": attempt_number,
                    "max_generation_attempts": int(self.runtime.settings.spec_generation_max_attempts),
                    "validated_spec": validation_result.validated_spec,
                    "validation_errors": validation_result.validation_errors,
                    "repair_hints": validation_result.repair_hints,
                },
            ),
        }

    @traceable(name="virage.technical_decision")
    def technical_decision_node(self, state: PipelineState) -> dict:
        validation = state["spec_validation"]
        attempt_number = max(1, int(state.get("technical_attempt_number", 1) or 1))
        max_attempts = max(1, int(self.runtime.settings.spec_generation_max_attempts))
        artifact_paths = dict(state.get("artifact_paths", {}))

        if validation.is_valid:
            summary = {
                "status": "ok",
                "succeeded": True,
                "completed_generation_attempts": attempt_number,
                "max_generation_attempts": max_attempts,
            }
            artifact_paths = self._save_attempt_into(artifact_paths, state, "technical_decision", summary)
            return {
                "technical_status": "ok",
                "stage": PipelineStage.SPEC_VALIDATION,
                "trace": self._trace(state, "technical_decision"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="technical_decision",
                    title="Technical retry decision",
                    summary="technical validation accepted",
                    outputs=["ok"],
                    details=summary,
                ),
            }

        feedback = {
            "validation_errors": validation.validation_errors,
            "repair_hints": validation.repair_hints,
            "invalid_spec": state["vega_spec"].spec_json,
        }
        if attempt_number < max_attempts:
            summary = {
                "status": "retry",
                "succeeded": False,
                "completed_generation_attempts": attempt_number,
                "next_generation_attempt": attempt_number + 1,
                "max_generation_attempts": max_attempts,
                "validation_errors": validation.validation_errors,
                "repair_hints": validation.repair_hints,
            }
            artifact_paths = self._save_attempt_into(artifact_paths, state, "technical_decision", summary)
            return {
                "technical_status": "retry",
                "technical_attempt_number": attempt_number + 1,
                "technical_retry_feedback": feedback,
                "stage": PipelineStage.SPEC_VALIDATION,
                "trace": self._trace(state, "technical_decision_retry"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="technical_decision",
                    title="Technical retry decision",
                    summary=f"retry spec generation {attempt_number + 1}/{max_attempts}",
                    outputs=["retry"],
                    details=summary,
                ),
            }

        summary = {
            "status": "failed",
            "succeeded": False,
            "completed_generation_attempts": attempt_number,
            "max_generation_attempts": max_attempts,
            "validation_errors": validation.validation_errors,
            "repair_hints": validation.repair_hints,
        }
        artifact_paths = self._save_attempt_into(artifact_paths, state, "technical_decision", summary)
        self.runtime.save_text_artifact(
            "errors/spec_validation_failed.txt",
            "\n".join(validation.validation_errors),
            run_id=state["run_id"],
            numbered=True,
        )
        raise RuntimeError(
            f"Specification validation failed after {max_attempts} graph-level generation attempt(s). "
            "See spec_validation_attempt_* artifacts for spec code, errors and repair hints."
        )

    @traceable(name="virage.vegalite_plot_drawing")
    def vegalite_plot_drawing_node(self, state: PipelineState) -> dict:
        result = self.vegalite_plot_drawing.invoke(state["spec_validation"], state["run_id"], runtime=self.runtime)
        artifact_paths = self._save(state, "plot_rendering", result.model_dump())
        return {
            "plot_rendering": result,
            "plot_image": result.plot_image.model_dump(),
            "stage": PipelineStage.PLOT_RENDERING,
            "trace": self._trace(state, "vegalite_plot_drawing"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="vegalite_plot_drawing",
                title="Plot rendering",
                summary=result.plot_image.image_path,
                inputs=["validated_spec"],
                outputs=result.render_notes[:5],
                details={"artifact": artifact_paths["plot_rendering"],
                         "rendered_scenegraph": result.rendered_scenegraph},
            ),
        }

    @traceable(name="virage.scenegraph_check")
    def scenegraph_check_node(self, state: PipelineState) -> dict:
        result = self.scenegraph_check.invoke(state["plot_rendering"])
        artifact_paths = self._save(state, "scenegraph_check", result.model_dump())
        return {
            "scenegraph_check": result,
            "stage": PipelineStage.SCENEGRAPH_CHECK,
            "trace": self._trace(state, "scenegraph_check"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="scenegraph_check",
                title="Scenegraph check",
                summary="Scenegraph inspected",
                inputs=["rendered_scenegraph"],
                outputs=result.notes[:5],
                details={"artifact": artifact_paths["scenegraph_check"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.empty_chart_check")
    def empty_chart_check_node(self, state: PipelineState) -> dict:
        result = self.empty_chart_check.invoke(state["scenegraph_check"])
        artifact_paths = self._save(state, "empty_chart_check", result.model_dump())
        if result.empty_chart_signal:
            self.runtime.save_text_artifact("errors/empty_chart_detected.txt", result.empty_chart_status,
                                            run_id=state["run_id"], numbered=True)
            raise RuntimeError(
                "Rendered chart is empty or unusable. See run artifacts and errors directory for numbered details.")

        live_preview = _build_live_chart_preview_payload(state, result)

        return {
            "empty_chart_check": result,
            "stage": PipelineStage.EMPTY_CHART_CHECK,
            "trace": self._trace(state, "empty_chart_check"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="empty_chart_check",
                title="Empty chart check",
                summary=result.empty_chart_status,
                inputs=["scenegraph_status"],
                outputs=[str(result.empty_chart_signal), "live_preview_ready"],
                details={
                    "artifact": artifact_paths["empty_chart_check"],
                    "live_chart_preview": live_preview,
                    **result.model_dump(),
                },
            ),
        }

    @traceable(name="virage.spec_score")
    def spec_score_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_spec_score:
            return {}
        ground_truth = state.get("user_context", {}).get("ground_truth_spec")
        if not isinstance(ground_truth, dict):
            return {}
        result = self.spec_score.invoke(
            state["spec_validation"],
            ground_truth,
            user_prompt=state.get("query"),
            empty_chart_check=state.get("empty_chart_check"),
        )
        artifact_paths = self._save(state, "structural_spec_metric", result.model_dump())
        return {
            "structural_spec_metric": result,
            "stage": PipelineStage.EVALUATION,
            "trace": self._trace(state, "spec_score"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="spec_score",
                title="Structural metric",
                summary=f"score={result.score:.3f}",
                inputs=["validated spec", "ground truth spec"],
                outputs=result.details[:3],
                details={"artifact": artifact_paths["structural_spec_metric"], **result.model_dump()},
            ),
        }
