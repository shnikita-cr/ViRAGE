from types import SimpleNamespace

from src.application.config.settings import ViRAGESettings
from src.graph.builder import _route_after_semantic_decision, _route_after_spec_score


def test_analytics_tail_enabled_by_default() -> None:
    assert ViRAGESettings().analytics_tail_enabled is True


def test_graph_skips_tail_when_disabled_and_semantic_loop_disabled() -> None:
    runtime = SimpleNamespace(
        settings=SimpleNamespace(analytics_tail_enabled=False, semantic_feedback_loop_enabled=False)
    )

    assert _route_after_spec_score({}, runtime=runtime) == "completed"


def test_graph_runs_semantic_retry_when_analytics_tail_disabled() -> None:
    runtime = SimpleNamespace(
        settings=SimpleNamespace(analytics_tail_enabled=False, semantic_feedback_loop_enabled=True)
    )

    assert _route_after_spec_score({}, runtime=runtime) == "semantic_loop"


def test_graph_keeps_tail_when_enabled_and_semantic_loop_disabled() -> None:
    runtime = SimpleNamespace(
        settings=SimpleNamespace(analytics_tail_enabled=True, semantic_feedback_loop_enabled=False)
    )

    assert _route_after_spec_score({}, runtime=runtime) == "analytics_tail"


def test_semantic_accept_finishes_when_analytics_tail_disabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=False))

    assert _route_after_semantic_decision({"semantic_status": "accepted"}, runtime=runtime) == "accepted_done"


def test_semantic_retry_is_preserved_when_analytics_tail_disabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=False))

    assert _route_after_semantic_decision({"semantic_status": "retry"}, runtime=runtime) == "retry"
