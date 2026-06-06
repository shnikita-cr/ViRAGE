from types import SimpleNamespace

from src.application.config.settings import ViRAGESettings
from src.graph.builder import _route_after_spec_score, _route_analytics_tail, _route_semantic_decision


def test_analytics_tail_enabled_by_default() -> None:
    assert ViRAGESettings().analytics_tail_enabled is True


def test_graph_route_skips_tail_when_disabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=False))

    assert _route_analytics_tail({}, runtime=runtime) == "disabled"


def test_graph_route_keeps_tail_when_enabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=True))

    assert _route_analytics_tail({}, runtime=runtime) == "enabled"


def test_disabled_analytics_tail_keeps_semantic_loop_enabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=False, semantic_feedback_loop_enabled=True))

    assert _route_after_spec_score({}, runtime=runtime) == "semantic_loop"


def test_disabled_analytics_tail_completes_after_semantic_accept() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=False))

    assert _route_semantic_decision({"semantic_status": "accepted"}, runtime=runtime) == "accepted_done"
