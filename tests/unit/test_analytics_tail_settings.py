from types import SimpleNamespace

from src.application.config.settings import ViRAGESettings
from src.graph.builder import _route_analytics_tail


def test_analytics_tail_enabled_by_default() -> None:
    assert ViRAGESettings().analytics_tail_enabled is True


def test_graph_route_skips_tail_when_disabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=False))

    assert _route_analytics_tail({}, runtime=runtime) == "disabled"


def test_graph_route_keeps_tail_when_enabled() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(analytics_tail_enabled=True))

    assert _route_analytics_tail({}, runtime=runtime) == "enabled"
