from __future__ import annotations

from src.services.verifier import VerifierService


def test_verifier_service_marks_supported_statements(reasoning_result, runtime) -> None:
    service = VerifierService()

    result = service.invoke(reasoning=reasoning_result, runtime=runtime)

    assert result.all_verified is True
    assert result.findings[0].status == "supported"
