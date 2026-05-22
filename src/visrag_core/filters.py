from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult


def domain_semantics_gate(query_analysis: QueryRequestAnalysisResult, data_profile: DataProfile) -> bool:
    text = " ".join([
        query_analysis.normalized_query,
        " ".join(query_analysis.selected_fields),
        " ".join(column.name for column in data_profile.columns),
    ]).lower()
    domain_markers = {
        "biology", "biological", "medical", "clinical", "gene", "protein", "species",
        "hba1c", "crp", "ast", "alt", "bmi", "gdp", "inflation", "population",
        "finance", "economic", "economics", "ecology", "treatment", "diagnosis",
    }
    return any(marker in text for marker in domain_markers)
