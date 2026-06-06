from __future__ import annotations

_ROLE_ALIASES = {
    "number": "quantitative",
    "numeric": "quantitative",
    "measure": "quantitative",
    "quantitative": "quantitative",
    "int": "quantitative",
    "int64": "quantitative",
    "int32": "quantitative",
    "integer": "quantitative",
    "float": "quantitative",
    "float64": "quantitative",
    "float32": "quantitative",
    "double": "quantitative",
    "decimal": "quantitative",
    "date": "temporal",
    "time": "temporal",
    "datetime": "temporal",
    "datetime64": "temporal",
    "datetime64[ns]": "temporal",
    "timestamp": "temporal",
    "temporal": "temporal",
    "year": "temporal",
    "month": "temporal",
    "string": "nominal",
    "str": "nominal",
    "object": "nominal",
    "category": "nominal",
    "categorical": "nominal",
    "dimension": "nominal",
    "nominal": "nominal",
    "ordinal": "ordinal",
    "bool": "boolean",
    "boolean": "boolean",
}

_VEGA_TYPES = {
    "quantitative": "quantitative",
    "temporal": "temporal",
    "nominal": "nominal",
    "ordinal": "ordinal",
    "boolean": "nominal",
}


def normalize_semantic_type(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _ROLE_ALIASES.get(text, text)


def semantic_type_from_role_or_dtype(role: str | None, semantic_type: str | None, raw_dtype: str | None = None) -> str:
    for value in (role, semantic_type, raw_dtype):
        normalized = normalize_semantic_type(value)
        if normalized:
            return normalized
    return ""


def vega_type(value: str | None) -> str:
    normalized = normalize_semantic_type(value)
    return _VEGA_TYPES.get(normalized, normalized or "nominal")
