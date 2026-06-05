from __future__ import annotations


class HybridRetrievalScorer:
    def __init__(self, *, method: str = "cc", semantic_weight: float = 0.1, rrf_k: float = 60.0) -> None:
        self.method = method.strip().lower()
        self.semantic_weight = max(0.0, min(float(semantic_weight), 1.0))
        self.rrf_k = max(float(rrf_k), 1.0)

    def score(self, semantic_scores: dict[str, float], lexical_scores: dict[str, float]) -> dict[str, float]:
        if self.method == "cc":
            return self._convex_combination(semantic_scores, lexical_scores)
        if self.method == "rrf":
            return self._reciprocal_rank_fusion(semantic_scores, lexical_scores)
        raise RuntimeError(f"Unsupported VisRAG hybrid method: {self.method!r}. Use cc or rrf.")

    def _convex_combination(self, semantic_scores: dict[str, float], lexical_scores: dict[str, float]) -> dict[str, float]:
        semantic_norm = self._minmax_normalize(semantic_scores)
        lexical_norm = self._minmax_normalize(lexical_scores)
        lexical_weight = 1.0 - self.semantic_weight
        keys = set(semantic_scores) | set(lexical_scores)
        return {
            key: lexical_weight * lexical_norm.get(key, 0.0) + self.semantic_weight * semantic_norm.get(key, 0.0)
            for key in keys
        }

    def _reciprocal_rank_fusion(self, semantic_scores: dict[str, float], lexical_scores: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for score_map in (semantic_scores, lexical_scores):
            ranked = sorted(score_map.items(), key=lambda item: (-float(item[1]), item[0]))
            for rank, (chunk_id, score) in enumerate(ranked, start=1):
                if score <= 0:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return scores

    @staticmethod
    def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
        positive = [float(value) for value in scores.values() if float(value) > 0]
        if not positive:
            return {key: 0.0 for key in scores}
        low = min(positive)
        high = max(positive)
        if high == low:
            return {key: 1.0 if float(value) > 0 else 0.0 for key, value in scores.items()}
        return {key: ((float(value) - low) / (high - low)) if float(value) > 0 else 0.0 for key, value in scores.items()}
