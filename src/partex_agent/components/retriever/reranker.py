from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-ranks a candidate set of retrieved chunks against the query for
    precision. Uses a local cross-encoder model (free, no API key, runs
    on CPU) instead of a paid reranking API - swap `reranker_model` in
    config.yaml for "rerank-english-v3.0" plus a Cohere key later if you
    want a hosted, higher-throughput reranker.
    """

    def __init__(self, model: str, top_n: int):
        self.model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        reranked = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return reranked[: self.top_n]
