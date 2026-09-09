from partex_agent.components.retriever.vector_store import KnowledgeBaseStore
from partex_agent.components.retriever.reranker import Reranker
from partex_agent.entity import RetrievalConfig
from partex_agent.logging import logger


class RetrievalAgent:
    """
    Agent #2 in the graph. Given the user's query (and any specialist's
    follow-up query), fetches grounding evidence from the internal
    knowledge base before any specialist agent reasons - this is what
    keeps profiling/risk/hypothesis outputs citation-backed instead of
    purely parametric LLM knowledge.
    """

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.store = KnowledgeBaseStore(cfg.vector_store)
        self.reranker = Reranker(cfg.reranker_model, cfg.rerank_top_n)

    def retrieve(self, query: str, source_filter: str | None = None) -> list[dict]:
        candidates = self.store.search(query, top_k=self.cfg.top_k, source_filter=source_filter)
        reranked = self.reranker.rerank(query, candidates)
        logger.info(f"retrieval for '{query[:60]}...' -> {len(reranked)} grounded chunks")
        return reranked

    def as_context_block(self, query: str, source_filter: str | None = None) -> str:
        """Formats retrieved chunks as a single context block ready to inject into a prompt."""
        chunks = self.retrieve(query, source_filter)
        if not chunks:
            return "No grounding evidence found in the knowledge base for this query."
        return "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
