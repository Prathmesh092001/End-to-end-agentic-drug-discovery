import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_huggingface import HuggingFaceEmbeddings

from partex_agent.entity import VectorStoreConfig
from partex_agent.logging import logger


class KnowledgeBaseStore:
    """
    Thin wrapper around Qdrant that handles collection creation, embedding
    via the configured embedding model, and hybrid search (dense vector +
    keyword filter on metadata). Swappable for pgvector/Pinecone by
    changing `vector_store.provider` in config.yaml without touching
    any agent code, since agents only ever call `.search()`.

    Uses a local HuggingFace sentence-transformer for embeddings by
    default (free, no API key, runs on CPU) - swap `embedding_model` in
    config.yaml for an OpenAI/Cohere model later if you want higher
    quality embeddings and have a paid key.
    """

    def __init__(self, cfg: VectorStoreConfig):
        self.cfg = cfg
        self.client = QdrantClient(host=cfg.host, port=cfg.port)
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.cfg.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.cfg.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.cfg.embedding_dim, distance=qmodels.Distance.COSINE
                ),
            )
            logger.info(f"created qdrant collection: {self.cfg.collection_name}")

    def upsert_chunks(self, chunks: list[dict]):
        texts = [c["text"] for c in chunks]
        vectors = self.embeddings.embed_documents(texts)
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk["text"], **chunk["metadata"]},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.cfg.collection_name, points=points)
        logger.info(f"upserted {len(points)} chunks into {self.cfg.collection_name}")

    def search(self, query: str, top_k: int, source_filter: str | None = None) -> list[dict]:
        query_vector = self.embeddings.embed_query(query)
        qfilter = None
        if source_filter:
            qfilter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source_filter))]
            )
        hits = self.client.search(
            collection_name=self.cfg.collection_name,
            query_vector=query_vector,
            query_filter=qfilter,
            limit=top_k,
        )
        return [{"text": h.payload["text"], "score": h.score, "metadata": h.payload} for h in hits]
