from partex_agent.config.configuration import ConfigurationManager
from partex_agent.components.retriever.chunking import ContextualChunker
from partex_agent.components.retriever.vector_store import KnowledgeBaseStore
from partex_agent.pipeline.stage_01_ingest import run_ingestion_stage
from partex_agent.utils.common import timed
from partex_agent.logging import logger


@timed
def run_indexing_stage():
    config_manager = ConfigurationManager()
    retrieval_cfg = config_manager.get_retrieval_config()

    docs = run_ingestion_stage()
    if not docs:
        logger.warning("no documents to index - skipping")
        return

    chunker = ContextualChunker(retrieval_cfg.chunking)
    store = KnowledgeBaseStore(retrieval_cfg.vector_store)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc["text"], doc["metadata"]))

    store.upsert_chunks(all_chunks)
    logger.info(f"indexing stage complete: {len(all_chunks)} chunks indexed")


if __name__ == "__main__":
    run_indexing_stage()
