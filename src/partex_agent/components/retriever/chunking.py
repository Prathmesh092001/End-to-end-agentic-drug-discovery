from langchain_text_splitters import RecursiveCharacterTextSplitter

from partex_agent.entity import ChunkingConfig


class ContextualChunker:
    """
    Splits scientific documents into overlapping chunks, then prepends a
    short LLM-free context header (source, section) to each chunk so
    retrieval doesn't lose provenance once text is embedded in isolation.
    """

    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, text: str, metadata: dict) -> list[dict]:
        raw_chunks = self.splitter.split_text(text)
        source = metadata.get("source", "unknown")
        title = metadata.get("title", "")

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            header = f"[Source: {source} | {title}]\n" if title else f"[Source: {source}]\n"
            chunks.append(
                {
                    "text": header + chunk_text,
                    "metadata": {**metadata, "chunk_index": i, "chunk_count": len(raw_chunks)},
                }
            )
        return chunks
