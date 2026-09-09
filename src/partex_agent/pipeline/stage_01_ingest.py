from pathlib import Path

from partex_agent.config.configuration import ConfigurationManager
from partex_agent.utils.common import timed
from partex_agent.logging import logger


@timed
def run_ingestion_stage(source_dir: str = "data/internal_kb") -> list[dict]:
    """
    Reads raw text/markdown documents from the internal knowledge base
    directory and returns them as (text, metadata) records ready for
    chunking. In production this would also pull from Confluence/SharePoint
    connectors or a literature feed - kept file-based here for portability.
    """
    ConfigurationManager()  # validates config.yaml is well-formed before proceeding
    docs = []
    root = Path(source_dir)
    if not root.exists():
        logger.warning(f"{source_dir} does not exist yet - nothing to ingest")
        return docs

    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append({"text": text, "metadata": {"source": "internal_kb", "title": path.stem}})

    logger.info(f"ingested {len(docs)} documents from {source_dir}")
    return docs


if __name__ == "__main__":
    run_ingestion_stage()
