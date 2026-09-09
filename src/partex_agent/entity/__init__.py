from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class MCPServerConfig:
    """Config for one MCP tool server (one external data source)."""
    name: str
    base_url: str


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str
    host: str
    port: int
    collection_name: str
    embedding_model: str
    embedding_dim: int


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    strategy: str


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    rerank_top_n: int
    reranker_model: str
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class AgentConfig:
    """Per-agent runtime parameters, pulled from params.yaml."""
    name: str
    temperature: float
    llm: LLMConfig


@dataclass(frozen=True)
class EvaluationConfig:
    ragas_metrics: List[str]
    judge_model: str
    groundedness_threshold: float


@dataclass(frozen=True)
class OrchestrationConfig:
    max_turns: int
    recursion_limit: int
    parallel_specialists: bool


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class ServingConfig:
    host: str
    port: int
