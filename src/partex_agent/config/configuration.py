from partex_agent.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from partex_agent.utils.common import read_yaml, create_directories
from partex_agent.entity import (
    MCPServerConfig,
    VectorStoreConfig,
    ChunkingConfig,
    RetrievalConfig,
    LLMConfig,
    AgentConfig,
    EvaluationConfig,
    OrchestrationConfig,
    MLflowConfig,
    ServingConfig,
)


class ConfigurationManager:
    """
    Mirrors the reference repo's ConfigurationManager: reads the two YAML
    files once, exposes typed getters that every component/agent depends on.
    Nothing downstream ever reads a raw yaml file directly.
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_mcp_server_configs(self) -> dict:
        sources = self.config.data_sources
        return {
            name: MCPServerConfig(name=name, base_url=cfg.get("base_url", ""))
            for name, cfg in sources.items()
            if "base_url" in cfg
        }

    def get_llm_config(self) -> LLMConfig:
        c = self.config.llm
        return LLMConfig(
            provider=c.provider,
            model_name=c.model_name,
            temperature=c.temperature,
            max_tokens=c.max_tokens,
        )

    def get_retrieval_config(self) -> RetrievalConfig:
        vs = self.config.vector_store
        ch = self.config.chunking
        r = self.config.retrieval
        return RetrievalConfig(
            top_k=r.top_k,
            rerank_top_n=r.rerank_top_n,
            reranker_model=r.reranker_model,
            vector_store=VectorStoreConfig(
                provider=vs.provider,
                host=vs.host,
                port=vs.port,
                collection_name=vs.collection_name,
                embedding_model=vs.embedding_model,
                embedding_dim=vs.embedding_dim,
            ),
            chunking=ChunkingConfig(
                chunk_size=ch.chunk_size,
                chunk_overlap=ch.chunk_overlap,
                strategy=ch.strategy,
            ),
        )

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        base_llm = self.get_llm_config()
        agent_params = self.params.AGENTS.get(agent_name, {})
        return AgentConfig(
            name=agent_name,
            temperature=agent_params.get("temperature", base_llm.temperature),
            llm=LLMConfig(
                provider=base_llm.provider,
                model_name=base_llm.model_name,
                temperature=agent_params.get("temperature", base_llm.temperature),
                max_tokens=base_llm.max_tokens,
            ),
        )

    def get_orchestration_config(self) -> OrchestrationConfig:
        o = self.params.ORCHESTRATION
        return OrchestrationConfig(
            max_turns=o.max_turns,
            recursion_limit=o.recursion_limit,
            parallel_specialists=o.parallel_specialists,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        e = self.config.evaluation
        return EvaluationConfig(
            ragas_metrics=list(e.ragas_metrics),
            judge_model=e.judge_model,
            groundedness_threshold=e.groundedness_threshold,
        )

    def get_mlflow_config(self) -> MLflowConfig:
        m = self.config.mlflow
        return MLflowConfig(tracking_uri=m.tracking_uri, experiment_name=m.experiment_name)

    def get_serving_config(self) -> ServingConfig:
        s = self.config.serving
        return ServingConfig(host=s.host, port=s.port)
