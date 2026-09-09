import pytest

from partex_agent.config.configuration import ConfigurationManager


def test_config_manager_loads():
    cm = ConfigurationManager()
    assert cm.config.llm.model_name
    assert cm.params.AGENTS.profiling_agent.temperature is not None


def test_mcp_server_configs_present():
    cm = ConfigurationManager()
    servers = cm.get_mcp_server_configs()
    for expected in ["pubchem", "chembl", "uniprot", "pdb", "tavily"]:
        assert expected in servers


def test_retrieval_config_shape():
    cm = ConfigurationManager()
    rc = cm.get_retrieval_config()
    assert rc.top_k > 0
    assert rc.vector_store.collection_name == "partex_knowledge_base"


@pytest.mark.skip(reason="requires live API keys and a running Qdrant instance")
def test_graph_end_to_end():
    from partex_agent.components.orchestrator.graph import run_pipeline

    result = run_pipeline(query="risk profile?", compound_or_target="imatinib")
    assert result["final_report"]
