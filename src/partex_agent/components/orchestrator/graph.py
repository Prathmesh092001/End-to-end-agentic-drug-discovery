from langgraph.graph import StateGraph, END

from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.components.agents.profiling_agent import ProfilingAgent
from partex_agent.components.agents.risk_agent import RiskAgent
from partex_agent.components.agents.hypothesis_agent import HypothesisAgent
from partex_agent.components.agents.competitive_intel_agent import CompetitiveIntelAgent
from partex_agent.components.agents.critic_agent import CriticAgent
from partex_agent.components.retriever.retrieval_agent import RetrievalAgent
from partex_agent.config.configuration import ConfigurationManager
from partex_agent.logging import logger


def _retrieval_node(retrieval_agent: RetrievalAgent):
    def node(state: DrugIntelligenceState) -> DrugIntelligenceState:
        state["retrieval_context"] = retrieval_agent.as_context_block(state["query"])
        return state

    return node


def _compile_report_node(state: DrugIntelligenceState) -> DrugIntelligenceState:
    verdict = state.get("critic_verdict", {})
    if not verdict.get("approved", False):
        state["final_report"] = (
            "This report did not pass the internal groundedness check and was withheld.\n"
            f"Issues found: {verdict.get('issues_found')}\n"
            f"Required fixes: {verdict.get('required_fixes')}"
        )
        return state

    p = state.get("profiling_result", {})
    r = state.get("risk_result", {})
    h = state.get("hypothesis_result", {})
    c = state.get("competitive_intel_result", {})

    state["final_report"] = f"""# Drug intelligence report: {state['compound_or_target']}

## Molecular profile
{p.get('summary', 'n/a')}

## Risk assessment
Overall risk: {r.get('overall_risk_level', 'n/a')} (confidence {r.get('confidence', 'n/a')})
{r.get('rationale', '')}

## Hypotheses
{chr(10).join('- ' + h_['statement'] for h_ in h.get('hypotheses', []))}

## Competitive landscape
{c.get('landscape_summary', 'n/a')}

---
Groundedness score: {verdict.get('groundedness_score', 'n/a')}
"""
    return state


def build_graph(recursion_limit: int = 25):
    """
    Builds the supervisor graph:

        retrieval -> [profiling, risk] -> hypothesis -> competitive_intel
                  -> critic -> compile_report -> END

    Profiling and risk run before hypothesis generation because the
    hypothesis agent depends on both. Competitive intel runs independently
    since it only needs the raw compound/target name, not internal results.
    """
    config_manager = ConfigurationManager()

    retrieval_agent = RetrievalAgent(config_manager.get_retrieval_config())
    profiling_agent = ProfilingAgent(config_manager.get_agent_config("profiling_agent"))
    risk_agent = RiskAgent(config_manager.get_agent_config("risk_agent"))
    hypothesis_agent = HypothesisAgent(config_manager.get_agent_config("hypothesis_agent"))
    intel_agent = CompetitiveIntelAgent(config_manager.get_agent_config("competitive_intel_agent"))
    critic_agent = CriticAgent(config_manager.get_agent_config("critic_agent"))

    graph = StateGraph(DrugIntelligenceState)

    graph.add_node("retrieval", _retrieval_node(retrieval_agent))
    graph.add_node("profiling", profiling_agent.run)
    graph.add_node("risk", risk_agent.run)
    graph.add_node("hypothesis", hypothesis_agent.run)
    graph.add_node("competitive_intel", intel_agent.run)
    graph.add_node("critic", critic_agent.run)
    graph.add_node("compile_report", _compile_report_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "profiling")
    graph.add_edge("profiling", "risk")
    graph.add_edge("risk", "hypothesis")
    graph.add_edge("hypothesis", "competitive_intel")
    graph.add_edge("competitive_intel", "critic")
    graph.add_edge("critic", "compile_report")
    graph.add_edge("compile_report", END)

    logger.info("agent graph compiled: retrieval -> profiling -> risk -> hypothesis -> competitive_intel -> critic -> report")
    return graph.compile()


def run_pipeline(query: str, compound_or_target: str) -> DrugIntelligenceState:
    app = build_graph()
    initial_state: DrugIntelligenceState = {
        "messages": [],
        "query": query,
        "compound_or_target": compound_or_target,
        "profiling_result": None,
        "risk_result": None,
        "hypothesis_result": None,
        "competitive_intel_result": None,
        "retrieval_context": None,
        "critic_verdict": None,
        "final_report": None,
        "next_agent": None,
    }
    return app.invoke(initial_state)
