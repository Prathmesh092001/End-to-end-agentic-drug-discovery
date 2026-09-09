from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class DrugIntelligenceState(TypedDict):
    """
    Shared state threaded through the entire LangGraph. Every agent reads
    what it needs and writes only its own field(s) - this keeps agents
    decoupled and makes the graph trivially inspectable/debuggable.
    """
    messages: Annotated[list, add_messages]

    # inputs
    query: str
    compound_or_target: str

    # per-agent outputs
    profiling_result: Optional[dict]
    risk_result: Optional[dict]
    hypothesis_result: Optional[dict]
    competitive_intel_result: Optional[dict]
    retrieval_context: Optional[str]

    # QA layer
    critic_verdict: Optional[dict]
    final_report: Optional[str]

    # routing
    next_agent: Optional[str]
