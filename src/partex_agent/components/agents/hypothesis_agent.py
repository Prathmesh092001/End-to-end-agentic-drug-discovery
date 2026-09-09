from typing import List

from pydantic import BaseModel, Field

from partex_agent.components.agents.base_agent import BaseAgent
from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.mcp_servers.mcp_client import call_tool_sync
from partex_agent.logging import logger


class Hypothesis(BaseModel):
    statement: str
    supporting_evidence: List[str] = Field(default_factory=list)
    testability: str = Field(description="how a wet-lab or computational team could test this")
    novelty_rationale: str


class HypothesisOutput(BaseModel):
    hypotheses: List[Hypothesis]
    target_context: str = ""


HYPOTHESIS_SYSTEM_PROMPT = """You are Partex's hypothesis generation agent.
Using the profiling and risk agent outputs, the target's protein context,
and retrieved literature, propose 2-3 novel, testable research hypotheses.
Every hypothesis must cite specific supporting evidence from what you were
given - never invent a mechanism that isn't traceable to the provided data
or retrieval context. Prefer hypotheses that are non-obvious but
defensible over safe, generic statements."""


class HypothesisAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg, output_schema=HypothesisOutput)

    def run(self, state: DrugIntelligenceState) -> DrugIntelligenceState:
        target = state["compound_or_target"]
        logger.info(f"[hypothesis_agent] processing '{target}'")

        uniprot_hits = call_tool_sync("uniprot", "search_protein", query=target, size=3)

        prompt = f"""{HYPOTHESIS_SYSTEM_PROMPT}

Compound/target: {target}
Profiling agent output: {state.get('profiling_result')}
Risk agent output: {state.get('risk_result')}
UniProt target context: {uniprot_hits}
Retrieved grounding context:
{state.get('retrieval_context', 'none')}
"""
        result: HypothesisOutput = self.llm.invoke(prompt)
        state["hypothesis_result"] = result.model_dump()
        return state
