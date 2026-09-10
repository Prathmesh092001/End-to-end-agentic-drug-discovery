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
and retrieved literature, propose 2-3 testable research hypotheses.

STRICT GROUNDING & VERIFICATION CONSTRAINTS:
1. ONLY assert target binding, off-target interactions, or mechanistic links that are explicitly present in the provided UniProt, profiling, risk, or retrieval context.
2. DO NOT invent, extrapolate, or speculate on unverified off-target mechanisms (e.g., NQO2/QR2 binding or lysosomal sequestration) unless directly supported by the provided data.
3. Use appropriate uncertainty qualifiers for hypotheses (e.g., "hypothesized", "potentially", "may contribute to") rather than stating unverified mechanisms as definitive facts.
4. Attribute every hypothesis statement to specific, traceable supporting evidence present in the supplied context.
5. Keep explanations concise and strictly verifiable against the input data."""


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
        result: HypothesisOutput = self.invoke_with_retry(prompt)
        state["hypothesis_result"] = result.model_dump()
        return state