from typing import List

from pydantic import BaseModel, Field

from partex_agent.components.agents.base_agent import BaseAgent
from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.logging import logger


class CriticVerdict(BaseModel):
    approved: bool
    groundedness_score: float = Field(ge=0.0, le=1.0)
    issues_found: List[str] = Field(default_factory=list)
    required_fixes: List[str] = Field(default_factory=list)


CRITIC_SYSTEM_PROMPT = """You are Partex's critic / evaluation agent - the
final safety gate before a drug-intelligence report reaches a scientist.
Check every specialist agent's output for: (1) claims not traceable to the
provided tool data or retrieval context, (2) internal contradictions
between agents, (3) missing uncertainty flags where data was sparse.
Set approved=false if any specialist output contains an unsupported
factual claim, especially about safety/toxicity. Be strict."""


class CriticAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg, output_schema=CriticVerdict)

    def run(self, state: DrugIntelligenceState) -> DrugIntelligenceState:
        logger.info("[critic_agent] evaluating combined agent output")

        prompt = f"""{CRITIC_SYSTEM_PROMPT}

Profiling result: {state.get('profiling_result')}
Risk result: {state.get('risk_result')}
Hypothesis result: {state.get('hypothesis_result')}
Competitive intel result: {state.get('competitive_intel_result')}
Retrieval context used: {state.get('retrieval_context', 'none')}
"""
        verdict: CriticVerdict = self.llm.invoke(prompt)
        state["critic_verdict"] = verdict.model_dump()
        if not verdict.approved:
            logger.warning(f"critic rejected output: {verdict.issues_found}")
        return state
