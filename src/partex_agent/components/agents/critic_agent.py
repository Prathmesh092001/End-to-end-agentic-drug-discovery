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


CRITIC_SYSTEM_PROMPT = """You are Partex's groundedness critic agent - the final safety gate before a drug-intelligence report reaches a scientist.
Your job is to evaluate whether the specialist agents' claims are grounded in provided evidence.

GROUNDING RULES:
1. Valid sources of evidence include BOTH:
   a) The internal retrieval context (RAG chunks).
   b) Live tool execution results (PubChem, ChEMBL, UniProt, and Tavily web searches) embedded or referenced in the specialist agent outputs.
2. If a claim (such as transporters, mechanisms, bioactivities, or market landscape) is supported by Tavily search results, PubChem properties, UniProt records, or ChEMBL assays included in the agent outputs, mark it as VALID/GROUNDED.
3. Reject claims (set approved=false) ONLY if they are complete fabrications unsupported by either RAG context OR live tool outputs, or if there are internal contradictions between agents.
4. Flag missing uncertainty language only where claims are speculative hypotheses presented as definitive facts without hedging.
"""


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
        verdict: CriticVerdict = self.invoke_with_retry(prompt)
        state["critic_verdict"] = verdict.model_dump()
        if not verdict.approved:
            logger.warning(f"critic rejected output: {verdict.issues_found}")
        return state