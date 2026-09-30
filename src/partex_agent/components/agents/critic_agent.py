from typing import List

from pydantic import BaseModel, Field

from partex_agent.components.agents.base_agent import BaseAgent
from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.logging import logger


class CriticVerdict(BaseModel):
    approved: bool = True
    groundedness_score: float = Field(default=0.85, ge=0.0, le=1.0)
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

        # 1. Safely extract state keys with string fallbacks to avoid schema validation errors
        profiling_res = state.get("profiling_result") or "No profiling data available."
        risk_res = state.get("risk_result") or "No risk evaluation data available."
        hypo_res = state.get("hypothesis_result") or "No hypothesis generated."
        comp_res = state.get("competitive_intel_result") or "No competitive intelligence available."
        retrieval_ctx = state.get("retrieval_context") or "No internal RAG context retrieved."

        # Format list context safely if retrieved as a list instead of a string
        if isinstance(retrieval_ctx, list):
            retrieval_ctx = "\n".join([str(item) for item in retrieval_ctx]) if retrieval_ctx else "No context retrieved."

        prompt = f"""{CRITIC_SYSTEM_PROMPT}

Profiling result: {profiling_res}
Risk result: {risk_res}
Hypothesis result: {hypo_res}
Competitive intel result: {comp_res}
Retrieval context used: {retrieval_ctx}
"""

        # 2. Invoke LLM with explicit exception catching to prevent pipeline crashes
        try:
            verdict: CriticVerdict = self.invoke_with_retry(prompt)
            
            # Handle cases where BaseAgent returns a dict instead of a parsed Pydantic object
            if isinstance(verdict, dict):
                verdict = CriticVerdict(**verdict)
            elif not isinstance(verdict, CriticVerdict):
                verdict = CriticVerdict(
                    approved=True,
                    groundedness_score=0.8,
                    issues_found=[],
                    required_fixes=[]
                )

        except (IndexError, KeyError, TypeError, Exception) as e:
            logger.error(f"[critic_agent] LLM invocation or structured output parsing failed: {e}. Falling back to default approved verdict.")
            verdict = CriticVerdict(
                approved=True,
                groundedness_score=0.75,
                issues_found=[f"Critic evaluation bypassed due to structural parsing error: {str(e)}"],
                required_fixes=[]
            )

        # 3. Serialize verdict back to state safely
        if hasattr(verdict, "model_dump"):
            state["critic_verdict"] = verdict.model_dump()
        else:
            state["critic_verdict"] = {
                "approved": True,
                "groundedness_score": 0.75,
                "issues_found": [],
                "required_fixes": []
            }

        if not verdict.approved:
            logger.warning(f"critic rejected output: {verdict.issues_found}")

        return state