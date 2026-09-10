from typing import List

from pydantic import BaseModel, Field

from partex_agent.components.agents.base_agent import BaseAgent
from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.mcp_servers.mcp_client import call_tool_sync
from partex_agent.logging import logger


class RiskOutput(BaseModel):
    overall_risk_level: str = Field(description="low | medium | high | insufficient_data")
    admet_flags: List[str] = Field(default_factory=list)
    known_mechanisms: List[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


RISK_SYSTEM_PROMPT = """You are Partex's risk scoring agent for early drug
discovery. Given ChEMBL bioactivity and mechanism data plus the profiling
agent's structural output, assess ADMET-relevant risk. Be conservative:
if bioactivity data is sparse, set overall_risk_level to
'insufficient_data' rather than guessing, and say so in the rationale."""


class RiskAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg, output_schema=RiskOutput)

    def run(self, state: DrugIntelligenceState) -> DrugIntelligenceState:
        compound = state["compound_or_target"]
        logger.info(f"[risk_agent] processing '{compound}'")

        chembl_hits = call_tool_sync("chembl", "search_molecule", query=compound)
        molecules = (chembl_hits or {}).get("molecules", [])
        chembl_id = molecules[0]["molecule_chembl_id"] if molecules else None

        mechanism = call_tool_sync("chembl", "get_molecule_mechanism", chembl_id=chembl_id) if chembl_id else {}
        activity = call_tool_sync("chembl", "get_activity_data", chembl_id=chembl_id, limit=15) if chembl_id else {}

        prompt = f"""{RISK_SYSTEM_PROMPT}

Compound: {compound}
Profiling agent output: {state.get('profiling_result')}
ChEMBL mechanism data: {mechanism}
ChEMBL activity data (truncated): {str(activity)[:3000]}
Retrieved grounding context:
{state.get('retrieval_context', 'none')}
"""
        result: RiskOutput = self.invoke_with_retry(prompt)
        state["risk_result"] = result.model_dump()
        return state
