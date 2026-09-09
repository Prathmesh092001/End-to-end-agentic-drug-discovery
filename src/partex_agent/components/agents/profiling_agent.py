from typing import List

from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors

from partex_agent.components.agents.base_agent import BaseAgent
from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.mcp_servers.mcp_client import call_tool_sync
from partex_agent.logging import logger


class ProfilingOutput(BaseModel):
    compound_name: str
    canonical_smiles: str = ""
    molecular_weight: float = 0.0
    logp: float = 0.0
    lipinski_violations: int = 0
    structural_notes: List[str] = Field(default_factory=list)
    summary: str


PROFILING_SYSTEM_PROMPT = """You are Partex's molecular profiling agent.
Given a compound or target name, the tool results retrieved for it, and any
grounding context, produce a concise structural/physicochemical profile.
Flag Lipinski Rule-of-Five violations explicitly. Never state a numeric
property you were not given by a tool - if a tool call failed, say so
instead of estimating from memory."""


class ProfilingAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg, output_schema=ProfilingOutput)

    def _rdkit_descriptors(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "h_donors": Descriptors.NumHDonors(mol),
            "h_acceptors": Descriptors.NumHAcceptors(mol),
        }

    def run(self, state: DrugIntelligenceState) -> DrugIntelligenceState:
        compound = state["compound_or_target"]
        logger.info(f"[profiling_agent] processing '{compound}'")

        pubchem_hits = call_tool_sync("pubchem", "search_compound", name_or_smiles=compound, by="name")
        cid = (pubchem_hits or {}).get("cids", [None])[0]
        props = call_tool_sync("pubchem", "get_compound_properties", cid=cid) if cid else {}
        smiles = props.get("CanonicalSMILES", "")
        rdkit_props = self._rdkit_descriptors(smiles) if smiles else {}

        prompt = f"""{PROFILING_SYSTEM_PROMPT}

Compound: {compound}
PubChem properties: {props}
RDKit descriptors: {rdkit_props}
Retrieved grounding context:
{state.get('retrieval_context', 'none')}
"""
        result: ProfilingOutput = self.llm.invoke(prompt)
        state["profiling_result"] = result.model_dump()
        return state
