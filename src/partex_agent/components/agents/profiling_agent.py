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
        if not smiles:
            return {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            
            # Calculate Lipinski violations
            violations = 0
            if mw > 500:
                violations += 1
            if logp > 5:
                violations += 1
            if h_donors > 5:
                violations += 1
            if h_acceptors > 10:
                violations += 1

            return {
                "molecular_weight": round(mw, 2),
                "logp": round(logp, 2),
                "h_donors": h_donors,
                "h_acceptors": h_acceptors,
                "lipinski_violations": violations,
            }
        except Exception as e:
            logger.warning(f"[profiling_agent] RDKit descriptor calculation error: {e}")
            return {}

    def run(self, state: DrugIntelligenceState) -> DrugIntelligenceState:
        compound = state.get("compound_or_target", "")
        logger.info(f"[profiling_agent] processing '{compound}'")

        # 1. Safely search PubChem
        pubchem_hits = call_tool_sync("pubchem", "search_compound", name_or_smiles=compound, by="name")
        
        # Safe extraction: prevents IndexError if cids key exists as an empty list []
        cids = (pubchem_hits or {}).get("cids", [])
        cid = cids[0] if cids else None

        # 2. Fetch properties if CID exists
        props = call_tool_sync("pubchem", "get_compound_properties", cid=cid) if cid else {}
        smiles = props.get("CanonicalSMILES", "") if isinstance(props, dict) else ""
        
        # 3. Compute RDKit descriptors if SMILES is available
        rdkit_props = self._rdkit_descriptors(smiles) if smiles else {}

        # 4. Construct prompt safely
        retrieval_context = state.get("retrieval_context", "none")
        if isinstance(retrieval_context, list):
            retrieval_context = "\n".join([str(item) for item in retrieval_context])

        prompt = f"""{PROFILING_SYSTEM_PROMPT}

Compound/Target: {compound}
PubChem search status: {"Found CID " + str(cid) if cid else "No small molecule CID found (may be a biologic target)"}
PubChem properties: {props if props else "None"}
RDKit descriptors: {rdkit_props if rdkit_props else "None"}
Retrieved grounding context:
{retrieval_context}
"""

        # 5. Invoke LLM with safety fallbacks
        try:
            result: ProfilingOutput = self.invoke_with_retry(prompt)
            if isinstance(result, dict):
                result = ProfilingOutput(**result)
        except Exception as e:
            logger.error(f"[profiling_agent] Execution or structured parsing failed: {e}")
            result = ProfilingOutput(
                compound_name=compound,
                canonical_smiles=smiles,
                molecular_weight=rdkit_props.get("molecular_weight", 0.0),
                logp=rdkit_props.get("logp", 0.0),
                lipinski_violations=rdkit_props.get("lipinski_violations", 0),
                structural_notes=[f"Profiling notice: Target/Compound evaluated with partial external tool data."],
                summary=f"Profile generated for target/compound '{compound}'. External tool lookup yielded no small-molecule CID."
            )

        # 6. Save back to state
        if hasattr(result, "model_dump"):
            state["profiling_result"] = result.model_dump()
        else:
            state["profiling_result"] = {
                "compound_name": compound,
                "canonical_smiles": smiles,
                "molecular_weight": 0.0,
                "logp": 0.0,
                "lipinski_violations": 0,
                "structural_notes": [],
                "summary": f"Profile completed for {compound}."
            }

        return state