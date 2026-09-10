from typing import List

from pydantic import BaseModel, Field

from partex_agent.components.agents.base_agent import BaseAgent
from partex_agent.components.agents.state import DrugIntelligenceState
from partex_agent.mcp_servers.mcp_client import call_tool_sync
from partex_agent.logging import logger


class CompetitiveAsset(BaseModel):
    name: str
    sponsor: str = ""
    stage: str = ""
    source_url: str = ""
    note: str = ""


class CompetitiveIntelOutput(BaseModel):
    landscape_summary: str = Field(
        description="Concise summary under 150 words of the competitive landscape."
    )
    competing_assets: List[CompetitiveAsset] = Field(
        default_factory=list,
        description="Top 5 competing assets maximum."
    )
    recency_note: str = Field(
        description="Brief note on how current this intel is, based on search result dates."
    )


COMPETITIVE_SYSTEM_PROMPT = """You are Partex's competitive intelligence
agent. Using live web search results, summarize the current competitive
landscape for the given compound/target: competing assets, development
stage, sponsors, and any recent patent or trial news. Only report facts
present in the search results - attribute every claim to a source URL,
and flag if results seem outdated or sparse.

STRICT JSON OUTPUT CONSTRAINTS TO PREVENT TRUNCATION:
1. Provide a maximum of 9 top competing assets in 'competing_assets'.
2. Keep 'landscape_summary' concise and under 500 words.
3. Keep 'note' and 'recency_note' brief (4-6 sentences each).
4. Ensure the output is valid, complete JSON without hitting length limits."""


class CompetitiveIntelAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg, output_schema=CompetitiveIntelOutput)

    def run(self, state: DrugIntelligenceState) -> DrugIntelligenceState:
        target = state["compound_or_target"]
        logger.info(f"[competitive_intel_agent] searching for '{target}'")

        search_results = call_tool_sync(
            "tavily",
            "web_search",
            query=f"{target} drug pipeline competitor patent clinical trial",
            search_depth="advanced",
            max_results=5,
        )

        prompt = f"""{COMPETITIVE_SYSTEM_PROMPT}

Compound/target: {target}
Web search results: {search_results}
"""
        result: CompetitiveIntelOutput = self.invoke_with_retry(prompt)
        state["competitive_intel_result"] = result.model_dump()
        return state