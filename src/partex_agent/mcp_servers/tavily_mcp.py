"""
Tavily MCP server - real-time web search, used by the competitive
intelligence agent to scan for competing assets, patents, and news
that pre-trained knowledge and static databases won't have.

Requires TAVILY_API_KEY in the environment.

Run standalone:
    python -m partex_agent.mcp_servers.tavily_mcp
"""
import os

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

from partex_agent.logging import logger

mcp = FastMCP("tavily")
_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ["TAVILY_API_KEY"]
        _client = TavilyClient(api_key=api_key)
    return _client


@mcp.tool()
def web_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    include_domains: list[str] | None = None,
) -> dict:
    """
    Run a web search for competitive intelligence: competing drug assets,
    patent filings, clinical trial announcements, pipeline news.

    Args:
        query: search query, e.g. "KRAS G12C inhibitor pipeline 2026"
        search_depth: "basic" or "advanced"
        max_results: number of results to return
        include_domains: optional allowlist, e.g. ["clinicaltrials.gov", "fiercebiotech.com"]
    """
    client = _get_client()
    result = client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_domains=include_domains or [],
    )
    logger.info(f"tavily search '{query}' -> {len(result.get('results', []))} results")
    return result


@mcp.tool()
def extract_page(url: str) -> dict:
    """Extract the clean text content of a specific web page for deeper reading."""
    client = _get_client()
    return client.extract(urls=[url])


if __name__ == "__main__":
    mcp.run(transport="stdio")
