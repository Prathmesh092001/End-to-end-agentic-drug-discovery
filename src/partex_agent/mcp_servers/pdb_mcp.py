"""
RCSB PDB MCP server - 3D structure and protein-ligand complex data,
used by the profiling agent for structural context.

Run standalone:
    python -m partex_agent.mcp_servers.pdb_mcp
"""
import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = "https://data.rcsb.org/rest/v1/core"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
mcp = FastMCP("pdb")


@mcp.tool()
async def search_structures(query_text: str, rows: int = 5) -> dict:
    """Full-text search across the PDB for structures matching a query (target, ligand, keyword)."""
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query_text},
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(SEARCH_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_structure_entry(pdb_id: str) -> dict:
    """Fetch metadata (resolution, method, organism, title) for a PDB entry ID."""
    url = f"{BASE_URL}/entry/{pdb_id}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
