"""
ChEMBL MCP server - bioactivity and drug-mechanism data, used mainly
by the risk scoring agent for ADMET-relevant signal.

Run standalone:
    python -m partex_agent.mcp_servers.chembl_mcp
"""
import httpx
from mcp.server.fastmcp import FastMCP

from partex_agent.logging import logger

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
mcp = FastMCP("chembl")


@mcp.tool()
async def search_molecule(query: str) -> dict:
    """Search ChEMBL for a molecule by name or synonym."""
    url = f"{BASE_URL}/molecule/search"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params={"q": query, "format": "json"})
        resp.raise_for_status()
        data = resp.json()
    molecules = data.get("molecules", [])
    logger.info(f"chembl search '{query}' -> {len(molecules)} hits")
    return {"molecules": molecules[:10]}


@mcp.tool()
async def get_molecule_mechanism(chembl_id: str) -> dict:
    """Fetch known mechanism-of-action records for a ChEMBL molecule ID."""
    url = f"{BASE_URL}/mechanism"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            url, params={"molecule_chembl_id": chembl_id, "format": "json"}
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_activity_data(chembl_id: str, limit: int = 20) -> dict:
    """Fetch bioactivity (IC50/EC50/Ki etc.) records for a ChEMBL molecule ID."""
    url = f"{BASE_URL}/activity"
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            url,
            params={
                "molecule_chembl_id": chembl_id,
                "format": "json",
                "limit": limit,
            },
        )
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
