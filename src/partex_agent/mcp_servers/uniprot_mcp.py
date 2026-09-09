"""
UniProt MCP server - protein sequence, function, and target annotation,
used mainly by the profiling and hypothesis generation agents.

Run standalone:
    python -m partex_agent.mcp_servers.uniprot_mcp
"""
import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = "https://rest.uniprot.org/uniprotkb"
mcp = FastMCP("uniprot")


@mcp.tool()
async def search_protein(query: str, size: int = 5) -> dict:
    """Search UniProtKB for a protein/target by gene name, protein name, or keyword."""
    url = f"{BASE_URL}/search"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            url,
            params={
                "query": query,
                "size": size,
                "fields": "accession,id,gene_names,protein_name,organism_name,length",
                "format": "json",
            },
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_protein_entry(accession: str) -> dict:
    """Fetch the full UniProt entry (function, domains, sequence) for an accession ID."""
    url = f"{BASE_URL}/{accession}.json"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
