"""
PubChem MCP server.

Exposes PubChem's PUG REST API as standardized MCP tools so any agent
(or any other MCP client) can look up a compound the same way it would
call any other tool - no bespoke HTTP code inside agent logic.

Run standalone:
    python -m partex_agent.mcp_servers.pubchem_mcp
"""
import httpx
from mcp.server.fastmcp import FastMCP

from partex_agent.logging import logger

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
mcp = FastMCP("pubchem")


@mcp.tool()
async def search_compound(name_or_smiles: str, by: str = "name") -> dict:
    """
    Look up a compound in PubChem by name, SMILES, or InChIKey.

    Args:
        name_or_smiles: the identifier to search (e.g. "aspirin" or a SMILES string)
        by: one of "name", "smiles", "inchikey"
    """
    url = f"{BASE_URL}/compound/{by}/{name_or_smiles}/cids/JSON"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            
            # Catch 404 (e.g., for biologics/monoclonal antibodies like pembrolizumab)
            if resp.status_code == 404:
                logger.warning(
                    f"pubchem search '{name_or_smiles}' -> 404 Not Found "
                    "(may be a biologic/antibody or unlisted compound)"
                )
                return {
                    "cids": [],
                    "status": "not_found",
                    "message": f"'{name_or_smiles}' not found in PubChem small molecule database."
                }
            
            resp.raise_for_status()
            data = resp.json()

        cids = data.get("IdentifierList", {}).get("CID", [])
        logger.info(f"pubchem search '{name_or_smiles}' -> {len(cids)} CIDs")
        return {"cids": cids, "status": "success"}

    except Exception as e:
        logger.error(f"Error in search_compound for '{name_or_smiles}': {e}")
        return {
            "cids": [],
            "status": "error",
            "message": f"PubChem search failed: {str(e)}"
        }


@mcp.tool()
async def get_compound_properties(cid: int) -> dict:
    """
    Fetch key physicochemical properties for a PubChem CID:
    molecular weight, XLogP, TPSA, H-bond donors/acceptors, canonical SMILES.
    """
    props = "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,CanonicalSMILES,IUPACName"
    url = f"{BASE_URL}/compound/cid/{cid}/property/{props}/JSON"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return {"status": "not_found", "message": f"No properties found for CID {cid}."}
            resp.raise_for_status()
            data = resp.json()
        return data.get("PropertyTable", {}).get("Properties", [{}])[0]
    except Exception as e:
        logger.error(f"Error in get_compound_properties for CID {cid}: {e}")
        return {"status": "error", "message": f"Failed to fetch properties: {str(e)}"}


@mcp.tool()
async def get_compound_bioassay_summary(cid: int) -> dict:
    """Fetch a summary of bioassay results a compound has been tested in."""
    url = f"{BASE_URL}/compound/cid/{cid}/assaysummary/JSON"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return {"status": "not_found", "message": f"No bioassay summary found for CID {cid}."}
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Error in get_compound_bioassay_summary for CID {cid}: {e}")
        return {"status": "error", "message": f"Failed to fetch bioassay summary: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")