"""
Unified MCP client.

Agents never import pubchem_mcp/chembl_mcp/etc. directly. Instead they call
`MCPToolRegistry.call(server, tool, **kwargs)`, which spawns (or reuses) a
stdio connection to the right MCP server and invokes the tool. This is the
same decoupling the reference repo got from routing all component I/O
through a ConfigurationManager - here it's routed through one MCP gateway.
"""

import json
import threading
import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from partex_agent.logging import logger

# module_path : how to launch each MCP server as a subprocess
SERVER_REGISTRY = {
    "pubchem": "partex_agent.mcp_servers.pubchem_mcp",
    "chembl": "partex_agent.mcp_servers.chembl_mcp",
    "uniprot": "partex_agent.mcp_servers.uniprot_mcp",
    "pdb": "partex_agent.mcp_servers.pdb_mcp",
    "tavily": "partex_agent.mcp_servers.tavily_mcp",
}


class MCPToolRegistry:
    """Lazily starts one MCP server subprocess per source and caches the session."""

    def __init__(self):
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}

    async def _get_session(self, server: str) -> ClientSession:
        if server in self._sessions:
            return self._sessions[server]
        if server not in SERVER_REGISTRY:
            raise ValueError(f"Unknown MCP server '{server}'. Known: {list(SERVER_REGISTRY)}")

        params = StdioServerParameters(
            command="python", args=["-m", SERVER_REGISTRY[server]]
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server] = session
        logger.info(f"MCP session started: {server}")
        return session

    async def call(self, server: str, tool: str, **kwargs) -> Any:
        session = await self._get_session(server)
        result = await session.call_tool(tool, arguments=kwargs)

        if result.isError:
            error_text = "".join(
                block.text for block in result.content if getattr(block, "type", None) == "text"
            )
            raise RuntimeError(f"MCP tool '{server}.{tool}' returned an error: {error_text}")

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured

        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    async def list_tools(self, server: str) -> list[str]:
        session = await self._get_session(server)
        tools = await session.list_tools()
        return [t.name for t in tools.tools]

    async def aclose(self):
        await self._stack.aclose()


# Singleton used throughout the agent layer
registry = MCPToolRegistry()

_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_loop_thread.start()


def call_tool_sync(server: str, tool: str, **kwargs) -> Any:
    """Sync convenience wrapper for LangGraph nodes that aren't async.
    Schedules the coroutine onto the single persistent background loop instead
    of spinning up (and tearing down) a new loop per call."""
    future = asyncio.run_coroutine_threadsafe(registry.call(server, tool, **kwargs), _loop)
    return future.result()