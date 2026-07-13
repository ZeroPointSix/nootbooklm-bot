from __future__ import annotations

from functools import lru_cache

from config import Settings
from notebooklm_mcp.client import MCPClient


@lru_cache(maxsize=1)
def get_shared_mcp_client() -> MCPClient:
    settings = Settings.from_env()
    return MCPClient(
        transport=settings.mcp_transport,
        command=settings.mcp_command,
        url=settings.mcp_url,
        timeout=settings.mcp_timeout_seconds,
    )


def reset_shared_mcp_client() -> None:
    client = get_shared_mcp_client()
    client.close()
    get_shared_mcp_client.cache_clear()
