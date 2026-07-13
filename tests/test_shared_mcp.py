from notebooklm_mcp import shared


def test_shared_mcp_client_is_reused(monkeypatch):
    created = []

    class Client:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(shared, "MCPClient", Client)
    shared.get_shared_mcp_client.cache_clear()
    try:
        first = shared.get_shared_mcp_client()
        second = shared.get_shared_mcp_client()
        assert first is second
        assert len(created) == 1
    finally:
        shared.get_shared_mcp_client.cache_clear()
