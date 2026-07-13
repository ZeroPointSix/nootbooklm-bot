import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from notebooklm_mcp.client import MCPClient, MCPError, ToolDefinition


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        if request["method"] == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        elif request["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "notebook_list",
                        "description": "list",
                        "inputSchema": {
                            "type": "object",
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        elif request["method"] == "tools/call":
            result = {"content": [{"type": "text", "text": "ok"}]}
        else:
            self.send_response(202)
            self.end_headers()
            return
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": request["id"], "result": result}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


@pytest.fixture
def mcp_server():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/mcp"
    server.shutdown()


def test_http_dynamic_tool_discovery_and_call(mcp_server):
    client = MCPClient(transport="http", url=mcp_server, timeout=2)
    tools = client.list_tools()
    assert tools == [
        ToolDefinition(
            "notebook_list", "list", {"type": "object", "additionalProperties": False}
        )
    ]
    assert client.call_tool("notebook_list", {})["content"][0]["text"] == "ok"
    client.close()


def test_http_failure_is_safely_normalized():
    client = MCPClient(transport="http", url="http://127.0.0.1:1", timeout=0.1)
    with pytest.raises(MCPError, match="不可用") as caught:
        client.list_tools()
    assert "127.0.0.1" not in str(caught.value)


def test_http_transport_does_not_inherit_environment_proxy(mcp_server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "socks5h://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "socks5h://127.0.0.1:1")
    client = MCPClient(transport="http", url=mcp_server, timeout=2)
    assert client.list_tools()[0].name == "notebook_list"
    client.close()
