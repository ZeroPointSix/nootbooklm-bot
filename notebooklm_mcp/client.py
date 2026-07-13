from __future__ import annotations

import json
import select
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

import httpx


class MCPError(RuntimeError):
    """A safe, normalized MCP failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "strict": False,
        }


class _StdioTransport:
    def __init__(self, command: tuple[str, ...], timeout: float):
        self.command = command
        self.timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    def close(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def _start(self) -> None:
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError) as exc:
            raise MCPError("MCP_UNAVAILABLE", "NotebookLM MCP 服务无法启动") from exc
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "notebooklm-slack-agent", "version": "0.1.0"},
            },
            ensure_started=False,
        )
        self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _write(self, payload: dict[str, Any]) -> None:
        assert self._process and self._process.stdin
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _request(
        self, method: str, params: dict[str, Any], *, ensure_started: bool = True
    ) -> dict[str, Any]:
        if ensure_started and (not self._process or self._process.poll() is not None):
            self._start()
        assert self._process and self._process.stdout
        self._next_id += 1
        request_id = self._next_id
        self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        while True:
            ready, _, _ = select.select([self._process.stdout], [], [], self.timeout)
            if not ready:
                self.close()
                raise MCPError("MCP_TIMEOUT", "NotebookLM MCP 请求超时")
            line = self._process.stdout.readline()
            if not line:
                raise MCPError("MCP_UNAVAILABLE", "NotebookLM MCP 服务意外退出")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise MCPError("MCP_TOOL_ERROR", "NotebookLM 工具执行失败")
            return response.get("result", {})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._request(method, params)


class _HTTPTransport:
    def __init__(self, url: str, timeout: float):
        self.url = url
        # MCP endpoints are deployment-controlled service endpoints. Inheriting
        # HTTP(S)_PROXY here can unexpectedly route credentials and tool data
        # through a workstation/container proxy, and can also break loopback
        # connections when optional SOCKS support is absent.
        self.client = httpx.Client(timeout=timeout, trust_env=False)
        self._next_id = 0
        self._session_id: str | None = None
        self._initialized = False

    def close(self) -> None:
        self.client.close()

    def _post(
        self, method: str, params: dict[str, Any], *, notification: bool = False
    ) -> dict[str, Any]:
        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if not notification:
            payload["id"] = self._next_id
        headers = {"Accept": "application/json, text/event-stream"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            response = self.client.post(
                self.url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            self._session_id = response.headers.get("Mcp-Session-Id", self._session_id)
            if notification or not response.content:
                return {}
            if "text/event-stream" in response.headers.get("content-type", ""):
                data = [
                    line.removeprefix("data:").strip()
                    for line in response.text.splitlines()
                    if line.startswith("data:")
                ]
                response_payload = json.loads(data[-1])
            else:
                response_payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MCPError("MCP_UNAVAILABLE", "NotebookLM MCP 服务不可用") from exc
        if "error" in response_payload:
            raise MCPError("MCP_TOOL_ERROR", "NotebookLM 工具执行失败")
        return response_payload.get("result", {})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            self._post(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "notebooklm-slack-agent",
                        "version": "0.1.0",
                    },
                },
            )
            self._post("notifications/initialized", {}, notification=True)
            self._initialized = True
        return self._post(method, params)


class MCPClient:
    def __init__(
        self,
        *,
        transport: str,
        command: tuple[str, ...] = (),
        url: str | None = None,
        timeout: float = 30,
    ):
        self._configuration = {
            "transport": transport,
            "command": command,
            "url": url,
            "timeout": timeout,
        }
        self._transport = self._build_transport()

    def _build_transport(self):
        transport = self._configuration["transport"]
        command = self._configuration["command"]
        url = self._configuration["url"]
        timeout = self._configuration["timeout"]
        if transport == "stdio":
            return _StdioTransport(command, timeout)
        if transport == "http" and url:
            return _HTTPTransport(url, timeout)
        raise ValueError("无效的 MCP 传输配置")

    def close(self) -> None:
        self._transport.close()

    def reconnect(self) -> None:
        self._transport.close()
        self._transport = self._build_transport()

    def list_tools(self) -> list[ToolDefinition]:
        result = self._transport.request("tools/list", {})
        return [
            ToolDefinition(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("inputSchema", {"type": "object"}),
            )
            for item in result.get("tools", [])
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._transport.request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if result.get("isError"):
            raise MCPError("MCP_TOOL_ERROR", "NotebookLM 工具执行失败")
        return result
