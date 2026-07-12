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
            ready, _, _ = select.select(
                [self._process.stdout], [], [], self.timeout
            )
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
        self.client = httpx.Client(timeout=timeout)
        self._next_id = 0

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        try:
            response = self.client.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id,
                    "method": method,
                    "params": params,
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MCPError("MCP_UNAVAILABLE", "NotebookLM MCP 服务不可用") from exc
        if "error" in payload:
            raise MCPError("MCP_TOOL_ERROR", "NotebookLM 工具执行失败")
        return payload.get("result", {})


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
