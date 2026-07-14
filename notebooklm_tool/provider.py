from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse


DEFAULT_DRIVE_MIME_TYPE = "application/vnd.google-apps.document"
SOURCE_UPLOAD_BYTES_B64_LIMIT = 10_000
TOOL_CONTRACT = "notebooklm-py-0.8.0a3-35-tools"
LOGGER = logging.getLogger(__name__)
READY_SOURCE_STATUSES = {"ready"}
PROCESSING_SOURCE_STATUSES = {"processing", "preparing", "1", "5"}
FAILED_SOURCE_STATUSES = {"error", "3"}
SOURCE_STATUS_CODE_LABELS = {
    "1": "processing",
    "2": "ready",
    "3": "error",
    "5": "preparing",
}
SENSITIVE_EXCEPTION_MARKERS = {
    "authorization",
    "cookie",
    "password",
    "storage_state",
    "token",
}


TOOL_NAMES = (
    "notebook_list",
    "notebook_create",
    "notebook_describe",
    "notebook_rename",
    "notebook_delete",
    "source_list",
    "source_read",
    "source_rename",
    "source_delete",
    "source_wait",
    "source_add",
    "source_add_and_wait",
    "source_upload_bytes",
    "source_add_drive_file",
    "chat_ask",
    "chat_configure",
    "suggest_prompts",
    "note_save",
    "studio_list",
    "studio_generate",
    "studio_status",
    "studio_get_prompt",
    "studio_download",
    "studio_rename",
    "studio_retry",
    "studio_delete",
    "research_start",
    "research_status",
    "research_import",
    "research_cancel",
    "share_status",
    "share_set_access",
    "share_set_user",
    "share_remove_user",
    "server_info",
)

CONFIRMATION_TOOLS = {
    "notebook_delete",
    "source_delete",
    "studio_delete",
    "share_set_access",
    "share_set_user",
    "share_remove_user",
}


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


class NotebookToolError(RuntimeError):
    """A safe, normalized NotebookLM tool failure."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)


class NotebookBackend(Protocol):
    def probe(self) -> dict[str, Any]: ...

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any: ...

    def reconnect(self) -> None: ...


class NotebookToolProvider(Protocol):
    def list_tools(self) -> list[ToolDefinition]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def reconnect(self) -> None: ...


def _object_schema(
    properties: dict[str, Any] | None = None,
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": list(required),
        "additionalProperties": False,
    }


def _string(description: str, *, enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        schema["enum"] = enum
    return schema


def _integer(description: str, *, minimum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def _number(description: str, *, minimum: float | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _array(description: str, item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "description": description, "items": item_schema}


def _json_value(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "anyOf": [
            {"type": "object"},
            {"type": "array"},
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
        ],
    }


def _notebook_ref() -> dict[str, Any]:
    return _string("NotebookLM notebook title, ID, or URL")


def _source_ref() -> dict[str, Any]:
    return _string("NotebookLM source title, ID, or URL")


def _artifact_ref() -> dict[str, Any]:
    return _string("NotebookLM studio artifact title, ID, or URL")


def _check(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


LOCAL_TOOLS = [
    ToolDefinition(
        "notebook_list",
        "List notebooks visible to the authenticated NotebookLM account.",
        _object_schema(
            {
                "limit": _integer("Maximum notebooks to return", minimum=1),
                "offset": _integer("Result offset", minimum=0),
            }
        ),
    ),
    ToolDefinition(
        "notebook_create",
        "Create a NotebookLM notebook.",
        _object_schema({"title": _string("Notebook title")}, required=("title",)),
    ),
    ToolDefinition(
        "notebook_describe",
        "Describe a NotebookLM notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "include_metadata": _boolean("Include metadata when available"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "notebook_rename",
        "Rename a NotebookLM notebook.",
        _object_schema(
            {"notebook": _notebook_ref(), "new_title": _string("New notebook title")},
            required=("notebook", "new_title"),
        ),
    ),
    ToolDefinition(
        "notebook_delete",
        "Delete a NotebookLM notebook after explicit confirmation.",
        _object_schema(
            {"notebook": _notebook_ref(), "confirm": _boolean("Required to delete")},
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "source_list",
        "List sources in a NotebookLM notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "status": _string("Filter by source status"),
                "label": _string("Filter by source label"),
                "detail": _string("Detail level", enum=["summary", "full"]),
                "limit": _integer("Maximum sources to return", minimum=1),
                "offset": _integer("Result offset", minimum=0),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "source_read",
        "Read a source from a NotebookLM notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "source": _source_ref(),
                "detail": _string("Detail level", enum=["summary", "full"]),
                "output_format": _string(
                    "Output format", enum=["text", "markdown", "json"]
                ),
                "max_chars": _integer("Maximum characters to return", minimum=1),
                "offset": _integer("Character offset", minimum=0),
            },
            required=("notebook", "source"),
        ),
    ),
    ToolDefinition(
        "source_rename",
        "Rename a source in a NotebookLM notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "source": _source_ref(),
                "new_title": _string("New source title"),
            },
            required=("notebook", "source", "new_title"),
        ),
    ),
    ToolDefinition(
        "source_delete",
        "Delete a source after explicit confirmation.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "source": _source_ref(),
                "confirm": _boolean("Required to delete"),
            },
            required=("notebook", "source"),
        ),
    ),
    ToolDefinition(
        "source_wait",
        "Wait for one or all NotebookLM sources to finish processing.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "source": _source_ref(),
                "timeout": _number("Maximum seconds to wait", minimum=1),
                "interval": _number("Polling interval in seconds", minimum=0.1),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "source_add",
        "Add URLs, text, local files, Drive files, or copied content to a NotebookLM notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "source_type": _string(
                    "Source type",
                    enum=["url", "text", "file", "drive", "youtube", "audio"],
                ),
                "url": _string("URL to add"),
                "urls": _array("URLs to add", _string("URL")),
                "text": _string("Text content to add"),
                "texts": _array("Text blocks to add", _string("Text content")),
                "path": _string("Local file path"),
                "document_id": _string("Google Drive file ID"),
                "mime_type": _string("Source MIME type"),
                "title": _string("Optional source title"),
                "allow_internal": _boolean("Allow internal or non-public URLs"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "source_add_and_wait",
        "Add a source and wait for NotebookLM processing to finish.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "source_type": _string(
                    "Source type",
                    enum=["url", "text", "file", "drive", "youtube", "audio"],
                ),
                "url": _string("URL to add"),
                "text": _string("Text content to add"),
                "path": _string("Local file path"),
                "document_id": _string("Google Drive file ID"),
                "mime_type": _string("Source MIME type"),
                "title": _string("Optional source title"),
                "allow_internal": _boolean("Allow internal or non-public URLs"),
                "timeout": _number("Maximum seconds to wait", minimum=1),
                "interval": _number("Polling interval in seconds", minimum=0.1),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "source_upload_bytes",
        "Add a small file to a NotebookLM notebook from base64-encoded bytes.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "bytes_base64": _string(
                    "Standard base64 file payload, max 10000 chars"
                ),
                "filename": _string(
                    "Original file name used for extension and default title"
                ),
                "mime_type": _string("Optional MIME type"),
                "title": _string("Optional source title"),
            },
            required=("notebook", "bytes_base64"),
        ),
    ),
    ToolDefinition(
        "source_add_drive_file",
        "Add a Google Drive file to a NotebookLM notebook as a source.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "document_id": _string("Google Drive file ID"),
                "file_id": _string("Google Drive file ID alias"),
                "title": _string("Drive file title"),
                "mime_type": _string("Drive MIME type"),
                "wait": _boolean("Wait for source processing"),
                "timeout": _number("Maximum seconds to wait", minimum=1),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "chat_ask",
        "Ask a question in a NotebookLM notebook chat.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "question": _string("Question for NotebookLM"),
                "conversation_id": _string("Existing conversation ID"),
                "references": _array("Reference IDs", _string("Reference ID")),
                "source_ids": _array(
                    "Source IDs to constrain the answer", _string("Source ID")
                ),
                "history": _array("Prior chat turns", _json_value("Chat turn")),
                "suggest_followups": _boolean(
                    "Ask NotebookLM for follow-up suggestions"
                ),
            },
            required=("notebook", "question"),
        ),
    ),
    ToolDefinition(
        "chat_configure",
        "Configure NotebookLM chat behavior for a notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "chat_mode": _string("Chat mode"),
                "goal": _string("User goal or instruction"),
                "response_length": _string("Desired response length"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "suggest_prompts",
        "Suggest NotebookLM prompts for a notebook or selected sources.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "surface": _string("Suggestion surface"),
                "source_ids": _array("Source IDs", _string("Source ID")),
                "query": _string("Prompt topic or query"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "note_save",
        "Save a note in a NotebookLM notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "note": _string("Existing note ID"),
                "title": _string("Note title"),
                "content": _string("Note content"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "studio_list",
        "List NotebookLM studio artifacts.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "item": _artifact_ref(),
                "kind": _string("Artifact kind"),
                "detail": _string("Detail level", enum=["summary", "full"]),
                "limit": _integer("Maximum artifacts to return", minimum=1),
                "offset": _integer("Result offset", minimum=0),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "studio_generate",
        "Generate a NotebookLM studio artifact.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "artifact_type": _string("Artifact type"),
                "prompt": _string("Generation prompt"),
                "topic": _string("Artifact topic"),
                "instruction": _string("Generation instruction"),
                "source_ids": _array("Source IDs", _string("Source ID")),
                "note_ids": _array("Note IDs", _string("Note ID")),
            },
            required=("notebook", "artifact_type"),
        ),
    ),
    ToolDefinition(
        "studio_status",
        "Read the status of a NotebookLM studio generation task.",
        _object_schema(
            {"notebook": _notebook_ref(), "task_id": _string("Studio task ID")},
            required=("notebook", "task_id"),
        ),
    ),
    ToolDefinition(
        "studio_get_prompt",
        "Read the prompt behind a NotebookLM studio artifact.",
        _object_schema(
            {"notebook": _notebook_ref(), "artifact": _artifact_ref()},
            required=("notebook", "artifact"),
        ),
    ),
    ToolDefinition(
        "studio_download",
        "Download or export a NotebookLM studio artifact.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "artifact": _artifact_ref(),
                "artifact_type": _string("Artifact type"),
                "artifact_id": _string("Artifact ID"),
                "path": _string("Destination path"),
                "output_format": _string("Output format"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "studio_rename",
        "Rename a NotebookLM studio artifact.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "item": _artifact_ref(),
                "new_title": _string("New artifact title"),
            },
            required=("notebook", "item", "new_title"),
        ),
    ),
    ToolDefinition(
        "studio_retry",
        "Retry a NotebookLM studio artifact generation.",
        _object_schema(
            {"notebook": _notebook_ref(), "artifact": _artifact_ref()},
            required=("notebook", "artifact"),
        ),
    ),
    ToolDefinition(
        "studio_delete",
        "Delete a NotebookLM studio artifact after explicit confirmation.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "item": _artifact_ref(),
                "confirm": _boolean("Required to delete"),
            },
            required=("notebook", "item"),
        ),
    ),
    ToolDefinition(
        "research_start",
        "Start NotebookLM research for a notebook.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "query": _string("Research query"),
                "source": _string("Research source scope"),
                "mode": _string("Research mode"),
            },
            required=("notebook", "query", "source", "mode"),
        ),
    ),
    ToolDefinition(
        "research_status",
        "Read NotebookLM research status.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "poll_task_id": _string("Research task ID"),
                "include_report": _boolean("Include report content"),
                "report_max_chars": _integer("Maximum report characters", minimum=1),
                "source_limit": _integer("Maximum sources", minimum=1),
                "source_offset": _integer("Source offset", minimum=0),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "research_import",
        "Import completed NotebookLM research results into a notebook.",
        _object_schema(
            {"notebook": _notebook_ref(), "poll_task_id": _string("Research task ID")},
            required=("notebook", "poll_task_id"),
        ),
    ),
    ToolDefinition(
        "research_cancel",
        "Cancel an active NotebookLM research task.",
        _object_schema(
            {"notebook": _notebook_ref(), "poll_task_id": _string("Research task ID")},
            required=("notebook", "poll_task_id"),
        ),
    ),
    ToolDefinition(
        "share_status",
        "Read NotebookLM notebook sharing status.",
        _object_schema({"notebook": _notebook_ref()}, required=("notebook",)),
    ),
    ToolDefinition(
        "share_set_access",
        "Change public access for a NotebookLM notebook after confirmation.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "public": _boolean("Whether public access is enabled"),
                "view_level": _string("Public view level"),
                "confirm": _boolean("Required to change sharing"),
            },
            required=("notebook",),
        ),
    ),
    ToolDefinition(
        "share_set_user",
        "Grant or change a user's notebook permission after confirmation.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "email": _string("User email"),
                "permission": _string("Permission level"),
                "notify": _boolean("Send notification"),
                "message": _string("Notification message"),
                "confirm": _boolean("Required to change sharing"),
            },
            required=("notebook", "email"),
        ),
    ),
    ToolDefinition(
        "share_remove_user",
        "Remove a user's notebook access after confirmation.",
        _object_schema(
            {
                "notebook": _notebook_ref(),
                "email": _string("User email"),
                "confirm": _boolean("Required to remove sharing"),
            },
            required=("notebook", "email"),
        ),
    ),
    ToolDefinition(
        "server_info",
        "Report the built-in NotebookLM provider version, tool contract, and auth health.",
        _object_schema(
            {"include_account": _boolean("Include live account details when available")}
        ),
    ),
]


class SDKNotebookLMBackend:
    """Library-mode adapter around notebooklm-py core client APIs."""

    _STUDIO_GENERATORS = {
        "audio": "generate_audio",
        "video": "generate_video",
        "cinematic-video": "generate_cinematic_video",
        "slide-deck": "generate_slide_deck",
        "quiz": "generate_quiz",
        "flashcards": "generate_flashcards",
        "infographic": "generate_infographic",
        "data-table": "generate_data_table",
        "mind-map": "generate_mind_map",
        "report": "generate_report",
        "study-guide": "generate_study_guide",
    }
    _STUDIO_DOWNLOADS = {
        "audio": "download_audio",
        "video": "download_video",
        "slide-deck": "download_slide_deck",
        "quiz": "download_quiz",
        "flashcards": "download_flashcards",
        "infographic": "download_infographic",
        "data-table": "download_data_table",
        "mind-map": "download_mind_map",
        "report": "download_report",
    }
    _PROMPT_SURFACES = {
        "ask": 4,
        "audio-deep-dive": 1,
        "audio-brief": 2,
        "audio-critique": 5,
        "audio-debate": 6,
        "video-explainer": 3,
        "video-short": 10,
        "quiz": 8,
        "flashcards": 9,
    }

    def __init__(self, profile_path: str | Path):
        self.profile_path = Path(profile_path)

    def reconnect(self) -> None:
        return None

    def probe(self) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._probe_async())
        raise NotebookToolError(
            "NOTEBOOKLM_RUNTIME_UNSUPPORTED",
            "NotebookLM health 当前在同步运行时执行；请在非事件循环线程中调用。",
        )

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._invoke_async(tool_name, arguments))
        raise NotebookToolError(
            "NOTEBOOKLM_RUNTIME_UNSUPPORTED",
            "NotebookLM 工具当前在同步运行时执行；请在非事件循环线程中调用。",
        )

    async def _invoke_async(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return await self._with_client(
            lambda client: self._dispatch(client, tool_name, arguments)
        )

    async def _probe_async(self) -> dict[str, Any]:
        return await self._with_client(self._probe_client)

    async def _with_client(self, callback: Any) -> Any:
        try:
            from notebooklm import NotebookLMClient
        except ImportError as exc:
            raise NotebookToolError(
                "NOTEBOOKLM_SDK_MISSING",
                "缺少 notebooklm-py==0.8.0a3，无法执行 NotebookLM 工具。",
            ) from exc

        factory = getattr(NotebookLMClient, "from_storage", None)
        if factory is None:
            raise NotebookToolError(
                "NOTEBOOKLM_SDK_INCOMPATIBLE",
                "当前 notebooklm-py 版本缺少 NotebookLMClient.from_storage。",
            )

        context = await _maybe_await(factory(str(self.profile_path)))
        if hasattr(context, "__aenter__"):
            async with context as client:
                return await _maybe_await(callback(client))
        if hasattr(context, "__enter__"):
            with context as client:
                return await _maybe_await(callback(client))
        return await _maybe_await(callback(context))

    async def _probe_client(self, client: Any) -> dict[str, Any]:
        notebooks = list(await client.notebooks.list())
        return {
            "ok": True,
            "probe": "notebooks.list",
            "notebook_count": len(notebooks),
        }

    async def _dispatch(
        self, client: Any, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        args = _sdk_arguments(arguments)
        match tool_name:
            case "notebook_list":
                notebooks = await client.notebooks.list()
                return _page_payload("notebooks", notebooks, args)
            case "notebook_create":
                return await client.notebooks.create(args["title"])
            case "notebook_describe":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                description = await client.notebooks.get_description(notebook_id)
                payload: dict[str, Any] = {
                    "notebook_id": notebook_id,
                    "description": _json_safe(description),
                }
                if args.get("include_metadata"):
                    payload["metadata"] = _json_safe(
                        await client.notebooks.get_metadata(notebook_id)
                    )
                return payload
            case "notebook_rename":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                return await client.notebooks.rename(notebook_id, args["new_title"])
            case "notebook_delete":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                await client.notebooks.delete(notebook_id)
                return {"status": "deleted", "notebook_id": notebook_id}
            case "source_list":
                return await self._source_list(client, args)
            case "source_read":
                return await self._source_read(client, args)
            case "source_rename":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                source_id = await self._resolve_source(
                    client, notebook_id, args["source"]
                )
                return await client.sources.rename(
                    notebook_id, source_id, args["new_title"]
                )
            case "source_delete":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                source_id = await self._resolve_source(
                    client, notebook_id, args["source"]
                )
                await client.sources.delete(notebook_id, source_id)
                return {
                    "status": "deleted",
                    "notebook_id": notebook_id,
                    "source_id": source_id,
                }
            case "source_wait":
                return await self._source_wait(client, args)
            case "source_add":
                return await self._source_add(client, args, wait=False)
            case "source_add_and_wait":
                return await self._source_add(client, args, wait=True)
            case "source_upload_bytes":
                return await self._source_upload_bytes(client, args)
            case "source_add_drive_file":
                return await self._source_add_drive_file(client, args)
            case "chat_ask":
                return await self._chat_ask(client, args)
            case "chat_configure":
                return await self._chat_configure(client, args)
            case "suggest_prompts":
                return await self._suggest_prompts(client, args)
            case "note_save":
                return await self._note_save(client, args)
            case "studio_list":
                return await self._studio_list(client, args)
            case "studio_generate":
                return await self._studio_generate(client, args)
            case "studio_status":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                return await client.artifacts.poll_status(notebook_id, args["task_id"])
            case "studio_get_prompt":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                artifact_id = await self._resolve_artifact(
                    client, notebook_id, args["artifact"]
                )
                return {
                    "notebook_id": notebook_id,
                    "artifact_id": artifact_id,
                    "prompt": await client.artifacts.get_prompt(
                        notebook_id, artifact_id
                    ),
                }
            case "studio_download":
                return await self._studio_download(client, args)
            case "studio_rename":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                artifact_id = await self._resolve_artifact(
                    client, notebook_id, args["item"]
                )
                return await client.artifacts.rename(
                    notebook_id, artifact_id, args["new_title"]
                )
            case "studio_retry":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                artifact_id = await self._resolve_artifact(
                    client, notebook_id, args["artifact"]
                )
                return await client.artifacts.retry_failed(notebook_id, artifact_id)
            case "studio_delete":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                artifact_id = await self._resolve_artifact(
                    client, notebook_id, args["item"]
                )
                await client.artifacts.delete(notebook_id, artifact_id)
                return {
                    "status": "deleted",
                    "notebook_id": notebook_id,
                    "artifact_id": artifact_id,
                }
            case "research_start":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                result = await client.research.start(
                    notebook_id,
                    args["query"],
                    args.get("source") or "web",
                    args.get("mode") or "fast",
                )
                payload = _json_safe(result)
                payload["notebook_id"] = notebook_id
                payload["poll_task_id"] = (
                    payload.get("report_id")
                    if (args.get("mode") or "fast") == "deep"
                    else payload.get("task_id")
                )
                return payload
            case "research_status":
                return await self._research_status(client, args)
            case "research_import":
                return await self._research_import(client, args)
            case "research_cancel":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                poll_task_id = args["poll_task_id"]
                await client.research.cancel(notebook_id, poll_task_id)
                return {
                    "status": "cancel_requested",
                    "notebook_id": notebook_id,
                    "poll_task_id": poll_task_id,
                }
            case "share_status":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                return await client.sharing.get_status(notebook_id)
            case "share_set_access":
                return await self._share_set_access(client, args)
            case "share_set_user":
                return await self._share_set_user(client, args)
            case "share_remove_user":
                notebook_id = await self._resolve_notebook(client, args["notebook"])
                await client.sharing.remove_user(notebook_id, args["email"])
                return {
                    "status": "removed",
                    "notebook_id": notebook_id,
                    "email": args["email"],
                }
            case "server_info":
                return await self._server_info(client, args)
        raise NotebookToolError("UNKNOWN_TOOL", f"未实现工具：{tool_name}")

    async def _source_list(self, client: Any, args: dict[str, Any]) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        sources = list(await client.sources.list(notebook_id))
        status = args.get("status")
        if status:
            sources = [source for source in sources if _status_label(source) == status]
        return {"notebook_id": notebook_id, **_page_payload("sources", sources, args)}

    async def _source_read(self, client: Any, args: dict[str, Any]) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        sources = list(await client.sources.list(notebook_id))
        source_id = await self._resolve_id(sources, args["source"], "source")
        listed_source = _find_item_by_id(sources, source_id)
        if listed_source is not None:
            _ensure_source_ready_for_read(listed_source)
        if args.get("detail") == "summary":
            guide = await client.sources.get_guide(notebook_id, source_id)
            return {
                "notebook_id": notebook_id,
                "source_id": source_id,
                "summary": _json_safe(guide),
            }
        source = await client.sources.get(notebook_id, source_id)
        _ensure_source_ready_for_read(source)
        fulltext = _json_safe(
            await client.sources.get_fulltext(
                notebook_id,
                source_id,
                output_format=args.get("output_format") or "text",
            )
        )
        _slice_text_fields(fulltext, args.get("offset") or 0, args.get("max_chars"))
        return {
            "notebook_id": notebook_id,
            "source_id": source_id,
            "source": _json_safe(source),
            "fulltext": fulltext,
        }

    async def _source_wait(self, client: Any, args: dict[str, Any]) -> Any:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        timeout = args.get("timeout") or 120.0
        interval = args.get("interval") or 1.0
        if args.get("source"):
            source_id = await self._resolve_source(client, notebook_id, args["source"])
            return await client.sources.wait_until_ready(
                notebook_id, source_id, timeout=timeout, initial_interval=interval
            )
        sources = await client.sources.list(notebook_id)
        source_ids = [_item_id(source) for source in sources]
        return await client.sources.wait_for_sources(
            notebook_id, source_ids, timeout=timeout, initial_interval=interval
        )

    async def _source_add(
        self, client: Any, args: dict[str, Any], *, wait: bool
    ) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        wait_timeout = args.get("timeout") or 120.0
        added = []
        urls = _coerce_string_list(args.get("urls"))
        texts = _coerce_string_list(args.get("texts"))
        if args.get("url"):
            urls.append(args["url"])
        if args.get("text"):
            texts.append(args["text"])
        for url in urls:
            added.append(
                await client.sources.add_url(
                    notebook_id, url, wait=wait, wait_timeout=wait_timeout
                )
            )
        for index, text in enumerate(texts, start=1):
            title = args.get("title") or f"Text Source {index}"
            added.append(
                await client.sources.add_text(
                    notebook_id,
                    title,
                    text,
                    wait=wait,
                    wait_timeout=wait_timeout,
                )
            )
        if args.get("path"):
            added.append(
                await client.sources.add_file(
                    notebook_id,
                    args["path"],
                    wait=wait,
                    wait_timeout=wait_timeout,
                    title=args.get("title"),
                )
            )
        if args.get("document_id") or args.get("file_id"):
            added.append(
                await self._add_drive_file(client, notebook_id, args, wait=wait)
            )
        if not added:
            raise NotebookToolError(
                "NOTEBOOKLM_ARGUMENTS",
                "source_add 需要提供 url、urls、text、texts、path 或 document_id。",
            )
        return {"notebook_id": notebook_id, "sources": _json_safe(added)}

    async def _source_upload_bytes(
        self, client: Any, args: dict[str, Any]
    ) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        raw = _decode_upload_bytes(args.get("bytes_base64"))
        filename = _safe_upload_filename(args.get("filename") or args.get("title"))
        suffix = Path(filename).suffix or ".bin"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", prefix="notebooklm-upload-", suffix=suffix, delete=False
            ) as handle:
                handle.write(raw)
                temp_path = Path(handle.name)
            source = await client.sources.add_file(
                notebook_id,
                temp_path,
                mime_type=args.get("mime_type"),
                title=args.get("title") or filename,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return {
            "notebook_id": notebook_id,
            "filename": filename,
            "source": _json_safe(source),
        }

    async def _source_add_drive_file(
        self, client: Any, args: dict[str, Any]
    ) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        source = await self._add_drive_file(
            client,
            notebook_id,
            args,
            wait=bool(args.get("wait")),
        )
        return {"notebook_id": notebook_id, "source": _json_safe(source)}

    async def _add_drive_file(
        self, client: Any, notebook_id: str, args: dict[str, Any], *, wait: bool
    ) -> Any:
        file_id = args.get("document_id") or args.get("file_id")
        if not file_id:
            raise NotebookToolError(
                "NOTEBOOKLM_ARGUMENTS",
                "source_add_drive_file 需要提供 document_id 或 file_id。",
            )
        title = args.get("title") or ""
        return await client.sources.add_drive(
            notebook_id,
            file_id,
            title,
            mime_type=args.get("mime_type") or DEFAULT_DRIVE_MIME_TYPE,
            wait=wait,
            wait_timeout=args.get("timeout") or 120.0,
        )

    async def _server_info(self, client: Any, args: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "server": "nootbooklm-bot-native",
            "version": _notebooklm_version(),
            "backend": "native",
            "tool_count": len(LOCAL_TOOLS),
            "tool_contract": TOOL_CONTRACT,
            "external_protocol_required": False,
            "bridge": False,
        }
        if args.get("include_account"):
            info["account"] = await _account_snapshot(client)
        return info

    async def _chat_ask(self, client: Any, args: dict[str, Any]) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        source_ids = await self._resolve_source_ids(
            client, notebook_id, args.get("source_ids")
        )
        result = await client.chat.ask(
            notebook_id,
            args["question"],
            source_ids=source_ids,
            conversation_id=args.get("conversation_id"),
        )
        payload = _json_safe(result)
        payload["notebook_id"] = notebook_id
        if source_ids is not None:
            payload["source_ids"] = source_ids
        return payload

    async def _chat_configure(
        self, client: Any, args: dict[str, Any]
    ) -> dict[str, Any]:
        from notebooklm.rpc.types import ChatGoal, ChatResponseLength

        notebook_id = await self._resolve_notebook(client, args["notebook"])
        goal_text = args.get("goal") or args.get("chat_mode") or ""
        response_length = _enum_choice(
            ChatResponseLength,
            args.get("response_length"),
            {"short": "SHORTER", "concise": "SHORTER", "long": "LONGER"},
            default="DEFAULT",
        )
        goal = ChatGoal.CUSTOM if goal_text else ChatGoal.DEFAULT
        await client.chat.configure(
            notebook_id,
            goal=goal,
            response_length=response_length,
            custom_prompt=goal_text or None,
        )
        return {"status": "configured", "notebook_id": notebook_id}

    async def _suggest_prompts(
        self, client: Any, args: dict[str, Any]
    ) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        source_ids = await self._resolve_source_ids(
            client, notebook_id, args.get("source_ids")
        )
        mode = self._PROMPT_SURFACES.get(args.get("surface") or "ask", 4)
        suggestions = await client.notebooks.suggest_prompts(
            notebook_id,
            source_ids=source_ids,
            mode=mode,
            query=args.get("query"),
        )
        return {
            "notebook_id": notebook_id,
            "suggested_prompts": _json_safe(suggestions),
        }

    async def _note_save(self, client: Any, args: dict[str, Any]) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        if not args.get("note"):
            note = await client.notes.create(
                notebook_id,
                title=args.get("title") or "New Note",
                content=args.get("content") or "",
            )
            return {
                "status": "created",
                "notebook_id": notebook_id,
                "note": _json_safe(note),
            }
        note_id = await self._resolve_note(client, notebook_id, args["note"])
        current = await client.notes.get(notebook_id, note_id)
        await client.notes.update(
            notebook_id,
            note_id,
            content=args.get("content")
            if args.get("content") is not None
            else getattr(current, "content", ""),
            title=args.get("title")
            if args.get("title") is not None
            else getattr(current, "title", "New Note"),
        )
        return {"status": "updated", "notebook_id": notebook_id, "note_id": note_id}

    async def _studio_list(self, client: Any, args: dict[str, Any]) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        kind = _normalize_kind(args.get("kind"))
        if args.get("item"):
            item = await self._resolve_studio_item(
                client, notebook_id, args["item"], kind
            )
            return {
                "notebook_id": notebook_id,
                "items": [item],
                "total": 1,
                "offset": 0,
                "has_more": False,
            }
        artifacts = [
            {"type": _artifact_kind(artifact), **_json_safe(artifact)}
            for artifact in await client.artifacts.list(notebook_id)
        ]
        notes = [
            {"type": "note", **_json_safe(note)}
            for note in await client.notes.list(notebook_id)
        ]
        items = artifacts + notes
        if kind:
            items = [
                item for item in items if _normalize_kind(item.get("type")) == kind
            ]
        return {"notebook_id": notebook_id, **_page_payload("items", items, args)}

    async def _studio_generate(self, client: Any, args: dict[str, Any]) -> Any:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        kind = _normalize_kind(args["artifact_type"])
        method_name = self._STUDIO_GENERATORS.get(kind)
        if method_name is None:
            raise NotebookToolError(
                "NOTEBOOKLM_ARGUMENTS", f"不支持的 studio 类型：{kind}"
            )
        method = getattr(client.artifacts, method_name)
        source_ids = await self._resolve_source_ids(
            client, notebook_id, args.get("source_ids")
        )
        instructions = (
            args.get("instructions")
            or args.get("instruction")
            or args.get("prompt")
            or args.get("topic")
        )
        kwargs = {
            "notebook_id": notebook_id,
            "source_ids": source_ids,
            "instructions": instructions,
            "custom_prompt": instructions,
            "extra_instructions": instructions,
        }
        return await _call_with_supported_args(method, kwargs)

    async def _studio_download(self, client: Any, args: dict[str, Any]) -> Any:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        output_path = args.get("path")
        if not output_path:
            raise NotebookToolError(
                "NOTEBOOKLM_ARGUMENTS", "studio_download 需要 path。"
            )
        kind = _normalize_kind(args.get("artifact_type") or "")
        artifact_ref = args.get("artifact") or args.get("artifact_id")
        artifact_id = None
        if artifact_ref:
            artifact_id = await self._resolve_artifact(
                client, notebook_id, artifact_ref
            )
        if not kind and artifact_id:
            artifact = await client.artifacts.get(notebook_id, artifact_id)
            kind = _artifact_kind(artifact)
        method_name = self._STUDIO_DOWNLOADS.get(kind)
        if method_name is None:
            raise NotebookToolError(
                "NOTEBOOKLM_ARGUMENTS", f"不支持下载的 studio 类型：{kind}"
            )
        method = getattr(client.artifacts, method_name)
        return await _call_with_supported_args(
            method,
            {
                "notebook_id": notebook_id,
                "output_path": output_path,
                "artifact_id": artifact_id,
                "output_format": args.get("output_format"),
            },
        )

    async def _research_status(
        self, client: Any, args: dict[str, Any]
    ) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        task = _json_safe(
            await client.research.poll(notebook_id, args.get("poll_task_id"))
        )
        if not args.get("include_report"):
            task.pop("report", None)
        task["notebook_id"] = notebook_id
        task["poll_task_id"] = task.get("task_id") or args.get("poll_task_id")
        return task

    async def _research_import(
        self, client: Any, args: dict[str, Any]
    ) -> dict[str, Any]:
        notebook_id = await self._resolve_notebook(client, args["notebook"])
        task_id = args["poll_task_id"]
        task = await client.research.poll(notebook_id, task_id)
        sources = getattr(task, "sources", None) or []
        imported = await client.research.import_sources(notebook_id, task_id, sources)
        return {
            "notebook_id": notebook_id,
            "poll_task_id": task_id,
            "imported": _json_safe(imported),
        }

    async def _share_set_access(self, client: Any, args: dict[str, Any]) -> Any:
        from notebooklm.rpc.types import ShareViewLevel

        notebook_id = await self._resolve_notebook(client, args["notebook"])
        status = None
        if args.get("view_level"):
            level = _enum_choice(
                ShareViewLevel,
                args["view_level"],
                {
                    "full": "FULL_NOTEBOOK",
                    "chat": "CHAT_ONLY",
                    "chat_only": "CHAT_ONLY",
                },
                default="FULL_NOTEBOOK",
            )
            status = await client.sharing.set_view_level(notebook_id, level)
        if args.get("public") is not None:
            status = await client.sharing.set_public(notebook_id, bool(args["public"]))
        if status is None:
            raise NotebookToolError(
                "NOTEBOOKLM_ARGUMENTS", "share_set_access 需要 public 或 view_level。"
            )
        return status

    async def _share_set_user(self, client: Any, args: dict[str, Any]) -> Any:
        from notebooklm.rpc.types import SharePermission

        notebook_id = await self._resolve_notebook(client, args["notebook"])
        permission = _enum_choice(
            SharePermission,
            args.get("permission") or "viewer",
            {"editor": "EDITOR", "viewer": "VIEWER"},
            default="VIEWER",
        )
        return await client.sharing.add_user(
            notebook_id,
            args["email"],
            permission=permission,
            notify=bool(args.get("notify")),
            welcome_message=args.get("message") or "",
        )

    async def _resolve_notebook(self, client: Any, ref: str) -> str:
        return await self._resolve_id(await client.notebooks.list(), ref, "notebook")

    async def _resolve_source(self, client: Any, notebook_id: str, ref: str) -> str:
        return await self._resolve_id(
            await client.sources.list(notebook_id), ref, "source"
        )

    async def _resolve_note(self, client: Any, notebook_id: str, ref: str) -> str:
        return await self._resolve_id(await client.notes.list(notebook_id), ref, "note")

    async def _resolve_artifact(self, client: Any, notebook_id: str, ref: str) -> str:
        return await self._resolve_id(
            await client.artifacts.list(notebook_id), ref, "artifact"
        )

    async def _resolve_studio_item(
        self, client: Any, notebook_id: str, ref: str, kind: str | None
    ) -> dict[str, Any]:
        items = [
            {"type": _artifact_kind(artifact), **_json_safe(artifact)}
            for artifact in await client.artifacts.list(notebook_id)
        ] + [
            {"type": "note", **_json_safe(note)}
            for note in await client.notes.list(notebook_id)
        ]
        if kind:
            items = [
                item for item in items if _normalize_kind(item.get("type")) == kind
            ]
        item_id = await self._resolve_id(items, ref, "studio item")
        for item in items:
            if item.get("id") == item_id:
                return item
        raise NotebookToolError("NOTEBOOKLM_NOT_FOUND", f"未找到 Studio 项：{ref}")

    async def _resolve_source_ids(
        self, client: Any, notebook_id: str, refs: Any
    ) -> list[str] | None:
        values = _coerce_string_list(refs)
        if not values:
            return None
        return [
            await self._resolve_source(client, notebook_id, value) for value in values
        ]

    async def _resolve_id(self, items: list[Any], ref: str, label: str) -> str:
        candidate = _extract_ref_id(ref)
        matches = []
        for item in items:
            item_id = _item_id(item)
            title = _item_title(item)
            if candidate in {item_id, title}:
                return item_id
            if len(candidate) >= 6 and item_id.startswith(candidate):
                matches.append(item_id)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise NotebookToolError(
                "NOTEBOOKLM_AMBIGUOUS_REF", f"{label} 引用不唯一：{ref}"
            )
        if _looks_like_id(candidate):
            return candidate
        raise NotebookToolError("NOTEBOOKLM_NOT_FOUND", f"未找到 {label}：{ref}")


class LocalNotebookToolProvider:
    """Built-in NotebookLM tool provider owned by this repository."""

    def __init__(self, profile_path: str, backend: NotebookBackend | None = None):
        self.profile_path = Path(profile_path)
        self.backend = backend or SDKNotebookLMBackend(self.profile_path)
        self._handlers = {
            "notebook_list": self._handle_notebook_list,
            "notebook_create": self._handle_notebook_create,
            "notebook_describe": self._handle_notebook_describe,
            "notebook_rename": self._handle_notebook_rename,
            "notebook_delete": self._handle_notebook_delete,
            "source_list": self._handle_source_list,
            "source_read": self._handle_source_read,
            "source_rename": self._handle_source_rename,
            "source_delete": self._handle_source_delete,
            "source_wait": self._handle_source_wait,
            "source_add": self._handle_source_add,
            "source_add_and_wait": self._handle_source_add_and_wait,
            "source_upload_bytes": self._handle_source_upload_bytes,
            "source_add_drive_file": self._handle_source_add_drive_file,
            "chat_ask": self._handle_chat_ask,
            "chat_configure": self._handle_chat_configure,
            "suggest_prompts": self._handle_suggest_prompts,
            "note_save": self._handle_note_save,
            "studio_list": self._handle_studio_list,
            "studio_generate": self._handle_studio_generate,
            "studio_status": self._handle_studio_status,
            "studio_get_prompt": self._handle_studio_get_prompt,
            "studio_download": self._handle_studio_download,
            "studio_rename": self._handle_studio_rename,
            "studio_retry": self._handle_studio_retry,
            "studio_delete": self._handle_studio_delete,
            "research_start": self._handle_research_start,
            "research_status": self._handle_research_status,
            "research_import": self._handle_research_import,
            "research_cancel": self._handle_research_cancel,
            "share_status": self._handle_share_status,
            "share_set_access": self._handle_share_set_access,
            "share_set_user": self._handle_share_set_user,
            "share_remove_user": self._handle_share_remove_user,
            "server_info": self._handle_server_info,
        }

    def reconnect(self) -> None:
        self.backend.reconnect()

    def list_tools(self) -> list[ToolDefinition]:
        return list(LOCAL_TOOLS)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise NotebookToolError(
                "UNKNOWN_TOOL", "模型请求了未注册的 NotebookLM 工具"
            )
        payload = dict(arguments or {})
        if name == "server_info":
            return handler(payload)
        health = self.health()
        if not health.get("authenticated"):
            raise NotebookToolError(
                "NOTEBOOK_LOGIN_REQUIRED",
                "NotebookLM 登录态不可用，请先执行 /notebook login。",
            )
        return handler(payload)

    def health(self) -> dict[str, Any]:
        checks: list[dict[str, str]] = []
        if not self.profile_path.is_file():
            checks.append(
                _check("profile_file", "failed", "未找到默认账号 storage_state")
            )
            return self._health(
                False, checks, "login_required", "需要先执行 /notebook login"
            )

        checks.append(_check("profile_file", "ok", "默认账号 storage_state 存在"))
        try:
            state = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checks.append(
                _check(
                    "storage_state", "failed", "storage_state 无法读取或不是合法 JSON"
                )
            )
            return self._health(
                False, checks, "profile_invalid", "登录态文件损坏，请重新登录"
            )

        cookies = state.get("cookies")
        origins = state.get("origins")
        if not isinstance(cookies, list) or not isinstance(origins, list):
            checks.append(
                _check("storage_state", "failed", "storage_state 缺少 cookies/origins")
            )
            return self._health(
                False, checks, "profile_invalid", "登录态格式无效，请重新登录"
            )
        checks.append(_check("storage_state", "ok", "storage_state 结构有效"))

        google_cookies = [
            item
            for item in cookies
            if isinstance(item, dict)
            and isinstance(item.get("domain"), str)
            and item["domain"].endswith(".google.com")
        ]
        if google_cookies:
            checks.append(_check("google_session", "ok", "检测到 Google 会话 cookie"))
        else:
            checks.append(
                _check("google_session", "failed", "未检测到 Google 会话 cookie")
            )

        notebooklm_origins = [
            item
            for item in origins
            if isinstance(item, dict)
            and isinstance(item.get("origin"), str)
            and "notebooklm.google" in item["origin"]
        ]
        if notebooklm_origins:
            checks.append(
                _check("notebooklm_origin", "ok", "检测到 NotebookLM origin state")
            )
        else:
            checks.append(
                _check(
                    "notebooklm_origin",
                    "warning",
                    "未检测到 NotebookLM origin state；首次访问时可能仍需页面确认",
                )
            )

        if not google_cookies:
            return self._health(
                False, checks, "login_required", "需要重新登录 NotebookLM"
            )

        try:
            probe = self.backend.probe()
        except Exception:
            checks.append(
                _check(
                    "notebooklm_online",
                    "failed",
                    "storage_state 存在，但 NotebookLM 在线探针失败",
                )
            )
            return self._health(
                False,
                checks,
                "online_probe_failed",
                "登录态文件存在，但无法通过 NotebookLM 在线探针，请重新 /notebook login。",
            )
        checks.append(_check("notebooklm_online", "ok", "notebooks.list 在线探针通过"))
        return self._health(
            True,
            checks,
            "ready",
            "内置工具已通过 NotebookLM 在线探针",
            probe=probe,
        )

    def _health(
        self,
        ready: bool,
        checks: list[dict[str, str]],
        stage: str,
        summary: str,
        probe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "backend": "native",
            "ready": ready,
            "authenticated": ready,
            "stage": stage,
            "summary": summary,
            "profile_path": str(self.profile_path),
            "checks": checks,
            "capabilities": {
                "readiness_probe": "profile_state+notebooks.list",
                "tool_count": len(LOCAL_TOOLS),
                "tool_contract": TOOL_CONTRACT,
                "external_protocol_required": False,
                "bridge": False,
            },
        }
        if probe is not None:
            payload["probe"] = _json_safe(probe)
        return payload

    def _handle_notebook_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("notebook_list", arguments)

    def _handle_notebook_create(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("notebook_create", arguments)

    def _handle_notebook_describe(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("notebook_describe", arguments)

    def _handle_notebook_rename(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("notebook_rename", arguments)

    def _handle_notebook_delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        confirmation = self._confirmation_required("notebook_delete", arguments)
        if confirmation:
            return confirmation
        return self._invoke("notebook_delete", arguments)

    def _handle_source_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_list", arguments)

    def _handle_source_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_read", arguments)

    def _handle_source_rename(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_rename", arguments)

    def _handle_source_delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        confirmation = self._confirmation_required("source_delete", arguments)
        if confirmation:
            return confirmation
        return self._invoke("source_delete", arguments)

    def _handle_source_wait(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_wait", arguments)

    def _handle_source_add(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_add", arguments)

    def _handle_source_add_and_wait(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_add_and_wait", arguments)

    def _handle_source_upload_bytes(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("source_upload_bytes", arguments)

    def _handle_source_add_drive_file(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self._invoke("source_add_drive_file", arguments)

    def _handle_chat_ask(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("chat_ask", arguments)

    def _handle_chat_configure(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("chat_configure", arguments)

    def _handle_suggest_prompts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("suggest_prompts", arguments)

    def _handle_note_save(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("note_save", arguments)

    def _handle_studio_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_list", arguments)

    def _handle_studio_generate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_generate", arguments)

    def _handle_studio_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_status", arguments)

    def _handle_studio_get_prompt(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_get_prompt", arguments)

    def _handle_studio_download(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_download", arguments)

    def _handle_studio_rename(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_rename", arguments)

    def _handle_studio_retry(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("studio_retry", arguments)

    def _handle_studio_delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        confirmation = self._confirmation_required("studio_delete", arguments)
        if confirmation:
            return confirmation
        return self._invoke("studio_delete", arguments)

    def _handle_research_start(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("research_start", arguments)

    def _handle_research_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("research_status", arguments)

    def _handle_research_import(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("research_import", arguments)

    def _handle_research_cancel(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("research_cancel", arguments)

    def _handle_share_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._invoke("share_status", arguments)

    def _handle_share_set_access(self, arguments: dict[str, Any]) -> dict[str, Any]:
        confirmation = self._confirmation_required("share_set_access", arguments)
        if confirmation:
            return confirmation
        return self._invoke("share_set_access", arguments)

    def _handle_share_set_user(self, arguments: dict[str, Any]) -> dict[str, Any]:
        confirmation = self._confirmation_required("share_set_user", arguments)
        if confirmation:
            return confirmation
        return self._invoke("share_set_user", arguments)

    def _handle_share_remove_user(self, arguments: dict[str, Any]) -> dict[str, Any]:
        confirmation = self._confirmation_required("share_remove_user", arguments)
        if confirmation:
            return confirmation
        return self._invoke("share_remove_user", arguments)

    def _handle_server_info(self, arguments: dict[str, Any]) -> dict[str, Any]:
        health = self.health()
        result: dict[str, Any] = {
            "server": "nootbooklm-bot-native",
            "version": _notebooklm_version(),
            "backend": "native",
            "tool_count": len(LOCAL_TOOLS),
            "tool_contract": TOOL_CONTRACT,
            "external_protocol_required": False,
            "bridge": False,
            "auth": {
                "authenticated": health["authenticated"],
                "stage": health["stage"],
                "summary": health["summary"],
            },
            "health": health,
        }
        if arguments.get("include_account") and health.get("authenticated"):
            try:
                backend_result = self.backend.invoke("server_info", arguments)
            except Exception:
                result["account"] = {
                    "available": False,
                    "reason": "account probe failed",
                }
            else:
                if isinstance(backend_result, dict) and "account" in backend_result:
                    result["account"] = backend_result["account"]
        return _success("server_info", result)

    def _confirmation_required(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        if tool_name not in CONFIRMATION_TOOLS or arguments.get("confirm") is True:
            return None
        return {
            "ok": False,
            "tool": tool_name,
            "needs_confirmation": True,
            "message": "该操作会删除或扩大共享范围，请在参数中显式传入 confirm=true 后再执行。",
            "preview": _redacted_preview(arguments),
        }

    def _invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.backend.invoke(tool_name, arguments)
        except NotebookToolError:
            raise
        except Exception as exc:
            safe_message = _safe_exception_message(exc)
            LOGGER.warning(
                "NotebookLM tool %s failed with %s: %s",
                tool_name,
                type(exc).__name__,
                safe_message,
            )
            raise _classify_tool_exception(tool_name, exc) from exc
        return _success(tool_name, result)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_path(root: Any, dotted_path: str) -> Any | None:
    current = root
    for part in dotted_path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if callable(current) else None


def _sdk_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key != "confirm"}


def _exception_text(exc: Exception) -> str:
    values = [type(exc).__name__, str(exc)]
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def _safe_exception_message(exc: Exception) -> str:
    message = " ".join((str(exc) or type(exc).__name__).split())
    if any(marker in message.lower() for marker in SENSITIVE_EXCEPTION_MARKERS):
        return f"{type(exc).__name__}: [redacted]"
    return message[:500] or type(exc).__name__


def _classify_tool_exception(tool_name: str, exc: Exception) -> NotebookToolError:
    text = _exception_text(exc)
    details = {"tool": tool_name, "exception_type": type(exc).__name__}
    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return NotebookToolError(
            "TIMEOUT",
            f"NotebookLM 工具执行超时：{tool_name}。请稍后重试。",
            details=details,
        )
    if any(
        marker in text
        for marker in (
            "401",
            "403",
            "auth",
            "login",
            "unauthor",
            "expired",
            "forbidden",
        )
    ):
        return NotebookToolError(
            "LOGIN_EXPIRED",
            "NotebookLM 登录态可能已失效，请重新执行 /notebook login 后再试。",
            details=details,
        )
    if tool_name == "source_read" and any(
        marker in text
        for marker in (
            "fulltext",
            "full text",
            "unsupported",
            "not supported",
            "source content",
        )
    ):
        return NotebookToolError(
            "SOURCE_READ_UNSUPPORTED",
            "当前 NotebookLM 来源暂不支持读取全文；可以先改用摘要或重新添加来源。",
            details=details,
        )
    if tool_name == "source_read" and any(
        marker in text
        for marker in (
            "processing failed",
            "source processing failed",
            "source_status_error",
            "status error",
        )
    ):
        return NotebookToolError(
            "SOURCE_PROCESSING_FAILED",
            "NotebookLM 来源处理失败，无法继续读取；请删除该来源后重新添加。",
            details=details,
        )
    if tool_name == "source_read" and any(
        marker in text
        for marker in ("not ready", "processing", "preparing", "source status")
    ):
        return NotebookToolError(
            "SOURCE_NOT_READY",
            "NotebookLM 来源仍在处理或尚未可读取；请先执行 source_wait，等待处理完成后再读取。",
            details=details,
        )
    return NotebookToolError(
        "NOTEBOOKLM_UPSTREAM_CHANGED",
        f"NotebookLM 工具执行失败：{tool_name}。上游页面或 SDK 返回格式可能发生变化。",
        details=details,
    )


async def _call_with_supported_args(method: Any, arguments: dict[str, Any]) -> Any:
    signature = inspect.signature(method)
    params = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        supported = {
            key: value for key, value in arguments.items() if value is not None
        }
    else:
        accepted = {
            param.name
            for param in params
            if param.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        supported = {
            key: value
            for key, value in arguments.items()
            if key in accepted and value is not None
        }
    return await _maybe_await(method(**supported))


def _page_payload(
    key: str, items: list[Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    total = len(items)
    offset = int(arguments.get("offset") or 0)
    limit = int(arguments.get("limit") or total or 50)
    page = items[offset : offset + limit]
    return {
        key: _json_safe(page),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
    }


def _slice_text_fields(payload: Any, offset: int, max_chars: int | None) -> None:
    if not isinstance(payload, dict):
        return
    for key in ("content", "text", "markdown"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        payload[f"{key}_char_count"] = len(value)
        limit = max_chars if max_chars is not None else 10000
        payload[key] = value[offset : offset + limit]
        payload[f"{key}_truncated"] = len(payload[key]) < payload[f"{key}_char_count"]


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _decode_upload_bytes(value: Any) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise NotebookToolError(
            "NOTEBOOKLM_ARGUMENTS", "source_upload_bytes 需要提供 bytes_base64。"
        )
    if len(value) > SOURCE_UPLOAD_BYTES_B64_LIMIT:
        raise NotebookToolError(
            "NOTEBOOKLM_ARGUMENTS",
            "source_upload_bytes 的 bytes_base64 不能超过 10000 个字符。",
        )
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise NotebookToolError(
            "NOTEBOOKLM_ARGUMENTS", "bytes_base64 必须是标准 base64。"
        ) from exc
    if not raw:
        raise NotebookToolError(
            "NOTEBOOKLM_ARGUMENTS", "source_upload_bytes 不能上传空文件。"
        )
    return raw


def _safe_upload_filename(value: Any) -> str:
    text = str(value or "upload.bin").strip().replace("\\", "/")
    filename = Path(text).name.strip()
    return filename or "upload.bin"


def _notebooklm_version() -> str:
    try:
        from notebooklm._version_info import version_string
    except Exception:
        return "unknown"
    return str(version_string())


async def _account_snapshot(client: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"available": False}
    get_email = getattr(client, "get_account_email", None)
    if callable(get_email):
        try:
            snapshot["email"] = await _maybe_await(get_email(live_fallback=True))
        except Exception:
            snapshot["email"] = None
    get_authuser = getattr(client, "get_account_authuser", None)
    if callable(get_authuser):
        try:
            snapshot["authuser"] = await _maybe_await(get_authuser())
        except Exception:
            snapshot["authuser"] = None
    settings_api = getattr(client, "settings", None)
    get_settings = getattr(settings_api, "get_user_settings", None)
    if callable(get_settings):
        try:
            settings = await _maybe_await(get_settings())
        except Exception:
            snapshot["reason"] = "settings unavailable"
        else:
            limits = getattr(settings, "limits", None)
            snapshot.update(
                {
                    "available": True,
                    "notebook_limit": getattr(limits, "notebook_limit", None),
                    "source_limit": getattr(limits, "source_limit", None),
                    "output_language": getattr(settings, "output_language", None),
                }
            )
    return snapshot


def _find_item_by_id(items: list[Any], item_id: str) -> Any | None:
    for item in items:
        try:
            if _item_id(item) == item_id:
                return item
        except NotebookToolError:
            continue
    return None


def _source_status_details(source: Any) -> dict[str, Any]:
    return {
        "source_id": _item_id(source),
        "source_title": _item_title(source),
        "source_status": _status_label(source),
    }


def _ensure_source_ready_for_read(source: Any) -> None:
    status = _status_label(source)
    if not status or status in READY_SOURCE_STATUSES:
        return
    details = _source_status_details(source)
    if status in FAILED_SOURCE_STATUSES:
        raise NotebookToolError(
            "SOURCE_PROCESSING_FAILED",
            "NotebookLM 来源处理失败，无法继续读取；请删除该来源后重新添加，处理成功后再读取。",
            details=details,
        )
    if status in PROCESSING_SOURCE_STATUSES:
        raise NotebookToolError(
            "SOURCE_NOT_READY",
            "NotebookLM 来源仍在处理；请先执行 source_wait，等待处理完成后再读取。",
            details=details,
        )
    raise NotebookToolError(
        "SOURCE_NOT_READY",
        "NotebookLM 来源尚未处于可读取状态；请先执行 source_wait，等待处理完成后再读取。",
        details=details,
    )


def _status_label(item: Any) -> str:
    raw_status = getattr(item, "status", None)
    raw_label = getattr(item, "status_label", None)
    for raw in (raw_status, raw_label):
        label = _sdk_source_status_label(raw)
        if label and label != "unknown":
            return label
    raw = raw_label if raw_label is not None else raw_status
    if raw is None:
        return ""
    if hasattr(raw, "name"):
        return str(raw.name).strip().lower()
    label = str(getattr(raw, "value", raw)).strip().lower()
    return SOURCE_STATUS_CODE_LABELS.get(label, label)


def _sdk_source_status_label(raw: Any) -> str:
    if raw is None:
        return ""
    try:
        from notebooklm.rpc.types import source_status_to_str
    except Exception:
        return ""
    value = getattr(raw, "value", raw)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            value = int(stripped)
    try:
        label = str(source_status_to_str(value)).strip().lower()
        return SOURCE_STATUS_CODE_LABELS.get(label, label)
    except Exception:
        return ""


def _enum_choice(
    enum_type: Any,
    value: Any,
    aliases: dict[str, str],
    *,
    default: str,
) -> Any:
    if value is None:
        return getattr(enum_type, default)
    normalized = str(value).strip().lower().replace("-", "_")
    member_name = aliases.get(normalized) or normalized.upper()
    return getattr(enum_type, member_name, getattr(enum_type, default))


def _normalize_kind(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _artifact_kind(artifact: Any) -> str:
    raw = getattr(artifact, "artifact_type", None) or getattr(artifact, "type", None)
    if hasattr(raw, "value"):
        raw = raw.value
    return _normalize_kind(raw)


def _item_id(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("id") or item.get("notebook_id") or item.get("source_id")
    else:
        value = getattr(item, "id", None)
    if value is None:
        raise NotebookToolError(
            "NOTEBOOKLM_BAD_RESPONSE", "NotebookLM 返回了缺少 id 的对象。"
        )
    return str(value)


def _item_title(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("title") or item.get("name")
    else:
        value = getattr(item, "title", None) or getattr(item, "name", None)
    return str(value) if value is not None else ""


def _extract_ref_id(ref: str) -> str:
    text = str(ref).strip()
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        query = parse_qs(parsed.query)
        for key in ("notebook_id", "source_id", "artifact_id", "id"):
            values = query.get(key)
            if values:
                return values[0]
        parts = [part for part in parsed.path.split("/") if part]
        if parts:
            return parts[-1]
    return text


def _looks_like_id(value: str) -> bool:
    return bool(value) and " " not in value and len(value) >= 8


def _success(tool_name: str, result: Any) -> dict[str, Any]:
    if (
        isinstance(result, dict)
        and result.get("ok") is True
        and result.get("tool") == tool_name
    ):
        return result
    return {"ok": True, "tool": tool_name, "result": _json_safe(result)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _json_safe(value.dict())
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _redacted_preview(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted = {}
    for key, value in arguments.items():
        redacted[key] = (
            "[provided]"
            if key.lower() in {"bytes_base64", "content_base64", "message"}
            else value
        )
    return redacted


def build_notebook_provider(settings: Any) -> NotebookToolProvider:
    return LocalNotebookToolProvider(settings.profile_path)
