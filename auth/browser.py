from __future__ import annotations

from typing import Protocol


class BrowserWorker(Protocol):
    """Starts and tears down an isolated, one-session browser viewer."""

    def start(self, session_id: str) -> str: ...

    def cancel(self, session_id: str) -> None: ...


class ExternalBrowserWorker:
    """Delegates browser isolation and noVNC to a separately secured worker."""

    def __init__(self, viewer_base_url: str):
        if not viewer_base_url.startswith("https://"):
            raise ValueError("远程浏览器地址必须使用 HTTPS")
        self.viewer_base_url = viewer_base_url.rstrip("/")

    def start(self, session_id: str) -> str:
        return f"{self.viewer_base_url}/session/{session_id}"

    def cancel(self, session_id: str) -> None:
        return None
