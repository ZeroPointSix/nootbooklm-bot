from __future__ import annotations

import hmac

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from agent.llm_caller import get_runtime
from auth.browser import ExternalBrowserWorker
from auth.sessions import LoginStatus
from config import Settings
from listeners.commands.notebook import get_profile_manager, get_session_store

settings = Settings.from_env()
app = FastAPI(
    title="NotebookLM Login Service",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.on_event("startup")
def validate_configuration() -> None:
    settings.validate_auth()


class CompleteRequest(BaseModel):
    session_id: str
    storage_state: dict


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/notebooklm/{token}", response_class=HTMLResponse)
def begin_login(token: str):
    session = get_session_store().consume(token)
    if not session:
        raise HTTPException(status_code=410, detail="链接无效、已使用或已过期")
    if not settings.auth_browser_viewer_url:
        raise HTTPException(status_code=503, detail="远程浏览器尚未配置")
    viewer = ExternalBrowserWorker(settings.auth_browser_viewer_url)
    response = RedirectResponse(viewer.start(session.session_id), status_code=303)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.post("/internal/auth/notebooklm/complete")
def complete_login(
    request: CompleteRequest,
    x_internal_token: str | None = Header(default=None),
):
    expected = settings.auth_internal_token or ""
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="未授权")
    session = get_session_store().get(request.session_id)
    if not session or session.status != LoginStatus.BROWSER_STARTED:
        raise HTTPException(status_code=409, detail="登录会话状态无效")

    def verify(_path) -> bool:
        try:
            return bool(get_runtime().mcp.list_tools())
        except Exception:
            return False

    try:
        get_profile_manager().install(request.storage_state, verify=verify)
        get_session_store().transition(request.session_id, LoginStatus.AUTHENTICATED)
    except Exception as exc:
        get_session_store().transition(
            request.session_id, LoginStatus.FAILED, "AUTH_VERIFICATION_FAILED"
        )
        raise HTTPException(status_code=422, detail="认证验证失败") from exc
    return {"status": "authenticated"}
