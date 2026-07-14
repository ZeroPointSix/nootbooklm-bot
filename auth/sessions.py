from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path


class LoginStatus(StrEnum):
    PENDING = "pending"
    BROWSER_STARTED = "browser_started"
    AUTHENTICATED = "authenticated"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


TERMINAL = {
    LoginStatus.AUTHENTICATED,
    LoginStatus.FAILED,
    LoginStatus.EXPIRED,
    LoginStatus.CANCELLED,
}


@dataclass(frozen=True)
class LoginSession:
    session_id: str
    token_hash: str
    slack_team_id: str
    slack_channel_id: str
    slack_thread_ts: str | None
    slack_user_id: str
    status: LoginStatus
    created_at: datetime
    expires_at: datetime
    error_code: str | None = None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class LoginSessionStore:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, LoginSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_ts: str | None,
        user_id: str,
        now: datetime | None = None,
    ) -> tuple[LoginSession, str]:
        now = now or datetime.now(UTC)
        with self._lock:
            self._expire(now)
            if any(
                session.status not in TERMINAL for session in self._sessions.values()
            ):
                raise ValueError("已有一个登录会话正在进行")
            token = secrets.token_urlsafe(32)
            session = LoginSession(
                session_id=secrets.token_urlsafe(18),
                token_hash=_token_hash(token),
                slack_team_id=team_id,
                slack_channel_id=channel_id,
                slack_thread_ts=thread_ts,
                slack_user_id=user_id,
                status=LoginStatus.PENDING,
                created_at=now,
                expires_at=now + timedelta(seconds=self.ttl_seconds),
            )
            self._sessions[session.session_id] = session
            return session, token

    def consume(self, token: str, now: datetime | None = None) -> LoginSession | None:
        now = now or datetime.now(UTC)
        digest = _token_hash(token)
        with self._lock:
            self._expire(now)
            for session_id, session in self._sessions.items():
                if secrets.compare_digest(session.token_hash, digest):
                    if session.status != LoginStatus.PENDING:
                        return None
                    updated = replace(session, status=LoginStatus.BROWSER_STARTED)
                    self._sessions[session_id] = updated
                    return updated
        return None

    def get(self, session_id: str, now: datetime | None = None) -> LoginSession | None:
        now = now or datetime.now(UTC)
        with self._lock:
            self._expire(now)
            return self._sessions.get(session_id)

    def transition(
        self, session_id: str, status: LoginStatus, error_code: str | None = None
    ) -> LoginSession:
        with self._lock:
            session = self._sessions[session_id]
            if session.status in TERMINAL:
                raise ValueError("登录会话已经结束")
            allowed = {
                LoginStatus.PENDING: {LoginStatus.CANCELLED, LoginStatus.EXPIRED},
                LoginStatus.BROWSER_STARTED: {
                    LoginStatus.AUTHENTICATED,
                    LoginStatus.FAILED,
                    LoginStatus.CANCELLED,
                    LoginStatus.EXPIRED,
                },
            }
            if status not in allowed.get(session.status, set()):
                raise ValueError("非法的登录状态转换")
            updated = replace(session, status=status, error_code=error_code)
            self._sessions[session_id] = updated
            return updated

    def active(self, now: datetime | None = None) -> LoginSession | None:
        now = now or datetime.now(UTC)
        with self._lock:
            self._expire(now)
            return next(
                (
                    item
                    for item in self._sessions.values()
                    if item.status not in TERMINAL
                ),
                None,
            )

    def cancel_active(self) -> LoginSession | None:
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.status not in TERMINAL:
                    updated = replace(session, status=LoginStatus.CANCELLED)
                    self._sessions[session_id] = updated
                    return updated
        return None

    def _expire(self, now: datetime) -> None:
        for session_id, session in list(self._sessions.items()):
            if session.status not in TERMINAL and now >= session.expires_at:
                self._sessions[session_id] = replace(
                    session, status=LoginStatus.EXPIRED, error_code="LOGIN_TIMEOUT"
                )


class SQLiteLoginSessionStore(LoginSessionStore):
    """Persistent store shared by the bot and auth service processes."""

    def __init__(self, path: str, ttl_seconds: int = 600):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS login_sessions (
            session_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
            slack_team_id TEXT NOT NULL, slack_channel_id TEXT NOT NULL,
            slack_thread_ts TEXT, slack_user_id TEXT NOT NULL, status TEXT NOT NULL,
            created_at TEXT NOT NULL, expires_at TEXT NOT NULL, error_code TEXT)"""
        )

    @staticmethod
    def _row(row) -> LoginSession | None:
        if row is None:
            return None
        return LoginSession(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            LoginStatus(row[6]),
            datetime.fromisoformat(row[7]),
            datetime.fromisoformat(row[8]),
            row[9],
        )

    def _expire_db(self, now: datetime) -> None:
        self._db.execute(
            """UPDATE login_sessions SET status=?, error_code=?
            WHERE status IN (?, ?) AND expires_at <= ?""",
            (
                LoginStatus.EXPIRED,
                "LOGIN_TIMEOUT",
                LoginStatus.PENDING,
                LoginStatus.BROWSER_STARTED,
                now.isoformat(),
            ),
        )

    def create(self, *, team_id, channel_id, thread_ts, user_id, now=None):
        now = now or datetime.now(UTC)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._expire_db(now)
                if self._db.execute(
                    "SELECT 1 FROM login_sessions WHERE status IN (?, ?) LIMIT 1",
                    (LoginStatus.PENDING, LoginStatus.BROWSER_STARTED),
                ).fetchone():
                    raise ValueError("已有一个登录会话正在进行")
                token = secrets.token_urlsafe(32)
                session = LoginSession(
                    secrets.token_urlsafe(18),
                    _token_hash(token),
                    team_id,
                    channel_id,
                    thread_ts,
                    user_id,
                    LoginStatus.PENDING,
                    now,
                    now + timedelta(seconds=self.ttl_seconds),
                )
                self._db.execute(
                    "INSERT INTO login_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session.session_id,
                        session.token_hash,
                        team_id,
                        channel_id,
                        thread_ts,
                        user_id,
                        session.status,
                        session.created_at.isoformat(),
                        session.expires_at.isoformat(),
                        None,
                    ),
                )
                self._db.execute("COMMIT")
                return session, token
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def consume(self, token: str, now=None):
        now = now or datetime.now(UTC)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            self._expire_db(now)
            session = self._row(
                self._db.execute(
                    "SELECT * FROM login_sessions WHERE token_hash=?",
                    (_token_hash(token),),
                ).fetchone()
            )
            if not session or session.status != LoginStatus.PENDING:
                self._db.execute("COMMIT")
                return None
            self._db.execute(
                "UPDATE login_sessions SET status=? WHERE session_id=?",
                (LoginStatus.BROWSER_STARTED, session.session_id),
            )
            self._db.execute("COMMIT")
            return replace(session, status=LoginStatus.BROWSER_STARTED)

    def get(self, session_id: str, now=None):
        now = now or datetime.now(UTC)
        with self._lock:
            self._expire_db(now)
            return self._row(
                self._db.execute(
                    "SELECT * FROM login_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
            )

    def transition(self, session_id, status, error_code=None):
        with self._lock:
            session = self.get(session_id)
            if not session or session.status in TERMINAL:
                raise ValueError("登录会话已经结束")
            allowed = {
                LoginStatus.PENDING: {LoginStatus.CANCELLED, LoginStatus.EXPIRED},
                LoginStatus.BROWSER_STARTED: {
                    LoginStatus.AUTHENTICATED,
                    LoginStatus.FAILED,
                    LoginStatus.CANCELLED,
                    LoginStatus.EXPIRED,
                },
            }
            if status not in allowed.get(session.status, set()):
                raise ValueError("非法的登录状态转换")
            self._db.execute(
                "UPDATE login_sessions SET status=?, error_code=? WHERE session_id=?",
                (status, error_code, session_id),
            )
            return replace(session, status=status, error_code=error_code)

    def active(self, now=None):
        now = now or datetime.now(UTC)
        with self._lock:
            self._expire_db(now)
            return self._row(
                self._db.execute(
                    "SELECT * FROM login_sessions WHERE status IN (?, ?) LIMIT 1",
                    (LoginStatus.PENDING, LoginStatus.BROWSER_STARTED),
                ).fetchone()
            )

    def cancel_active(self):
        session = self.active()
        return (
            self.transition(session.session_id, LoginStatus.CANCELLED)
            if session
            else None
        )
