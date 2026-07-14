from datetime import UTC, datetime, timedelta

import pytest

from auth.sessions import LoginSessionStore, LoginStatus, SQLiteLoginSessionStore


def _create(store, now):
    return store.create(
        team_id="T1", channel_id="C1", thread_ts="1.1", user_id="U1", now=now
    )


def test_token_is_hashed_one_time_and_never_stored_plaintext():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = LoginSessionStore()
    session, token = _create(store, now)
    assert token not in repr(session)
    assert store.consume(token, now).status == LoginStatus.BROWSER_STARTED
    assert store.consume(token, now) is None


def test_only_one_active_login():
    store = LoginSessionStore()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    _create(store, now)
    with pytest.raises(ValueError, match="已有"):
        _create(store, now)


def test_expiry_and_invalid_transitions():
    store = LoginSessionStore(ttl_seconds=60)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    session, token = _create(store, now)
    assert store.consume(token, now + timedelta(seconds=61)) is None
    assert store.get(session.session_id).status == LoginStatus.EXPIRED
    with pytest.raises(ValueError, match="结束"):
        store.transition(session.session_id, LoginStatus.AUTHENTICATED)


def test_authentication_requires_browser_started():
    store = LoginSessionStore()
    session, _ = _create(store, datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="非法"):
        store.transition(session.session_id, LoginStatus.AUTHENTICATED)


def test_sqlite_store_shares_one_time_state_between_process_instances(tmp_path):
    path = str(tmp_path / "sessions.db")
    first = SQLiteLoginSessionStore(path)
    session, token = first.create(
        team_id="T1", channel_id="C1", thread_ts=None, user_id="U1"
    )
    second = SQLiteLoginSessionStore(path)
    assert second.consume(token).session_id == session.session_id
    assert first.consume(token) is None
