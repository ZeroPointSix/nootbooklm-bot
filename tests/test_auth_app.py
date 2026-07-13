from types import SimpleNamespace

from auth.sessions import LoginSessionStore
from auth_app import CompleteRequest, complete_login


VALID_STATE = {
    "cookies": [{"name": "SID", "value": "secret", "domain": ".google.com"}],
    "origins": [],
}


class Profile:
    def __init__(self):
        self.installed = None

    def install(self, state, *, verify):
        self.installed = state
        assert verify(None) is True


def test_complete_login_message_does_not_claim_mcp_is_ready(monkeypatch):
    store = LoginSessionStore()
    session, token = store.create(
        team_id="T1", channel_id="C1", thread_ts="1.1", user_id="U1"
    )
    store.consume(token)
    profile = Profile()
    posted = []

    class SlackClient:
        def __init__(self, token):
            assert token == "xoxb-test"

        def chat_postMessage(self, **kwargs):
            posted.append(kwargs)

    monkeypatch.setattr(
        "auth_app.settings",
        SimpleNamespace(auth_internal_token="x" * 32, slack_bot_token="xoxb-test"),
    )
    monkeypatch.setattr("auth_app.get_session_store", lambda: store)
    monkeypatch.setattr("auth_app.get_profile_manager", lambda: profile)
    monkeypatch.setattr("auth_app.WebClient", SlackClient)

    response = complete_login(
        CompleteRequest(session_id=session.session_id, storage_state=VALID_STATE),
        x_internal_token="x" * 32,
    )

    assert response == {"status": "authenticated"}
    assert profile.installed == VALID_STATE
    assert posted
    message = posted[0]["text"]
    assert "登录态已保存" in message
    assert "在线验证" in message
    assert "默认账号现在可以使用" not in message
