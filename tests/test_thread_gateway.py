from agent.thread_gateway import (
    build_slack_thread_prompt,
    build_thread_context_key,
    compact_thread_messages,
    is_processable_assistant_message,
    resolve_thread_ts,
)


class FakeSlackClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def conversations_replies(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]


class FailingSlackClient:
    def conversations_replies(self, **kwargs):
        raise RuntimeError("missing_scope")


def user_prompt_text(prompts):
    return prompts[1]["content"]


def test_resolves_stable_thread_context_key():
    assert resolve_thread_ts({"ts": "200.000"}) == "200.000"
    assert resolve_thread_ts({"thread_ts": "100.000", "ts": "200.000"}) == "100.000"
    assert build_thread_context_key("T1", "C1", "100.000") == "slack:T1:C1:100.000"


def test_builds_prompt_from_slack_thread_history():
    client = FakeSlackClient(
        [
            {
                "messages": [
                    {"ts": "100.000", "user": "U1", "text": "root task"},
                    {"ts": "101.000", "user": "U2", "text": "extra context"},
                    {"ts": "102.000", "user": "U1", "text": "current request"},
                ]
            }
        ]
    )

    prompts = build_slack_thread_prompt(
        client=client,
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="102.000",
        current_user_id="U1",
        current_text="current request",
    )

    assert prompts[0]["role"] == "developer"
    content = user_prompt_text(prompts)
    assert "context_key: slack:T1:C1:100.000" in content
    assert "trigger_source: app_mention" in content
    assert "[100.000] user:U1: root task" in content
    assert "[101.000] user:U2: extra context" in content
    assert "Current user request:\ncurrent request" in content
    assert client.calls == [{"channel": "C1", "ts": "100.000", "limit": 200}]


def test_fetches_paginated_slack_thread_history():
    client = FakeSlackClient(
        [
            {
                "messages": [
                    {"ts": "100.000", "user": "U1", "text": "root task"},
                ],
                "response_metadata": {"next_cursor": "next-page"},
            },
            {
                "messages": [
                    {"ts": "101.000", "user": "U2", "text": "page two context"},
                    {"ts": "102.000", "user": "U1", "text": "current request"},
                ],
            },
        ]
    )

    prompts = build_slack_thread_prompt(
        client=client,
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="102.000",
        current_user_id="U1",
        current_text="current request",
    )

    content = user_prompt_text(prompts)
    assert "[101.000] user:U2: page two context" in content
    assert client.calls == [
        {"channel": "C1", "ts": "100.000", "limit": 200},
        {"channel": "C1", "ts": "100.000", "limit": 200, "cursor": "next-page"},
    ]


def test_appends_current_message_when_replies_lag_behind_event():
    client = FakeSlackClient(
        [
            {
                "messages": [
                    {"ts": "100.000", "user": "U1", "text": "root task"},
                ]
            }
        ]
    )

    prompts = build_slack_thread_prompt(
        client=client,
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="102.000",
        current_user_id="U1",
        current_text="current request",
    )

    content = user_prompt_text(prompts)
    assert "[100.000] user:U1: root task" in content
    assert "[102.000] user:U1: current request" in content


def test_fetch_failure_falls_back_to_current_request():
    prompts = build_slack_thread_prompt(
        client=FailingSlackClient(),
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="102.000",
        current_user_id="U1",
        current_text="current request",
    )

    content = user_prompt_text(prompts)
    assert "context_fetch_error: RuntimeError: missing_scope" in content
    assert "[102.000] user:U1: current request" in content
    assert "Current user request:\ncurrent request" in content


def test_compacts_long_threads_by_preserving_root_and_recent_messages():
    messages = [
        {"ts": f"{index}.000", "user": "U1", "text": f"message {index}"}
        for index in range(10)
    ]

    compaction = compact_thread_messages(messages, max_recent_messages=4)

    assert compaction.omitted_count == 6
    assert [message["text"] for message in compaction.transcript_messages] == [
        "message 0",
        "message 7",
        "message 8",
        "message 9",
    ]
    assert compaction.omitted_summary is not None
    assert "omitted_time_range: 1.000 -> 6.000" in compaction.omitted_summary
    assert "message 1" in compaction.omitted_summary


def test_long_thread_prompt_includes_omitted_summary_with_links_and_work_items():
    client = FakeSlackClient(
        [
            {
                "messages": [
                    {"ts": "100.000", "user": "U1", "text": "root task"},
                    {
                        "ts": "101.000",
                        "user": "U2",
                        "text": "Need to preserve https://example.com and ZER-88",
                    },
                    {"ts": "102.000", "user": "U2", "text": "middle decision"},
                    {"ts": "103.000", "user": "U1", "text": "current request"},
                ]
            }
        ]
    )

    prompts = build_slack_thread_prompt(
        client=client,
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="103.000",
        current_user_id="U1",
        current_text="current request",
        max_recent_messages=2,
    )

    content = user_prompt_text(prompts)
    assert "Compacted omitted thread summary:" in content
    assert "referenced_links_or_work_items: https://example.com, ZER-88" in content
    assert "[100.000] user:U1: root task" in content
    assert "[103.000] user:U1: current request" in content


def test_assistant_trigger_source_has_dedicated_developer_instruction():
    prompts = build_slack_thread_prompt(
        client=FakeSlackClient([{"messages": []}]),
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="102.000",
        current_user_id="U1",
        current_text="current request",
        trigger_source="assistant_user_message",
    )

    assert "Slack Assistant surface" in prompts[0]["content"]
    assert "trigger_source: assistant_user_message" in user_prompt_text(prompts)


def test_assistant_message_guard_skips_empty_or_bot_messages():
    assert is_processable_assistant_message({"text": "hello"}, {}) is True
    assert is_processable_assistant_message({"text": ""}, {}) is False
    assert (
        is_processable_assistant_message({"text": "hello", "bot_id": "B1"}, {}) is False
    )
    assert (
        is_processable_assistant_message(
            {"text": "hello", "subtype": "bot_message"}, {}
        )
        is False
    )
