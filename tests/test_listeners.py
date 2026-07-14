import logging
from unittest.mock import MagicMock, patch

from listeners.assistant.message import message
from listeners.events.app_mentioned import app_mentioned_callback


def _logger():
    return logging.getLogger("test_listeners")


def _streamer():
    return MagicMock()


@patch("listeners.events.app_mentioned.call_llm")
@patch("listeners.events.app_mentioned.build_slack_thread_prompt")
def test_app_mentioned_invokes_gateway_with_channel_trigger(mock_build, mock_llm):
    mock_build.return_value = [{"role": "user", "content": "prompt"}]
    client = MagicMock()
    client.chat_stream.return_value = _streamer()
    logger = _logger()
    event = {
        "channel": "C1",
        "team": "T1",
        "text": "<@B123> follow up",
        "ts": "200.000",
        "thread_ts": "100.000",
        "user": "U1",
    }

    app_mentioned_callback(client, event, logger, MagicMock())

    mock_build.assert_called_once_with(
        client=client,
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="200.000",
        current_user_id="U1",
        current_text="<@B123> follow up",
        logger=logger,
        trigger_source="app_mention",
    )
    mock_llm.assert_called_once()
    client.assistant_threads_setStatus.assert_called_once()
    client.chat_stream.assert_called_once()


def test_app_mentioned_ignores_missing_channel_or_thread_ts():
    client = MagicMock()
    logger = _logger()

    app_mentioned_callback(
        client,
        {"ts": "200.000", "user": "U1"},
        logger,
        MagicMock(),
    )
    app_mentioned_callback(
        client,
        {"channel": "C1", "user": "U1"},
        logger,
        MagicMock(),
    )

    client.chat_stream.assert_not_called()
    client.assistant_threads_setStatus.assert_not_called()


@patch("listeners.assistant.message.call_llm")
def test_assistant_message_ignores_empty_or_bot_messages(mock_llm):
    context = MagicMock(team_id="T1", user_id="U1")
    payload = {"channel": "C1", "thread_ts": "100.000", "ts": "200.000"}

    for message_payload in (
        {"text": "   "},
        {"text": "hello", "bot_id": "B1"},
        {"text": "hello", "subtype": "bot_message"},
    ):
        client = MagicMock()
        mock_llm.reset_mock()
        message(
            client=client,
            context=context,
            logger=_logger(),
            message=message_payload,
            payload=payload,
            say=MagicMock(),
            set_status=MagicMock(),
        )
        mock_llm.assert_not_called()
        client.chat_stream.assert_not_called()


@patch("listeners.assistant.message.call_llm")
@patch("listeners.assistant.message.build_slack_thread_prompt")
def test_assistant_message_invokes_gateway_with_assistant_trigger(mock_build, mock_llm):
    mock_build.return_value = [{"role": "user", "content": "prompt"}]
    client = MagicMock()
    client.chat_stream.return_value = _streamer()
    context = MagicMock(team_id="T1", user_id="U1")
    logger = _logger()

    message(
        client=client,
        context=context,
        logger=logger,
        message={"text": "help me plan", "ts": "200.000", "user": "U1"},
        payload={"channel": "C1", "thread_ts": "100.000", "ts": "200.000"},
        say=MagicMock(),
        set_status=MagicMock(),
    )

    mock_build.assert_called_once_with(
        client=client,
        team_id="T1",
        channel_id="C1",
        thread_ts="100.000",
        current_ts="200.000",
        current_user_id="U1",
        current_text="help me plan",
        logger=logger,
        trigger_source="assistant_user_message",
    )
    mock_llm.assert_called_once()
    client.chat_stream.assert_called_once()
