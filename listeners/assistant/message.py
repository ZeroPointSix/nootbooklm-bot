from logging import Logger

from slack_bolt import BoltContext, Say, SetStatus
from slack_sdk import WebClient

from agent.llm_caller import call_llm, format_error_message
from agent.thread_gateway import (
    build_slack_thread_prompt,
    is_processable_assistant_message,
    resolve_thread_ts,
)
from listeners.views.feedback_block import create_feedback_block


def message(
    client: WebClient,
    context: BoltContext,
    logger: Logger,
    message: dict,
    payload: dict,
    say: Say,
    set_status: SetStatus,
):
    try:
        channel_id = payload["channel"]
        thread_ts = resolve_thread_ts(payload)
        current_ts = message.get("ts") or payload.get("ts")
        user_id = context.user_id or message.get("user")
        text = message.get("text") or payload.get("text") or ""

        if not is_processable_assistant_message(message, payload):
            logger.debug("Ignoring assistant message without processable user text")
            return
        if not thread_ts:
            logger.warning("Ignoring assistant message without thread_ts")
            return

        set_status(
            status="正在研究…",
            loading_messages=["正在连接 NotebookLM…", "正在整理研究资料…"],
        )
        streamer = client.chat_stream(
            channel=channel_id,
            recipient_team_id=context.team_id,
            recipient_user_id=user_id,
            thread_ts=thread_ts,
            task_display_mode="timeline",
        )
        prompts = build_slack_thread_prompt(
            client=client,
            team_id=context.team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            current_ts=current_ts,
            current_user_id=user_id,
            current_text=str(text)[:20_000],
            logger=logger,
            trigger_source="assistant_user_message",
        )
        call_llm(streamer, prompts)
        streamer.stop(blocks=create_feedback_block())
    except Exception as exc:
        logger.exception("处理 Slack Assistant 消息失败")
        say(format_error_message(exc))
