from logging import Logger

from slack_bolt import BoltContext, Say, SetStatus
from slack_sdk import WebClient

from agent.llm_caller import call_llm, format_error_message
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
        thread_ts = payload["thread_ts"]
        set_status(
            status="正在研究…",
            loading_messages=["正在连接 NotebookLM…", "正在整理研究资料…"],
        )
        streamer = client.chat_stream(
            channel=channel_id,
            recipient_team_id=context.team_id,
            recipient_user_id=context.user_id,
            thread_ts=thread_ts,
            task_display_mode="timeline",
        )
        prompts: list[dict] = [
            {"role": "user", "content": str(message.get("text", ""))[:20_000]}
        ]
        call_llm(streamer, prompts)
        streamer.stop(blocks=create_feedback_block())
    except Exception as exc:
        logger.exception("处理 Slack Assistant 消息失败")
        say(format_error_message(exc))
