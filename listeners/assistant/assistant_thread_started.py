from logging import Logger

from slack_bolt import Say, SetSuggestedPrompts


def assistant_thread_started(
    say: Say,
    set_suggested_prompts: SetSuggestedPrompts,
    logger: Logger,
):
    """
    Handle the assistant thread start event by greeting the user and setting suggested prompts.

    Args:
        say: Function to send messages to the thread from the app
        set_suggested_prompts: Function to configure suggested prompt options
        logger: Logger instance for error tracking
    """
    try:
        say("今天想用 NotebookLM 研究什么？")
        set_suggested_prompts(
            prompts=[
                {
                    "title": "列出 Notebook",
                    "message": "列出当前账号可用的 Notebook。",
                },
                {
                    "title": "检查登录状态",
                    "message": "检查 NotebookLM 登录状态。",
                },
            ]
        )
    except Exception:
        logger.exception("处理 assistant_thread_started 失败")
        say(":warning: NotebookLM Assistant 初始化失败，请稍后重试。")
