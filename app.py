import logging
import os

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from listeners import register_listeners
from config import Settings

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

settings = Settings.from_env()
settings.validate_bot()

# Initialization
logging.basicConfig(level=logging.DEBUG)

app = App(
    token=settings.slack_bot_token,
    client=WebClient(
        base_url=os.environ.get("SLACK_API_URL", "https://slack.com/api"),
        token=settings.slack_bot_token,
    ),
)

# Register Listeners
register_listeners(app)

# Start Bolt app
if __name__ == "__main__":
    SocketModeHandler(app, settings.slack_app_token).start()
