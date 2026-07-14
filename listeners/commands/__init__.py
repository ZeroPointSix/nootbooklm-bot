from slack_bolt import App

from listeners.commands.notebook import notebook_command


def register(app: App) -> None:
    app.command("/notebook")(notebook_command)
