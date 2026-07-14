from slack_bolt import App

from listeners import actions, assistant, commands, events


def register_listeners(app: App):
    actions.register(app)
    assistant.register(app)
    commands.register(app)
    events.register(app)
