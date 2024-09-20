class Action:
    def __init__(self):
        pass


class State:
    actions: list[Action]

    def __init__(self):
        self.actions = []

    def add_action(self, action: Action):
        self.actions.append(action)
