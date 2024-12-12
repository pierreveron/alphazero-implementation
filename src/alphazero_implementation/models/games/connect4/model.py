import torch
import torch.nn.functional as F
from simulator.game.connect import Action, State  # type: ignore[attr-defined]

from ...base import ActionPolicy, Model, Value  # type: ignore[import]


class Connect4Model(Model):
    """
    An abstract class for a Connect4 model. The predict method is implemented here to generalize
    the prediction process for all Connect4 models.
    """

    def __init__(self):
        super().__init__()
        self.board_height = 6
        self.board_width = 7

    def predict(self, states: list[State]) -> tuple[list[ActionPolicy], list[Value]]:
        x = self._states_to_tensor(states)

        policy_logits, values_tensor = self.forward(x)

        policies: list[ActionPolicy] = []
        for i, state in enumerate(states):
            d: dict[Action, float] = {}
            policy = policy_logits[i]
            valid_actions = state.actions
            valid_logits = torch.tensor(
                [policy[action.column].item() for action in valid_actions]
            )

            # Apply softmax to get probabilities
            probabilities = F.softmax(valid_logits, dim=0)

            for action, prob in zip(valid_actions, probabilities):
                d[action] = prob.item()
            policies.append(d)

        values: list[Value] = values_tensor.detach().tolist()  # type: ignore[attr-defined]

        return policies, values

    def _policies_to_tensor(self, policies: list[ActionPolicy]) -> torch.Tensor:
        policy_targets = torch.zeros((len(policies), self.board_width))
        for i, policy in enumerate(policies):
            for action, prob in policy.items():
                column = action.column
                policy_targets[i, column] = prob
        return policy_targets
