import torch
import torch.nn.functional as F
from simulator.game.connect import Action, State  # type: ignore[attr-defined]

from alphazero_implementation.models.model import (  # type: ignore[import]
    ActionPolicy,
    Model,
    Value,
)


class Connect4Model(Model):
    """
    An abstract class for a Connect4 model. The predict method is implemented here to generalize
    the prediction process for all Connect4 models.
    """

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
