import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel


class Connect2Model(BaseModel):
    def __init__(self, board_size: int, action_size: int, device: torch.device):
        super(Connect2Model, self).__init__()  # type: ignore[no-untyped-call]

        self.device = device
        self.size = board_size
        self.action_size = action_size

        self.fc1 = nn.Linear(in_features=self.size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)

        # Two heads on our network
        self.action_head = nn.Linear(in_features=16, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=16, out_features=1)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return action_logits, value_logit

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            action_logits, value_logit = self.forward(board)
            action_probs = F.softmax(action_logits, dim=1)
            value = torch.tanh(value_logit)

        return action_probs.data.cpu().numpy()[0], value.data.cpu().item()
