import torch

from alphazero_less_simple.core.training import Trainer
from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.connect4_game import Connect4Game
from alphazero_simple.connect4_model import Connect4Model

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = Connect4Model(board_size, action_size, device)

config = AlphaZeroConfig(
    batch_size=32,
    num_iterations=200,
    num_simulations=100,
    num_episodes=100,
    num_iters_for_train_history=25,
    epochs=10,
)

trainer = Trainer(game, model, config)
trainer.learn()
