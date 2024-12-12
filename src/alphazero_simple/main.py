import torch

from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.connect4_game import Connect4Game
from alphazero_simple.connect4_model import Connect4Model
from alphazero_simple.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4Model(board_size, action_size, device)

config = AlphaZeroConfig(
    batch_size=64,
    num_iterations=500,
    num_simulations=25,
    num_episodes=100,
    num_iters_for_train_history=20,
    epochs=2,
    checkpoint_path="latest.pth",
)
trainer = Trainer(game, model, device, config)
trainer.learn()
