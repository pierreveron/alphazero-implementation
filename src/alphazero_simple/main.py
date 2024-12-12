import torch

from alphazero_simple.connect4_game import Connect4Game
from alphazero_simple.connect4_model import Connect4Model
from alphazero_simple.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    "batch_size": 64,
    "numIters": 500,  # Total number of training iterations
    "num_simulations": 25,  # Total number of MCTS simulations to run when deciding on a move to play
    "numEps": 100,  # Number of full games (episodes) to run during each iteration
    "numItersForTrainExamplesHistory": 20,
    "epochs": 2,  # Number of epochs of training per iteration
    "checkpoint_path": "latest.pth",  # location to save latest set of weights
}

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4Model(board_size, action_size, device)

trainer = Trainer(game, model, device, args)
trainer.learn()
