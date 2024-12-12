from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    batch_size: int  # Number of examples per batch
    num_iterations: int  # Total number of training iterations
    num_simulations: (
        int  # Total number of MCTS simulations to run when deciding on a move to play
    )
    num_episodes: int  # Number of full games (episodes) to run during each iteration
    num_iters_for_train_history: (
        int  # Number of iterations to store for training history
    )
    epochs: int  # Number of epochs of training per iteration
    checkpoint_path: str  # Location to save latest set of weights
