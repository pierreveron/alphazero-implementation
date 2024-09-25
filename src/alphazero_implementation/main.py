from simulator.game.connect import Config  # type: ignore[import]

from alphazero_implementation.alphazero import AlphaZero
from alphazero_implementation.models.neural_network import NeuralNetwork

config = Config(6, 7, 4)

state = config.sample_initial_state()


def train():
    config = Config(6, 7, 4)
    state = config.sample_initial_state()

    model = NeuralNetwork((-1, 3, 6, 7), 7)
    alphazero = AlphaZero(state, model)
    alphazero.run(num_iterations=50)

    # Save model after training
    model.save("trained_model.pt")


def play():
    """Play a game against the trained model."""
    return NotImplementedError("Not implemented yet")


def main():
    train()


if __name__ == "__main__":
    main()
