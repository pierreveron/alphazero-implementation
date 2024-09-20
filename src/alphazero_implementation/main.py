from alphazero_implementation.alphazero import AlphaZero
from alphazero_implementation.games.connect4 import Connect4
from alphazero_implementation.models.neural_network import NeuralNetwork


def train():
    game = Connect4()
    model = NeuralNetwork(game.input_shape, 7)
    alphazero = AlphaZero(game, model)
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
