import torch
from simulator.game.connect import Config, State  # type: ignore[import]
from torch import Tensor, nn

from alphazero_implementation.alphazero_mcgs import AlphaZeroMCGS
from alphazero_implementation.models.model import ActionPolicy, Value
from alphazero_implementation.models.neural_network import NeuralNetwork


def infer(model: nn.Module, X: Tensor) -> Tensor:
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        outputs = model(X)
    return outputs


def predict(model: nn.Module, states: list[State]) -> list[tuple[ActionPolicy, Value]]:
    X = torch.FloatTensor([state.to_input() for state in states])
    outputs = infer(model, X)
    return outputs


def train():
    config = Config(6, 7, 4)
    initial_state = config.sample_initial_state()

    # Initialize the neural network
    input_shape = (-1, 3, 6, 7)  # Example for Connect Four: 3 channels, 6x7 board
    num_actions = 7  # Example for Connect Four: 7 possible column choices
    model = NeuralNetwork(input_shape, num_actions)

    # Initialize AlphaZero with MCGS
    alpha_zero = AlphaZeroMCGS(neural_network=model, num_simulations=800)

    # Training parameters
    num_iterations = 10
    num_episodes = 100

    # Train AlphaZero
    print("Starting AlphaZero training...")
    alpha_zero.train(num_iterations, num_episodes, initial_state)
    print("Training completed.")

    # Optional: Save the model
    torch.save(model.state_dict(), "trained_model.pt")  # type: ignore[no-untyped-call]


def play():
    config = Config(6, 7, 4)
    initial_state = (
        config.sample_initial_state()
    )  # Assuming this creates an initial state for Connect Four

    input_shape = (-1, 3, 6, 7)  # Example for Connect Four: 3 channels, 6x7 board
    num_actions = 7  # Example for Connect Four: 7 possible column choices

    # Load the trained model
    nn = NeuralNetwork(input_shape, num_actions)
    nn.load("trained_model.pt", input_shape, num_actions)

    alpha_zero = AlphaZeroMCGS(neural_network=nn, num_simulations=800)

    # Demonstrate using the trained model to play a game
    print("Playing a game with the trained model...")
    state = initial_state
    while not state.has_ended:
        action = alpha_zero.get_best_action(state)
        print(f"Chosen action: {action}")
        state = action.sample_next_state()
        print(state)  # Assuming the State class has a string representation

    print("Game ended.")
    print(f"Final state: {state}")
    print(f"Reward: {state.reward[0]}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    play()
