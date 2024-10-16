from simulator.game.connect import Config  # type: ignore[import]

from alphazero_implementation.alphazero.trainer import AlphaZeroTrainer
from alphazero_implementation.models.games.connect4.v1 import BasicNN


def train():
    config = Config(6, 7, 4)
    initial_state = config.sample_initial_state()

    model = BasicNN(
        height=config.height,
        width=config.width,
        max_actions=config.width,
        num_players=config.num_players,
    )

    trainer = AlphaZeroTrainer(model=model, mcgs_num_simulations=2, mcgs_batch_size=1)

    trainer.train(
        num_iterations=3,
        initial_state=initial_state,
        max_epochs=100,
    )


# def play():
#     config = Config(6, 7, 4)
#     initial_state = (
#         config.sample_initial_state()
#     )  # Assuming this creates an initial state for Connect Four

#     input_shape = (-1, 3, 6, 7)  # Example for Connect Four: 3 channels, 6x7 board
#     num_actions = 7  # Example for Connect Four: 7 possible column choices

#     # Load the trained model
#     nn = NeuralNetwork(input_shape, num_actions)
#     nn.load("trained_model.pt", input_shape, num_actions)

#     alpha_zero = AlphaZeroMCGS(neural_network=nn, num_simulations=800)

#     # Demonstrate using the trained model to play a game
#     print("Playing a game with the trained model...")
#     state = initial_state
#     while not state.has_ended:
#         action = alpha_zero.get_best_action(state)
#         print(f"Chosen action: {action}")
#         state = action.sample_next_state()
#         print(state)  # Assuming the State class has a string representation

#     print("Game ended.")
#     print(f"Final state: {state}")
#     print(f"Reward: {state.reward[0]}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    train()
