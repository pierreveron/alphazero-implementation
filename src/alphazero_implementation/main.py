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

    trainer = AlphaZeroTrainer(
        model=model, mcgs_num_simulations=100, games_per_iteration=50
    )

    trainer.train(
        iterations=100,
        initial_state=initial_state,
        num_playouts=10,
    )


# def play():
#     config = Config(6, 7, 4)
#     initial_state = config.sample_initial_state()

#     model = BasicNN(
#         height=config.height,
#         width=config.width,
#         max_actions=config.width,
#         num_players=config.num_players,
#     )

#     app = ExampleApp()
#     app.run()

# model.load("trained_model.pt", input_shape, num_actions)

# Demonstrate using the trained model to play a game
# print("Playing a game with the trained model...")
# state = initial_state
# while not state.has_ended:
#     action = model.get_best_action(state)
#     print(f"Chosen action: {action}")
#     state = action.sample_next_state()
#     print(state)  # Assuming the State class has a string representation

# print("Game ended.")
# print(f"Final state: {state}")
# print(f"Reward: {state.reward[0]}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    # play()
    train()
