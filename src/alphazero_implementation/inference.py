from alphazero_implementation.games.connect4 import Connect4
from alphazero_implementation.games.state import Action, GameState
from alphazero_implementation.models.neural_network import NeuralNetwork


def load_model(model_path: str) -> NeuralNetwork:
    game = Connect4()
    model = NeuralNetwork(game.input_shape, 7)
    model.load(model_path, game.input_shape, 7)
    return model


def play_game(model: NeuralNetwork) -> None:
    game = Connect4()
    state = game.initial_state

    while not game.is_terminal(state):
        if game.current_player(state) == 1:
            # AI move
            action, _ = model.predict(state)
        else:
            # Human move
            action = get_human_move(state)

        state = game.next_state(state, action)
        game.display(state)

    winner = game.winner(state)
    print(
        f"Game over. Winner: {'Player 1' if winner == 1 else 'Player 2' if winner == -1 else 'Draw'}"
    )


def get_human_move(state: GameState) -> Action:
    # Implement logic to get human move input
    raise NotImplementedError("get_human_move not implemented")


def main() -> None:
    model = load_model("trained_model.pt")
    play_game(model)


if __name__ == "__main__":
    main()
