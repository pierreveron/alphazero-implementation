from alphazero_less_simple.core.training import Trainer
from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.connect4_game import Connect4Game
from alphazero_simple.resnet import ResNet


def main(config: AlphaZeroConfig):
    game = Connect4Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = ResNet(board_size, action_size, 9, 128)

    trainer = Trainer(game, model, config)
    trainer.learn()


if __name__ == "__main__":
    config = AlphaZeroConfig(
        batch_size=64,
        num_iterations=20,
        num_simulations=600,
        num_episodes=5000,
        num_iters_for_train_history=1,
        epochs=10,
        mem_buffer_size=int(5e5),
        background_generation=False,
    )
    main(config)
