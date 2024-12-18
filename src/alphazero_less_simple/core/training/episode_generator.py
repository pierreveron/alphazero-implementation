import numpy as np

from alphazero_simple.base_game import BaseGame
from alphazero_simple.base_model import BaseModel
from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.monte_carlo_tree_search import MCTS

from .episode import Episode, Sample


class EpisodeGenerator:
    def __init__(
        self,
        *,
        model: BaseModel,
        game: BaseGame,
        config: AlphaZeroConfig,
    ):
        self.mcts = MCTS(
            model=model,
            game=game,
            config=config,
        )
        self.model = model
        self.game = game
        self.config = config

    # def update_inference_model(self, model):
    #     """Update the inference model with the latest weights from the training model"""
    #     self.mcts.update_inference_model(model)

    def generate_episodes(self) -> list[Episode]:
        episodes = []
        for _ in range(self.config.num_episodes):
            episodes.append(self.execute_episode())
        return episodes

    def execute_episode(self) -> Episode:
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)

            self.mcts = MCTS(self.game, self.model, self.config)
            root = self.mcts.run(canonical_board, to_play=1)

            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
            action_probs = action_probs / np.sum(action_probs)

            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temperature=1)
            state, current_player = self.game.get_next_state(
                state, current_player, action
            )
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                episode = Episode()
                for (
                    hist_state,
                    hist_current_player,
                    hist_action_probs,
                ) in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    episode.add_sample(
                        Sample(
                            state=hist_state,
                            policy=hist_action_probs,
                            value=reward
                            * ((-1) ** (hist_current_player != current_player)),
                        )
                    )

                return episode
