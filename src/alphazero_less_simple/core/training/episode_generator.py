import copy
from typing import Generator

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
        game: BaseGame,
        config: AlphaZeroConfig,
    ):
        self.game = game
        self.config = config

    def generate_episodes(self, model: BaseModel) -> Generator[Episode, None, None]:
        model = copy.deepcopy(model)
        model.eval()

        mcts = MCTS(self.game, model, self.config.num_simulations)
        states = [self.game.get_init_board() for _ in range(self.config.num_episodes)]
        current_players = [1] * self.config.num_episodes
        train_examples_list = [[] for _ in range(self.config.num_episodes)]

        episode_count = 0
        while True:
            canonical_boards = [
                self.game.get_canonical_board(state, current_player)
                for state, current_player in zip(states, current_players)
            ]

            roots = mcts.run_batch(canonical_boards, current_players)

            for root, state, current_player, canonical_board, i in zip(
                roots,
                states,
                current_players,
                canonical_boards,
                range(self.config.num_episodes),
            ):
                action_probs = [0 for _ in range(self.game.get_action_size())]
                for k, v in root.children.items():
                    action_probs[k] = v.visit_count
                action_probs = action_probs / np.sum(action_probs)

                train_examples_list[i].append(
                    (canonical_board, current_player, action_probs)
                )

                action = root.select_action(temperature=1)
                state, current_player = self.game.get_next_state(  # noqa: PLW2901
                    state, current_player, action
                )
                states[i], current_players[i] = state, current_player
                reward = self.game.get_reward_for_player(state, current_player)

                if reward is not None:
                    episode = Episode()
                    for (
                        hist_state,
                        hist_current_player,
                        hist_action_probs,
                    ) in train_examples_list[i]:
                        # [Board, currentPlayer, actionProbabilities, Reward]
                        episode.add_sample(
                            Sample(
                                state=hist_state,
                                policy=hist_action_probs,
                                value=reward
                                * ((-1) ** (hist_current_player != current_player)),
                            )
                        )

                    yield episode

                    states[i] = self.game.get_init_board()
                    current_players[i] = 1
                    train_examples_list[i] = []

                    episode_count += 1

                    if episode_count >= self.config.num_episodes:
                        return
