import time
from functools import wraps
from typing import Generator

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from alphazero_simple.base_game import BaseGame
from alphazero_simple.base_model import BaseModel
from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.monte_carlo_tree_search import MCTS

from .episode import Episode, Sample


def with_progress_bar(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        generator = func(self, *args, **kwargs)
        start_time = time.time()
        progress_bar = tqdm(
            total=self.config.num_episodes, desc="Generating episodes", unit="episode"
        )

        try:
            for item in generator:
                yield item
                progress_bar.update(1)

                # Update progress bar description with elapsed time
                elapsed = time.time() - start_time
                progress_bar.set_description(
                    f"Generating episodes (elapsed: {elapsed:.1f}s)"
                )
        finally:
            progress_bar.close()

    return wrapper


class EpisodeGenerator:
    def __init__(
        self,
        *,
        game: BaseGame,
        config: AlphaZeroConfig,
    ):
        self.game = game
        self.config = config

    @with_progress_bar
    def generate_episodes(self, model: BaseModel) -> Generator[Episode, None, None]:
        # # Store original device
        original_device = next(model.parameters()).device
        # Move to CPU if needed
        if original_device.type != "cpu":
            model = model.to("cpu")
        model.share_memory()

        num_processes = mp.cpu_count()
        episodes_per_process = self.config.num_episodes // num_processes
        remaining_episodes = self.config.num_episodes % num_processes

        process_episodes = [episodes_per_process] * num_processes
        process_episodes[0] += (
            remaining_episodes  # Add remaining episodes to first process
        )

        # Create a queue to receive episodes
        queue = mp.Queue()

        # Use Process instead of Pool for more control
        processes: list[mp.Process] = []
        for n_episodes in process_episodes:
            p = mp.Process(
                target=self._generate_batch_episodes, args=(model, n_episodes, queue)
            )
            p.start()
            processes.append(p)

        # Track completed episodes
        completed_episodes = 0
        while completed_episodes < self.config.num_episodes:
            episode = queue.get()
            completed_episodes += 1
            yield episode

        # Clean up processes
        for p in processes:
            p.terminate()  # Force terminate any hanging processes
            p.join()

        model.to(original_device)

    def _generate_batch_episodes(
        self,
        model: BaseModel,
        num_episodes: int,
        queue: mp.Queue,
    ) -> None:
        # Move model back to MPS in each worker process
        if torch.backends.mps.is_available():
            model = model.to("mps")

        mcts = MCTS(self.game, model, self.config.num_simulations)
        states = [self.game.get_init_board() for _ in range(num_episodes)]
        current_players = [1] * num_episodes
        train_examples_list = [[] for _ in range(num_episodes)]

        episode_count = 0

        while episode_count < num_episodes:
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
                range(num_episodes),
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
                        episode.add_sample(
                            Sample(
                                state=hist_state,
                                policy=hist_action_probs,
                                value=reward
                                * ((-1) ** (hist_current_player != current_player)),
                            )
                        )
                    queue.put(episode)

                    states[i] = self.game.get_init_board()
                    current_players[i] = 1
                    train_examples_list[i] = []
