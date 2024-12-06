import json
import threading
import time
from collections import deque
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from alphazero_implementation.alphazero.types import Episode, Sample
from alphazero_implementation.models.model import Model
from alphazero_implementation.search.mcts import AlphaZeroEpisodeGenerator


class EpisodeGeneratorThread(threading.Thread):
    def __init__(self, agent: AlphaZeroEpisodeGenerator, buffer: deque[Episode]):
        super().__init__(daemon=True)
        self.agent = agent
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        episodes = self.agent.generate_episodes()
        for episode in episodes:
            if self.stop_event.is_set():
                break
            with self.lock:
                self.buffer.append(episode)

    def stop(self):
        self.stop_event.set()


class AlphaZeroDataModule(L.LightningDataModule):
    def __init__(
        self,
        model: Model,
        episode_generator: AlphaZeroEpisodeGenerator,
        buffer_size: int,
        save_every_n_iterations: int,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        persistent_workers: bool = True,
        save_dir: str | Path | None = None,
    ):
        super().__init__()
        self.model = model
        self.episode_generator = episode_generator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.buffer_size = buffer_size
        self.buffer: deque[Episode] = deque(maxlen=buffer_size)
        self.episode_generator_thread = EpisodeGeneratorThread(
            self.episode_generator, self.buffer
        )
        # Setup save directory
        self.save_every_n_iterations = save_every_n_iterations
        self.save_dir = Path(save_dir) if save_dir else Path("episodes")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.current_iteration = 0

    def setup(self, stage: str):
        if stage == "fit":
            self.episode_generator_thread.start()

    def _save_episodes(self):
        """Save current episodes in buffer to disk."""
        save_path = self.save_dir / f"episodes_iter{self.current_iteration}.json"

        episodes_data = [episode.to_dict() for episode in self.buffer]

        with open(save_path, "w") as f:
            json.dump(episodes_data, f)

        print(f"Saved {len(self.buffer)} episodes to {save_path}")

    def _load_episodes(self, path: Path) -> list[Episode]:
        """Load episodes from JSON file"""
        with open(path, "r") as f:
            episodes_data = json.load(f)

        return [Episode.from_dict(episode_data) for episode_data in episodes_data]

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        # Wait until we have new episodes
        start_time = time.time()
        while self.episode_generator_thread.is_alive():
            time.sleep(0.1)

        self.episode_generator_thread = EpisodeGeneratorThread(
            self.episode_generator, self.buffer
        )
        self.episode_generator.update_inference_model(self.model)
        self.episode_generator_thread.start()

        waited_time = time.time() - start_time

        print(
            f"Got {self.episode_generator.num_episodes} new episodes in {waited_time:.2f} seconds"
        )

        # Save the new episodes
        self.current_iteration += 1
        if self.current_iteration % self.save_every_n_iterations == 0:
            self._save_episodes()

        all_samples: list[Sample] = []
        for episode in self.buffer:
            all_samples.extend(episode.samples)

        states = [sample.state for sample in all_samples]
        policies = [sample.policy for sample in all_samples]
        values = [sample.value for sample in all_samples]

        dataset = self.model.format_dataset(states, policies, values)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def get_buffer_size(self) -> int:
        return len(self.buffer)

    def teardown(self, stage: str):
        if stage == "fit":
            self.episode_generator_thread.stop()
            self.episode_generator_thread.join()
