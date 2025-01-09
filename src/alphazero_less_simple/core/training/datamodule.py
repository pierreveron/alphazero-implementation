import copy
import glob
import json
import threading
import time
from collections import deque
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from alphazero_simple.base_model import BaseModel
from alphazero_simple.config import AlphaZeroConfig

from .episode import Sample
from .episode_generator import EpisodeGenerator


class EpisodeGeneratorThread(threading.Thread):
    def __init__(
        self, generator: EpisodeGenerator, buffer: deque[Sample], model: BaseModel
    ):
        super().__init__(daemon=True)
        self.generator = generator
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.model = model

    def run(self):
        device = self.model.device
        model = copy.deepcopy(self.model)
        model.to(device)
        model.eval()

        episodes = self.generator.generate_episodes(model)
        start_time = time.time()
        for episode in episodes:
            if self.stop_event.is_set():
                break
            with self.lock:
                self.buffer.extend(episode.samples)
        print(f"Generated new episodes in {time.time() - start_time:.2f} seconds")

    def stop(self):
        self.stop_event.set()


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        model: BaseModel,
        episode_generator: EpisodeGenerator,
        config: AlphaZeroConfig,
        shuffle: bool = True,
        num_workers: int = 7,
        save_dir: str | Path | None = None,
        initial_samples_dir: Path | None = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.episode_generator = episode_generator

        self.shuffle = shuffle
        self.num_workers = num_workers

        self.buffer: deque[Sample] = deque(maxlen=self.config.mem_buffer_size)

        # Load initial episodes if provided
        if initial_samples_dir is not None and initial_samples_dir.exists():
            sample_files = sorted(
                glob.glob(str(initial_samples_dir / "samples_iter*.json"))
            )
            for sample_file in sample_files:
                samples = self._load_samples(Path(sample_file))
                self.buffer.extend(samples)
            print(
                f"Loaded {len(self.buffer)} samples from {len(sample_files)} sample files"
            )

        self.episode_generator_thread = (
            EpisodeGeneratorThread(self.episode_generator, self.buffer, self.model)
            if config.background_generation
            else None
        )

        # Setup save directory
        self.save_every_n_iterations = self.config.num_iters_for_train_history
        self.save_dir = Path(save_dir) if save_dir else Path("episodes")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.current_iteration = 0

    def setup(self, stage: str):
        if stage == "fit" and self.episode_generator_thread:
            self.episode_generator_thread.start()

    def _save_samples(self):
        """Save current samples in buffer to disk."""
        save_path = self.save_dir / f"samples_iter{self.current_iteration}.json"

        # Convert samples to episodes for saving
        samples_data = [sample.to_dict() for sample in self.buffer]

        with open(save_path, "w") as f:
            json.dump(samples_data, f)

        print(f"Saved {len(self.buffer)} samples to {save_path}")

    def _load_samples(self, path: Path) -> list[Sample]:
        """Load samples from JSON file"""
        with open(path, "r") as f:
            samples_data = json.load(f)

        return [Sample.from_dict(sample_data) for sample_data in samples_data]

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        # Wait until we have new episodes
        start_time = time.time()

        if self.episode_generator_thread:
            while self.episode_generator_thread.is_alive():
                time.sleep(0.1)
            self.episode_generator_thread = EpisodeGeneratorThread(
                self.episode_generator, self.buffer, self.model
            )
            self.episode_generator_thread.start()
        else:
            new_episodes = self.episode_generator.generate_episodes(self.model)
            self.buffer.extend(
                [sample for episode in new_episodes for sample in episode.samples]
            )

        waited_time = time.time() - start_time

        print(
            f"Waited {waited_time:.2f} seconds for {self.config.num_episodes} new episodes. Buffer size: {len(self.buffer)}"
        )

        # Save the new episodes
        self.current_iteration += 1
        if self.current_iteration % self.config.num_iters_for_train_history == 0:
            self._save_samples()

        # Use all episodes in the buffer for training
        all_samples: list[Sample] = list(self.buffer)

        boards = [sample.state for sample in all_samples]
        policies = [sample.policy for sample in all_samples]
        values = [sample.value for sample in all_samples]

        dataset = self.model.format_dataset(boards, policies, values)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_buffer_size(self) -> int:
        return len(self.buffer)

    def teardown(self, stage: str):
        if stage == "fit" and self.episode_generator_thread:
            self.episode_generator_thread.stop()
            self.episode_generator_thread.join()
