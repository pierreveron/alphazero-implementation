import threading
import time
from collections import deque

import lightning as L
import torch
from torch.utils.data import DataLoader

from alphazero_implementation.alphazero.types import Sample
from alphazero_implementation.mcts.agent import MCTSAgent
from alphazero_implementation.models.model import Model


class EpisodeGenerator(threading.Thread):
    def __init__(self, agent: MCTSAgent, buffer: deque[Sample]):
        super().__init__(daemon=True)
        self.agent = agent
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.samples_added = 0

    def get_samples_added(self) -> int:
        with self.lock:
            return self.samples_added

    def run(self):
        while not self.stop_event.is_set():
            episodes = self.agent.generate_episodes()
            for episode in episodes:
                samples = episode.samples
                # print(f"Adding {len(episode.samples)} samples to the buffer")
                with self.lock:
                    self.buffer.extend(samples)
                    self.samples_added += len(samples)

    def stop(self):
        self.stop_event.set()


class AlphaZeroDataModule(L.LightningDataModule):
    def __init__(
        self,
        model: Model,
        agent: MCTSAgent,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        persistent_workers: bool = True,
        buffer_size: int = 50000,
        min_new_data: int = 100,
    ):
        super().__init__()
        self.model = model
        self.agent = agent
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.buffer_size = buffer_size
        self.buffer: deque[Sample] = deque(maxlen=buffer_size)
        self.episode_generator = EpisodeGenerator(self.agent, self.buffer)
        self.min_new_data = min_new_data
        self.last_samples_added = 0

    def setup(self, stage: str):
        if stage == "fit":
            self.episode_generator.start()

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        # print("Getting new training data")
        # Wait until we have min_new_data new samples
        start_time = time.time()
        while (
            self.episode_generator.get_samples_added() - self.last_samples_added
        ) < self.min_new_data:
            # print(
            #     f"Waiting for {self.min_new_data} new samples, {self.episode_generator.get_samples_added() - self.last_samples_added} already added in {time.time() - start_time:.2f} seconds"
            # )
            time.sleep(0.1)  # Sleep for a short time to avoid busy waiting

        waited_time = time.time() - start_time
        samples_added = self.episode_generator.get_samples_added()
        new_samples = samples_added - self.last_samples_added

        print(f"Got {new_samples} new samples in {waited_time:.2f} seconds")

        self.last_samples_added = samples_added

        states = [sample.state for sample in self.buffer]
        policies = [sample.policy for sample in self.buffer]
        values = [sample.value for sample in self.buffer]

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
            self.episode_generator.stop()
            self.episode_generator.join()
