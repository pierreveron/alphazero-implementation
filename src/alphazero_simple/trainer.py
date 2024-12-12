import os
from random import shuffle

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from .config import AlphaZeroConfig
from .connect4_game import Connect4Game
from .connect4_model import Connect4Model
from .lightning_data import AlphaZeroDataModule
from .lightning_model import AlphaZeroLitModule
from .monte_carlo_tree_search import MCTS


class Trainer:
    def __init__(
        self,
        game: Connect4Game,
        model: Connect4Model,
        device: torch.device,
        config: AlphaZeroConfig,
    ):
        self.game = game
        self.model = model
        self.device = device
        self.config = config
        self.mcts = MCTS(self.game, self.model, self.config)

    def execute_episode(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)

            self.mcts = MCTS(self.game, self.model, self.config)
            root = self.mcts.run(self.model, canonical_board, to_play=1)

            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temperature=0)
            state, current_player = self.game.get_next_state(
                state, current_player, action
            )
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for (
                    hist_state,
                    hist_current_player,
                    hist_action_probs,
                ) in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append(
                        (
                            hist_state,
                            hist_action_probs,
                            reward * ((-1) ** (hist_current_player != current_player)),
                        )
                    )

                return ret

    def learn(self):
        for i in range(1, self.config.num_iterations + 1):
            print("{}/{}".format(i, self.config.num_iterations))

            train_examples = []

            for eps in range(self.config.num_episodes):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.config.checkpoint_path
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        # Create Lightning components
        lit_model = AlphaZeroLitModule(self.model)
        data_module = AlphaZeroDataModule(examples, batch_size=self.config.batch_size)

        logger = TensorBoardLogger(save_dir="lightning_logs", name="alphazero_simple")

        # Create Lightning trainer
        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            enable_progress_bar=True,
        )

        # Train the model
        trainer.fit(lit_model, data_module)

        # Update the original model with trained weights
        self.model.load_state_dict(lit_model.model.state_dict())

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
            },
            filepath,
        )
