import os
import time
from collections import deque
from random import shuffle

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter

from .base_game import BaseGame
from .base_model import BaseModel
from .config import AlphaZeroConfig
from .monte_carlo_tree_search import MCTS


class Trainer:
    def __init__(
        self,
        game: BaseGame,
        model: BaseModel,
        device: torch.device,
        config: AlphaZeroConfig,
    ):
        self.game = game
        self.model = model
        self.device = device
        self.config = config
        self.mcts = MCTS(self.game, self.model, self.config.num_simulations)
        self.writer = SummaryWriter()
        self.global_step = 0
        self.train_examples = deque(maxlen=self.config.num_iters_for_train_history)

    def execute_episode(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)

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

            iteration_examples = []
            start_time = time.time()

            for eps in range(self.config.num_episodes):
                episode_examples = self.execute_episode()
                iteration_examples.extend(episode_examples)

            waited_time = time.time() - start_time
            print(
                f"Got {self.config.num_episodes} new episodes in {waited_time:.2f} seconds"
            )

            # Store all examples from this iteration as one entry
            self.train_examples.append(iteration_examples)

            # Flatten all stored iterations into a single list for training
            all_examples = []
            for iteration_examples in self.train_examples:
                all_examples.extend(iteration_examples)

            shuffle(all_examples)
            self.train(all_examples)
            filename = self.config.checkpoint_path
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)  # type: ignore[no-untyped-call]
        pi_losses = []
        v_losses = []

        for epoch in range(self.config.epochs):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.config.batch_size):
                sample_ids = np.random.randint(
                    len(examples), size=self.config.batch_size
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous().to(self.device)
                target_pis = target_pis.contiguous().to(self.device)
                target_vs = target_vs.contiguous().to(self.device)

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Log exactly the same metrics as the Lightning model
                self.writer.add_scalar(
                    "train_loss", float(total_loss), self.global_step
                )
                self.writer.add_scalar("policy_loss", float(l_pi), self.global_step)
                self.writer.add_scalar("value_loss", float(l_v), self.global_step)

                batch_idx += 1
                self.global_step += 1

        print("Policy Loss", np.mean(pi_losses))
        print("Value Loss", np.mean(v_losses))
        print("Examples:")
        print(out_pi[0].detach())
        print(target_pis[0])
        print()

    def loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

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

    def __del__(self):
        self.writer.close()
