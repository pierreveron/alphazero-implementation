import os
from random import shuffle
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter

from .connect4_game import Connect4Game
from .connect4_model import Connect4Model
from .monte_carlo_tree_search import MCTS


class Trainer:
    def __init__(
        self,
        game: Connect4Game,
        model: Connect4Model,
        device: torch.device,
        args: dict[str, Any],
    ):
        self.game = game
        self.model = model
        self.device = device
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)
        self.writer = SummaryWriter(
            os.path.join("runs", args.get("exp_name", "default_run"))
        )

    def exceute_episode(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)

            self.mcts = MCTS(self.game, self.model, self.args)
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
        for i in range(1, self.args["numIters"] + 1):
            print("{}/{}".format(i, self.args["numIters"]))

            train_examples = []

            for eps in range(self.args["numEps"]):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args["checkpoint_path"]
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args["epochs"]):
            self.model.train()
            epoch_pi_losses = []
            epoch_v_losses = []
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args["batch_size"]):
                sample_ids = np.random.randint(
                    len(examples), size=self.args["batch_size"]
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
                epoch_pi_losses.append(float(l_pi))
                epoch_v_losses.append(float(l_v))

                # Log batch losses
                global_step = (
                    epoch * int(len(examples) / self.args["batch_size"]) + batch_idx
                )
                self.writer.add_scalar(
                    "Loss/train/policy_loss", float(l_pi), global_step
                )
                self.writer.add_scalar("Loss/train/value_loss", float(l_v), global_step)
                self.writer.add_scalar(
                    "Loss/train/total_loss", float(total_loss), global_step
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            # Log epoch average losses
            epoch_pi_loss = np.mean(epoch_pi_losses)
            epoch_v_loss = np.mean(epoch_v_losses)
            self.writer.add_scalar("Loss/epoch/policy_loss", epoch_pi_loss, epoch)
            self.writer.add_scalar("Loss/epoch/value_loss", epoch_v_loss, epoch)

            # Log example predictions vs targets
            if epoch % 10 == 0:  # Log every 10 epochs to avoid too much data
                self.writer.add_histogram(
                    "Predictions/policy", out_pi[0].detach(), epoch
                )
                self.writer.add_histogram("Targets/policy", target_pis[0], epoch)
                self.writer.add_histogram("Predictions/value", out_v.detach(), epoch)
                self.writer.add_histogram("Targets/value", target_vs, epoch)

            print()
            print("Policy Loss", epoch_pi_loss)
            print("Value Loss", epoch_v_loss)
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
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
        # Close the tensorboard writer when the trainer is destroyed
        self.writer.close()
