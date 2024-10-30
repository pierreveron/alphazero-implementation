import asyncio
import random
from concurrent.futures import Executor, ThreadPoolExecutor

from simulator.game.connect import Action, Config  # type: ignore[attr-defined]
from simulator.textual.connect import ConnectBoard
from textual.app import App, ComposeResult
from textual.containers import Grid

from alphazero_implementation.models.games.connect4.v1 import BasicNN
from alphazero_implementation.textual.agent import Agent, AlphaZeroAgent, RandomAgent


class ArenaApp(App[None]):
    """Many games played by AI."""

    DEFAULT_CSS = """

    Grid {
        grid-size: 4;
    }

    BounceBoard {
        border: round white;
    }

    """

    def __init__(self, agent1: Agent, agent2: Agent, executor: Executor) -> None:
        self.agent1 = agent1
        self.agent2 = agent2
        self.executor = executor
        self.boards: list[ConnectBoard] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        with Grid():
            for _ in range(8):
                board = ConnectBoard(disabled=True)
                _ = asyncio.create_task(self._handle_board(board))  # noqa: RUF006
                yield board

    async def _handle_board(self, board: ConnectBoard) -> None:
        loop = asyncio.get_running_loop()
        # while True:
        state = ConnectBoard.DEFAULT_CONFIG.sample_initial_state()
        board.state = state
        board.styles.border = ("round", "white")
        while not state.has_ended:
            current_agent = self.agent1 if state.player == 0 else self.agent2
            policy = await loop.run_in_executor(
                self.executor, current_agent.predict, state
            )
            actions: list[Action] = list(policy.keys())
            weights: list[float] = list(policy.values())
            [action] = random.choices(actions, weights)
            state = action.sample_next_state()
            board.state = state
        winner = state.reward.argmax()  # type: ignore[attr-defined]
        board.styles.border = ("round", "green" if winner == 0 else "red")


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        config = Config(6, 7, 4)

        path = "/Users/pveron/Code/alphazero-implementation/lightning_logs/alphazero/run_029_iter10_sims100_batch50/checkpoints/epoch=9-step=304.ckpt"

        model = BasicNN.load_from_checkpoint(  # type: ignore[arg-type]
            path,
            height=config.height,
            width=config.width,
            max_actions=config.width,
            num_players=config.num_players,
        )

        model.eval()

        agent1 = AlphaZeroAgent(model)
        agent2 = RandomAgent()
        app = ArenaApp(agent1, agent2, executor)
        app.run()
