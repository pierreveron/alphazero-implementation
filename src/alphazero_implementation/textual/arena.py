import asyncio
from concurrent.futures import Executor

from simulator.textual.connect import ConnectBoard
from textual.app import App, ComposeResult
from textual.containers import Grid

from alphazero_implementation.textual.agent import Agent


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

    def __init__(
        self, agent1: Agent, agent2: Agent, executor: Executor, num_games: int
    ) -> None:
        self.agent1 = agent1
        self.agent2 = agent2
        self.executor = executor
        self.boards: list[ConnectBoard] = []
        self.num_games = num_games
        super().__init__()

    def compose(self) -> ComposeResult:
        with Grid():
            for _ in range(self.num_games):
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
            action = await loop.run_in_executor(
                self.executor, current_agent.predict_best_action, state
            )
            state = action.sample_next_state()
            board.state = state
        winner = state.reward.argmax()  # type: ignore[attr-defined]
        board.styles.border = ("round", "green" if winner == 0 else "red")
