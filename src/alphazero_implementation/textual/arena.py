import asyncio
from concurrent.futures import Executor

from simulator.textual.connect import ConnectBoard
from textual.app import App, ComposeResult
from textual.containers import Grid

from alphazero_implementation.textual.player import Player


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
        self, player1: Player, player2: Player, executor: Executor, num_games: int
    ) -> None:
        self.player1 = player1
        self.player2 = player2
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
            current_player = self.player1 if state.player == 0 else self.player2
            action = await loop.run_in_executor(
                self.executor, current_player.play, state
            )
            state = action.sample_next_state()
            board.state = state
        winner = state.reward.argmax()  # type: ignore[attr-defined]
        board.styles.border = ("round", "green" if winner == 0 else "red")
