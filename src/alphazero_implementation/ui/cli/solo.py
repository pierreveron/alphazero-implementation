import asyncio
from concurrent.futures import Executor

from simulator.game.connect import State  # type: ignore[attr-defined]
from simulator.textual.connect import ConnectBoard
from textual.app import App, ComposeResult

from .player import Player


class AgentApp(App[None]):
    """Play against the agent."""

    def __init__(self, player: Player, executor: Executor) -> None:
        self.player = player
        self.executor = executor
        super().__init__()

    def compose(self) -> ComposeResult:
        board = ConnectBoard()
        board.state = ConnectBoard.DEFAULT_CONFIG.sample_initial_state()
        board.disabled = board.state.player != 0
        yield board

    async def on_connect_board_reset(self, event: ConnectBoard.Reset) -> None:
        await self._play(
            event.board, ConnectBoard.DEFAULT_CONFIG.sample_initial_state()
        )

    async def on_connect_board_selected(self, event: ConnectBoard.Selected) -> None:
        await self._play(event.board, event.action.sample_next_state())

    async def _play(self, board: ConnectBoard, state: State) -> None:
        if state.player == 0:
            board.state = state
        else:
            board.state = state
            board.disabled = True
            _ = asyncio.create_task(self._play_agent(board, state))

    async def _play_agent(self, board: ConnectBoard, state: State) -> None:
        loop = asyncio.get_running_loop()
        while not state.has_ended and state.player != 0:
            action = await loop.run_in_executor(self.executor, self.player.play, state)
            state = action.sample_next_state()
        board.state = state
        board.disabled = False
        board.focus()
