import asyncio
import random
from concurrent.futures import Executor, ThreadPoolExecutor

from simulator.game.connect import Action, Config, State  # type: ignore[attr-defined]
from simulator.textual.connect import ConnectBoard
from textual.app import App, ComposeResult

from alphazero_implementation.models.games.connect4.v1 import BasicNN
from alphazero_implementation.textual.agent import Agent, AlphaZeroAgent


class AgentApp(App[None]):
    """Play against the agent."""

    def __init__(self, agent: Agent, executor: Executor) -> None:
        self.agent = agent
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
            policy = await loop.run_in_executor(
                self.executor, self.agent.predict, state
            )
            actions: list[Action] = list(policy.keys())
            weights: list[float] = list(policy.values())
            [action] = random.choices(actions, weights)
            state = action.sample_next_state()
        board.state = state
        board.disabled = False
        board.focus()


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

        agent = AlphaZeroAgent(model)
        app = AgentApp(agent, executor)
        app.run()
