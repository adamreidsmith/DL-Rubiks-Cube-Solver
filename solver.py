from pathlib import Path
from typing import Optional
from collections.abc import Sequence
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

from agent import Agent, MLPAgent, EmbeddingMLPAgent
from cube_model import RubiksCube


class RubiksCubeSolver:
    def __init__(
        self,
        agent_paths: str | Path | Sequence[str | Path],
        beam_sizes: Optional[int | Sequence[int]] = None,
        max_search_its: Optional[int | Sequence[int]] = None,
    ):
        if not isinstance(agent_paths, Sequence):
            agent_paths = [agent_paths]

        self.agents: list[Agent] = []
        for path in agent_paths:
            self.agents.append(Agent.from_path(path))

        for agent in self.agents:
            setattr(agent, 'verbose', False)

        if beam_sizes is None:
            self.beam_sizes = [agent.beam_size for agent in self.agents]
        else:
            self.set_beam_sizes(beam_sizes)

        if max_search_its is None:
            self.max_search_its = [agent.max_search_its for agent in self.agents]
        else:
            self.set_max_search_its(max_search_its)

    def set_beam_sizes(self, beam_sizes: int | Sequence[int]):
        if isinstance(beam_sizes, Sequence):
            if len(beam_sizes) != len(self.agents):
                raise ValueError('`beam_sizes` must be an integer or a sequence with the same length as `agent_paths`')
            self.beam_sizes = beam_sizes
        else:
            self.beam_sizes = [int(beam_sizes)] * len(self.agents)

        for agent, bs in zip(self.agents, self.beam_sizes):
            setattr(agent, 'beam_size', bs)

    def set_max_search_its(self, max_search_its: int | Sequence[int]):
        if isinstance(max_search_its, Sequence):
            if len(max_search_its) != len(self.agents):
                raise ValueError(
                    '`max_search_its` must be an integer or a sequence with the same length as `agent_paths`'
                )
            self.max_search_its = max_search_its
        else:
            self.max_search_its = [int(max_search_its)] * len(self.agents)

        for agent, msi in zip(self.agents, self.max_search_its):
            setattr(agent, 'max_search_its', msi)

    @staticmethod
    def _solver_helper(agent: Agent, start_state: RubiksCube) -> list[str] | None:
        return agent.search(start_state)

    def solve(self, start_state: RubiksCube, processes: int = 1) -> list[str] | None:
        if processes < 0:
            from os import cpu_count

            processes = cpu_count()

        if processes == 1:
            solutions = []
            for agent in tqdm(self.agents, desc='Solving cube'):
                solutions.append(agent.search(start_state))
        else:
            partial_solver = partial(self._solver_helper, start_state=start_state)
            with Pool(processes=processes) as pool:
                solutions = pool.map(partial_solver, self.agents)

        solutions = [sol for sol in solutions if sol is not None]
        if not solutions:
            return None

        best_solution = min(solutions, key=len)
        return best_solution


if __name__ == '__main__':
    agent_dir = Path('./agent_cache')
    agents = [
        'MLP_epoch2000_5660508080.pkl',
        'MLP_epoch2000_min_5889334240.pkl',
        'EmbeddingMLP_epoch2000_5368093568.pkl',
        'EmbeddingMLP_epoch2000_min_6085569248.pkl',
        'EmbeddingMLP_epoch2000_k20_4740610480.pkl',
        'EmbeddingMLP_epoch2000_5l_5247859456.pkl',
        'EmbeddingMLP_epoch2000_4l_4769708848.pkl',
    ]
    agents = [agent_dir / agent for agent in agents]

    solver = RubiksCubeSolver(agents, beam_sizes=2**11, max_search_its=50)

    cube = RubiksCube()
    moves = RubiksCube.get_random_moves(50)
    print('Scramble:', moves)

    cube.move(moves)
    print(cube)

    solution = solver.solve(cube)
    print('Solution:', solution)
    print('Solution length:', len(solution))

    cube.move(solution)
    print(cube)
