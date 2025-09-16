from pathlib import Path
import pandas as pd

from cube_model import RubiksCube
from solver import RubiksCubeSolver
from agent import MLPAgent, EmbeddingMLPAgent


def solve_santa():
    santa_path = Path.cwd() / 'santa-2023' / 'puzzles.csv'
    df = pd.read_csv(santa_path)

    df = df[df.solution_state.str.startswith('A;A;A;A;A;A;A;A;A;B')]

    cubes = []
    for state in df.initial_state.values:
        cubes.append(RubiksCube.from_santa(state))

    agent_dir = Path.cwd() / 'agent_cache'
    agents = [
        # 'MLP_epoch2000_5660508080.pkl',
        # 'MLP_epoch2000_min_5889334240.pkl',
        'EmbeddingMLP_epoch2000_5368093568.pkl',
        # 'EmbeddingMLP_epoch2000_min_6085569248.pkl',
        # 'EmbeddingMLP_epoch2000_k20_4740610480.pkl',
        # 'EmbeddingMLP_epoch2000_5l_5247859456.pkl',
        # 'EmbeddingMLP_epoch2000_4l_4769708848.pkl',
    ]
    agents = [agent_dir / agent for agent in agents]

    solver = RubiksCubeSolver(agents, beam_sizes=2**15, max_search_its=50)
    solutions = []
    for i, cube in enumerate(cubes):
        print('SOLVING CUBE', i + 1)
        print(cube)

        solution = solver.solve(cube)
        print('SOLUTION:', solution)

        if solution is not None:
            print('SOLUTION LENGTH:', len(solution))
            cube.move(solution)
            print(cube)

        solutions.append(solution)

    failures = sum(x for x in solutions if x is None)
    successes = len(solutions) - failures
    print('SUCCESSES', f'{successes}/{len(solutions)}')
    print('FAILURES', f'{failures}/{len(solutions)}')

    solutions = [x for x in solutions if x is not None]

    avg_len = sum(len(x) for x in solutions) / len(solutions)
    print('AVERAGE SOLUTION LENGTH', avg_len)


if __name__ == '__main__':
    solve_santa()
