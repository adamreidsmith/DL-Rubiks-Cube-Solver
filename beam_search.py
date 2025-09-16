from typing import Any, Optional
from collections.abc import Callable, Iterable, Sequence
import heapq

from tqdm import tqdm


class Node:
    def __init__(self, data: Any, parent: Optional['Node'] = None):
        self.data = data
        self.parent = parent

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.data == other.data

    def __hash__(self) -> int:
        return hash(self.data)


def beam_search(
    start_state: Any,
    successor_fn: Callable[[Any], Iterable[Any]],
    score_fn: Callable[[Any | Sequence[Any]], float | Sequence[float]],
    goal_fn: Callable[[Any], bool],
    beam_size: int,
    max_its: int,
    is_batch_score_fn: bool,
    verbose: bool = True,
) -> list[Any] | None:
    '''Runs the beam search algorithm.

    Parameters
    ----------
    start_state : Any
        The state at which to begin the search.
    successor_fn : Callable[[Any], Iterable[Any]]
        A function returning successors of the current state.
    score_fn : Callable[[Any | Sequence[Any]], float | Sequence[float]]
        A function that takes a state and returns its score, or takes a sequence of states and
        returns a sequence of scores. A smaller score indicates a better state.
    goal_fn : Callable[[Any], bool]
        A function that check whether or not a state is the goal.
    beam_size : int,
        The max number of states to include at each step.
    max_its : int
        The maximum number of iterations to perform.
    is_batch_score_fn : bool
        Whether or not the score function can be called with a batch of inputs.
    verbose : bool
        Whether or not to display a progress bar.

    Returns
    -------
    path : list[Any] | None
        A list of states if a goal is found, otherwise None.
    '''

    beam = [Node(start_state)]
    target = None

    with tqdm(desc='Beam searching', disable=not verbose) as pbar:
        for _ in range(max_its):
            next_states = list(set(Node(succ, state) for state in beam for succ in successor_fn(state.data)))

            if not next_states:
                return

            if is_batch_score_fn:
                scored_states = dict(zip(next_states, score_fn([state.data for state in next_states])))
                beam = heapq.nsmallest(beam_size, next_states, key=scored_states.get)
            else:
                beam = heapq.nsmallest(beam_size, next_states, key=lambda state: score_fn(state.data))

            for state in beam:
                if goal_fn(state.data):
                    target = state
                    break
            if target is not None:
                break

            pbar.update(1)
        else:
            return

    path = []
    while target:
        path.append(target.data)
        target = target.parent
    path.reverse()
    return path
