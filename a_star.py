'''
This file implements the A* search algorithm.
'''

from typing import Any
from collections.abc import Callable, Iterable
import itertools

from heapdict import heapdict
from tqdm import tqdm


class PriorityQueue:
    '''Implementation of a priority queue that uses HeapDict for efficiently maintinaing a minimum.
    Ties are returned in LIFO order to make A* behave more like depth-first search on paths with the
    same score.
    '''

    def __init__(self):
        self._heap = heapdict()
        self._counter = itertools.count()

    def push(self, item: Any, score: float):
        self._heap[item] = (score, -next(self._counter))  # LIFO

    def pop(self) -> Any:
        return self._heap.popitem()[0]

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return len(self._heap) > 0


def reconstruct_path(came_from: dict[Any, Any], current: Any) -> list[Any]:
    '''Reconstruct a path from the `came_from` dictionary returned by the A* algorithm.'''

    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def a_star(
    start_state: Any,
    successor_fn: Callable[[Any], Iterable[Any]],
    score_fn: Callable[[Any], float],
    goal_fn: Callable[[Any], bool],
) -> list[Any] | None:
    '''The A* search algorithm.

    Parameters
    ----------
    start_state:
        The state at which to begin the search.
    successor_fn : Callable[[Any], Iterable[Any]]
        A function returning successors of the current state.
    score_fn : Callable[[Any], float]
        A function that takes a state and returns its score. A smaller score indicates a better state.
    goal_fn : Callable[[Any], bool]
        A function that check whether or not a state is the goal.
    '''

    came_from = {}

    g_score = {}
    g_score[start_state] = 0

    f_score = {}
    f_score[start_state] = score_fn(start_state)

    priority_queue = PriorityQueue()
    priority_queue.push(start_state, f_score[start_state])

    visited = set()

    with tqdm(desc='A* searching') as pbar:
        while priority_queue:
            current = priority_queue.pop()

            if current in visited:
                continue
            visited.add(current)

            if goal_fn(current):
                return reconstruct_path(came_from, current)

            for neighbour in successor_fn(current):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    f_score[neighbour] = tentative_g_score + score_fn(neighbour)
                    priority_queue.push(neighbour, f_score[neighbour])

            pbar.update(1)


def a_star_vectorized(
    start_state: Any,
    score_fn: Callable[[list[Any]], list[float]],
    goal_fn: Callable[[Any], bool],
    successor_fn: Callable[[Any], Iterable[Any]],
    batch_size: int,
):
    '''A vectorized version of the A* search algorithm.

    Parameters
    ----------
    start_state:
        The state at which to begin the search.
    successor_fn : Callable[[Any], Iterable[Any]]
        A function returning successors of the current state.
    score_fn : Callable[[list[Any]], list[float]]
        A function that takes a list of states and returns a list of scores. A smaller score indicates
        a better state.
    goal_fn : Callable[[Any], bool]
        A function that check whether or not a state is the goal.
    batch_size : int
        Evaluate the score_fn on this number of states.
    '''

    came_from = {}

    g_score = {}
    g_score[start_state] = 0

    f_score = {}
    f_score[start_state] = score_fn([start_state])[0]

    priority_queue = PriorityQueue()
    priority_queue.push(start_state, f_score[start_state])

    visited = set()

    pending_scores = {}

    def _flush_pending_scores():
        if not pending_scores:
            return

        states_to_score = list(pending_scores.keys())
        heuristic_scores = score_fn(states_to_score)

        for state, h_score in zip(states_to_score, heuristic_scores):
            tentative_g_score = pending_scores[state]
            f_score[state] = tentative_g_score + h_score
            priority_queue.push(state, f_score[state])

        pending_scores.clear()

    with tqdm(desc='A* searching') as pbar:
        while priority_queue:
            current = priority_queue.pop()

            if current in visited:
                continue
            visited.add(current)

            if goal_fn(current):
                # _flush_pending_scores()
                return reconstruct_path(came_from, current)

            for neighbour in successor_fn(current):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score

                    pending_scores[neighbour] = tentative_g_score

                    if len(pending_scores) >= batch_size:
                        _flush_pending_scores()

            if not priority_queue and pending_scores:
                _flush_pending_scores()

            pbar.update(1)
