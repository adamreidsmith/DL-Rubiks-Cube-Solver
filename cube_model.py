import random
from typing import Optional
from collections.abc import Sequence, Iterator, Iterable

from bit_utils import (
    left_circular_shift24,
    cycle_9_bit_subfields,
    cycle_6_and_3_bit_subfields,
    swap_9_bits_twice,
    swap_9_6_3_bits,
    swap_6_3_bits_twice,
    extract_3,
    cycle_3_bit_subfields,  ##
)


class RubiksCube:
    _MOVES = ('U', 'Up', 'U2', 'F', 'Fp', 'F2', 'R', 'Rp', 'R2', 'B', 'Bp', 'B2', 'L', 'Lp', 'L2', 'D', 'Dp', 'D2')

    # Solved state 15929103523346434384942211689761829866700800
    _SOLVED_STATE = int(
        '101_101_101_101_101_101_101_101'
        '100_100_100_100_100_100_100_100'
        '011_011_011_011_011_011_011_011'
        '010_010_010_010_010_010_010_010'
        '001_001_001_001_001_001_001_001'
        '000_000_000_000_000_000_000_000',
        2,
    )

    def __init__(self, state: Optional[int] = None):
        '''
        A Rubik's Cube.

        The orientation of the Rubik's Cube represented by this class is white facing forward, and
        red on top. The cube is represented by one 144-bit python integer. Each consecutive subset
        of 3 bits represents the colour of a particular facelet. For example, a face may be
        represented as follows,

            W G R
            G   B  =  001 010 000 100 101 011 001 010
            W Y O

        where the bit position for each face are

            28 24 20
            0     16
            4  8  12

        and the color codes are determined according to

            red    = 000
            white  = 001
            green  = 010
            yellow = 011
            blue   = 100
            orange = 101

        Bit positions for the whole cube:

                       21  18  15
                       0   U   12
                       3   6   9

        117 114 111    45  42  39    69  66  63    93  90  87
        96   L  108    24  F   36    48  R   60    72  B   84
        99  102 105    27  30  33    51  54  57    75  78  81

                       141 138 135
                       120  D  132
                       123 126 129
        '''

        self.state = self._SOLVED_STATE if state is None else state

    @classmethod
    def from_santa(cls, s: str) -> 'RubiksCube':
        '''
                   0  1  2
                   3  4  5
                   6  7  8

        36 37 38   9  10 11   18 19 20   27 28 29
        39 40 41   12 13 14   21 22 23   30 31 32
        42 43 44   15 16 17   24 25 26   33 34 35

                   45 46 47
                   48 49 50
                   51 52 53
        '''
        cm = {'A': '000', 'B': '001', 'C': '010', 'D': '011', 'E': '100', 'F': '101'}

        s = s.split(';')

        orientation = [s[4], s[13], s[22], s[31], s[40], s[49]]
        s = (
            f'{cm[s[45]]}{cm[s[46]]}{cm[s[47]]}{cm[s[50]]}{cm[s[53]]}{cm[s[52]]}{cm[s[51]]}{cm[s[48]]}'
            f'{cm[s[36]]}{cm[s[37]]}{cm[s[38]]}{cm[s[41]]}{cm[s[44]]}{cm[s[43]]}{cm[s[42]]}{cm[s[39]]}'
            f'{cm[s[27]]}{cm[s[28]]}{cm[s[29]]}{cm[s[32]]}{cm[s[35]]}{cm[s[34]]}{cm[s[33]]}{cm[s[30]]}'
            f'{cm[s[18]]}{cm[s[19]]}{cm[s[20]]}{cm[s[23]]}{cm[s[26]]}{cm[s[25]]}{cm[s[24]]}{cm[s[21]]}'
            f'{cm[s[9]]}{cm[s[10]]}{cm[s[11]]}{cm[s[14]]}{cm[s[17]]}{cm[s[16]]}{cm[s[15]]}{cm[s[12]]}'
            f'{cm[s[0]]}{cm[s[1]]}{cm[s[2]]}{cm[s[5]]}{cm[s[8]]}{cm[s[7]]}{cm[s[6]]}{cm[s[3]]}'
        )

        cube: 'RubiksCube' = cls(int(s, 2))

        def rotate_back(cube: 'RubiksCube', orient: list[str]) -> tuple['RubiksCube', list[str]]:
            cube.move_Lp()
            cube.move_R()
            cube.state = cycle_3_bit_subfields(cube.state, 18, 78, 138, 42)
            cube.state = cycle_3_bit_subfields(cube.state, 6, 90, 126, 30)

            orient = [orient[1], orient[5], orient[2], orient[0], orient[4], orient[3]]
            return cube, orient

        def rotate_right(cube: 'RubiksCube', orient: list[str]) -> tuple['RubiksCube', list[str]]:
            cube.move_Up()
            cube.move_D()
            cube.state = cycle_3_bit_subfields(cube.state, 24, 48, 72, 96)
            cube.state = cycle_3_bit_subfields(cube.state, 36, 60, 84, 108)

            orient = [orient[0], orient[4], orient[1], orient[2], orient[3], orient[5]]
            return cube, orient

        if orientation[5] == 'A':
            cube, orientation = rotate_back(cube, orientation)
            cube, orientation = rotate_back(cube, orientation)

        if orientation[0] != 'A':
            while orientation[1] != 'A':
                cube, orientation = rotate_right(cube, orientation)
            cube, orientation = rotate_back(cube, orientation)

        while orientation[1] != 'B':
            cube, orientation = rotate_right(cube, orientation)

        assert orientation == ['A', 'B', 'C', 'D', 'E', 'F']

        return cube

    def move(self, moves: str | Sequence[str]):
        '''Perform a move or sequence of moves.

        Parameters
        ----------
        moves : str | Sequence[str]
            The move or moves to perform.
                90 degree clockwise:
                    `U`, `F`, `R`, `B`, `L`, `D`
                90 degree anticlockwise:
                    `U'`, `F'`, `R'`, `B'`, `L'`, `D'`
                    OR
                    `Up`, `Fp`, `Rp`, `Bp`, `Lp`, `Dp`
                180 degree rotations:
                    `U2`, `F2`, `R2`, `B2`, `L2`, `D2`
        '''

        if isinstance(moves, str):
            moves = [moves]

        for move in moves:
            move_modifier = '' if len(move) == 1 else ('2' if move[1] == '2' else 'p')
            self._move(f'{move[0]}{move_modifier}')

    def _move(self, move: str):
        '''Perform a single move.

        Parameters
        ----------
        move : str
            A single move.
        '''

        getattr(self, f'move_{move}')()

    def move_U(self):
        '''Perform a `U` rotation.'''

        self.state = left_circular_shift24(self.state, 0, 18)
        self.state = cycle_9_bit_subfields(self.state, 39, 111, 87, 63)

    def move_Up(self):
        '''Perform a `U'` rotation.'''

        self.state = left_circular_shift24(self.state, 0, 6)
        self.state = cycle_9_bit_subfields(self.state, 39, 63, 87, 111)

    def move_U2(self):
        '''Perform a `U2` rotation.'''

        self.state = left_circular_shift24(self.state, 0, 12)
        self.state = swap_9_bits_twice(self.state, 39, 87, 63, 111)

    def move_F(self):
        '''Perform a `F` rotation.'''

        self.state = left_circular_shift24(self.state, 24, 18)
        self.state = cycle_6_and_3_bit_subfields(self.state, 6, 48, 138, 108, 3, 69, 135, 105)

    def move_Fp(self):
        '''Perform a `F'` rotation.'''

        self.state = left_circular_shift24(self.state, 24, 6)
        self.state = cycle_6_and_3_bit_subfields(self.state, 6, 108, 138, 48, 3, 105, 135, 69)

    def move_F2(self):
        '''Perform a `F2` rotation.'''

        self.state = left_circular_shift24(self.state, 24, 12)
        self.state = swap_9_6_3_bits(self.state, 3, 135, 48, 108, 69, 105)

    def move_R(self):
        '''Perform a `R` rotation.'''

        self.state = left_circular_shift24(self.state, 48, 18)
        self.state = cycle_6_and_3_bit_subfields(self.state, 72, 132, 36, 12, 93, 129, 33, 9)

    def move_Rp(self):
        '''Perform a `R'` rotation.'''

        self.state = left_circular_shift24(self.state, 48, 6)
        self.state = cycle_6_and_3_bit_subfields(self.state, 72, 12, 36, 132, 93, 9, 33, 129)

    def move_R2(self):
        '''Perform a `R2` rotation.'''

        self.state = left_circular_shift24(self.state, 48, 12)
        self.state = swap_9_6_3_bits(self.state, 129, 9, 36, 72, 33, 93)

    def move_B(self):
        '''Perform a `B` rotation.'''

        self.state = left_circular_shift24(self.state, 72, 18)
        self.state = cycle_6_and_3_bit_subfields(self.state, 96, 126, 60, 18, 117, 123, 57, 15)

    def move_Bp(self):
        '''Perform a `B'` rotation.'''

        self.state = left_circular_shift24(self.state, 72, 6)
        self.state = cycle_6_and_3_bit_subfields(self.state, 96, 18, 60, 126, 117, 15, 57, 123)

    def move_B2(self):
        '''Perform a `B2` rotation.'''

        self.state = left_circular_shift24(self.state, 72, 12)
        self.state = swap_9_6_3_bits(self.state, 123, 15, 60, 96, 57, 117)

    def move_L(self):
        '''Perform a `L` rotation.'''

        self.state = left_circular_shift24(self.state, 96, 18)
        self.state = cycle_6_and_3_bit_subfields(self.state, 0, 24, 120, 84, 21, 45, 141, 81)

    def move_Lp(self):
        '''Perform a `L'` rotation.'''

        self.state = left_circular_shift24(self.state, 96, 6)
        self.state = cycle_6_and_3_bit_subfields(self.state, 0, 84, 120, 24, 21, 81, 141, 45)

    def move_L2(self):
        '''Perform a `L2` rotation.'''

        self.state = left_circular_shift24(self.state, 96, 12)
        self.state = swap_6_3_bits_twice(self.state, 120, 0, 24, 84, 21, 141, 45, 81)

    def move_D(self):
        '''Perform a `D` rotation.'''

        self.state = left_circular_shift24(self.state, 120, 18)
        self.state = cycle_9_bit_subfields(self.state, 27, 51, 75, 99)

    def move_Dp(self):
        '''Perform a `D'` rotation.'''

        self.state = left_circular_shift24(self.state, 120, 6)
        self.state = cycle_9_bit_subfields(self.state, 27, 99, 75, 51)

    def move_D2(self):
        '''Perform a `D2` rotation.'''

        self.state = left_circular_shift24(self.state, 120, 12)
        self.state = swap_9_bits_twice(self.state, 27, 75, 99, 51)

    @classmethod
    def inverse(cls, move: str) -> str:
        '''Get the inverse of a move.

        Paramerters
        -----------
        move : str
            The move to invert.

        Returns
        -------
        str
            The inverse move.
        '''

        if len(move) == 1:
            return f'{move}p'
        if move[1] == '2':
            return move
        return move[0]

    @classmethod
    def get_random_move(cls) -> str:
        '''Get a single random move.

        Returns
        -------
        str
            A random move.
        '''

        return random.choice(cls._MOVES)

    @classmethod
    def get_random_moves(cls, n: int, enforce_noncommutative: bool = True) -> list[str]:
        '''Get `n` random moves.

        Parameters
        ----------
        n : int
            The number of random moves to return.
        enforce_noncommutative : bool
            Enforce that the random sequence of moves cannot be trivially collapsed to a shorter sequence.

        Returns
        -------
        list
            A list of moves with dtype '<U2'.
        '''

        if enforce_noncommutative:
            opposite_face = {'U': 'D', 'F': 'B', 'R': 'L', 'B': 'F', 'L': 'R', 'D': 'U'}
            moves = []
            while len(moves) < n:
                move = cls.get_random_move()
                if moves and moves[-1][0] == move[0]:
                    continue
                if len(moves) >= 2 and moves[-1][0] == opposite_face[move[0]] and moves[-2][0] == move[0]:
                    continue
                moves.append(move)
            return moves

        return random.choices(cls._MOVES, k=n)

    def scramble(self, n: int = 20) -> 'RubiksCube':
        '''Scramble the cube using a random sequence of `n` moves.

        Parameters
        ----------
        n : int
            The number of scramble moves to perform. Default is 20.

        Returns
        -------
        self
            The scrambled cube.
        '''

        scramble_moves = self.get_random_moves(n)
        for move in scramble_moves:
            getattr(self, f'move_{move}')()
        return self

    @classmethod
    def iter_moves(cls) -> Iterator[str]:
        '''Iterate over all possible Rubik's Cube moves.

        Yields
        ------
        str
            A move.
        '''

        return iter(cls._MOVES)

    def iter_neighbours(self) -> Iterator['RubiksCube']:
        '''Iterate over the neighbours of self.

        Yields
        ------
        RubiksCube
            A state neighbouring the current one.
        '''

        for move in self._MOVES:
            copy = self.__copy__()
            getattr(copy, f'move_{move}')()
            yield copy

    def to_tensor(self, dtype: Optional['torch.dtype'] = None, one_hot: bool = True) -> 'torch.Tensor':  # type: ignore
        '''Convert self to a PyTorch tensor.

        Parameters
        ----------
        dtype : torch.dtype
            The dtype for the tensor. If None, `torch.uint8` is used.
        one_hot : bool
            Whether or not to one-hot encode the tensor.

        Returns
        -------
        torch.Tensor
            A tensor representing the cube state.
        '''

        import torch

        if dtype is None:
            dtype = torch.uint8

        return self._to_tensor(self.state, dtype, one_hot)

    @classmethod
    def _to_tensor(cls, state: int | Sequence[int], dtype: 'torch.dtype', one_hot: bool) -> 'torch.Tensor':  # type: ignore
        '''Convert a state or sequence of states to a PyTorch tensor.

        Parameters
        ----------
        state : int | Sequence[int]
            The integer representing the cube state.
        dtype : torch.dtype
            The dtype for the tensor.
        one_hot : bool
            Whether or not to one-hot encode the tensor.

        Returns
        -------
        torch.Tensor
            A tensor representing the cube state(s).
        '''

        import torch
        import torch.nn.functional as F

        mask_3 = 0x7
        # Check for reusable iterable (more general than `isinstance(state, Sequence)`)
        if isinstance(state, Iterable) and not isinstance(state, Iterator):
            mask_48 = 0xFFFFFFFFFFFF

            lo = torch.tensor([s & mask_48 for s in state], dtype=torch.long).unsqueeze(1)
            mi = torch.tensor([(s >> 48) & mask_48 for s in state], dtype=torch.long).unsqueeze(1)
            hi = torch.tensor([(s >> 96) & mask_48 for s in state], dtype=torch.long).unsqueeze(1)

            shifts = torch.arange(0, 48, 3)
            lo = (lo >> shifts) & mask_3
            mi = (mi >> shifts) & mask_3
            hi = (hi >> shifts) & mask_3
            result = torch.cat((lo, mi, hi), dim=1)

            if one_hot:
                result = F.one_hot(result).reshape(len(state), -1)

            return result.to(dtype)

        tensor = torch.empty(48, dtype=torch.long)
        for i in range(48):
            tensor[i] = (state >> (3 * i)) & mask_3

        if one_hot:
            tensor = F.one_hot(tensor).reshape(-1)

        return tensor.to(dtype)

    @classmethod
    def _to_ndarray(cls, state: int | Iterable[int], dtype: 'np.dtype', one_hot: bool) -> 'np.ndarray':  # type: ignore
        '''Convert a state or sequence of states to a NumPy array.

        Parameters
        ----------
        state : int | Sequence[int]
            The integer representing the cube state.
        dtype : np.dtype
            The dtype for the array.
        one_hot : bool
            Whether or not to one-hot encode the tensor.

        Returns
        -------
        np.ndarray
            An array representing the cube state(s).
        '''

        import numpy as np

        mask_3 = 0x7
        if isinstance(state, Iterable):
            mask_48 = 0xFFFFFFFFFFFF

            state = np.fromiter(state, dtype=object)
            length = len(state)
            lo = (state & mask_48).astype(np.int64).reshape(-1, 1)
            mi = ((state >> 48) & mask_48).astype(np.int64).reshape(-1, 1)
            hi = ((state >> 96) & mask_48).astype(np.int64).reshape(-1, 1)

            del state

            shifts = np.arange(0, 48, 3)
            lo = (lo >> shifts) & mask_3
            mi = (mi >> shifts) & mask_3
            hi = (hi >> shifts) & mask_3
            result = np.concat((lo, mi, hi), axis=1)

            del lo, mi, hi  # this probably does nothing since lo, mi, hi likely share memory with result

            if one_hot:
                return np.eye(6, dtype=dtype)[result].reshape(length, -1)
            return result.astype(dtype)

        arr = np.empty(48, dtype=np.uint8)
        for i in range(48):
            arr[i] = (state >> (3 * i)) & mask_3

        if one_hot:
            return np.eye(6, dtype=dtype)[arr].reshape(-1)
        return arr.astype(dtype)

    @classmethod
    def _to_mxarray(cls, state: int | Iterable[int], dtype: 'mlx.core.dtype', one_hot: bool) -> 'mlx.core.array':  # type: ignore
        '''Convert a state or sequence of states to a MLX array.

        Parameters
        ----------
        state : int | Sequence[int]
            The integer representing the cube state.
        dtype : mx.Dtype
            The dtype for the array.
        one_hot : bool
            Whether or not to one-hot encode the tensor.

        Returns
        -------
        mx.array
            An array representing the cube state(s).
        '''

        import mlx.core as mx

        mask_3 = 0x7
        # Check for reusable iterable (more general than `isinstance(state, Sequence)`)
        if isinstance(state, Iterable) and not isinstance(state, Iterator):
            mask_48 = 0xFFFFFFFFFFFF

            lo = mx.array([s & mask_48 for s in state], dtype=mx.int64).reshape(-1, 1)
            mi = mx.array([(s >> 48) & mask_48 for s in state], dtype=mx.int64).reshape(-1, 1)
            hi = mx.array([(s >> 96) & mask_48 for s in state], dtype=mx.int64).reshape(-1, 1)

            length = len(state)
            del state

            shifts = mx.arange(0, 48, 3, dtype=mx.int64)
            lo = (lo >> shifts) & mask_3
            mi = (mi >> shifts) & mask_3
            hi = (hi >> shifts) & mask_3
            result = mx.concat((lo, mi, hi), axis=1)

            del lo, mi, hi  # this probably does nothing since lo, mi, hi likely share memory with result

            if one_hot:
                return mx.eye(6, dtype=dtype)[result].reshape(length, -1)
            return result.astype(dtype)

        arr = mx.zeros(48, dtype=mx.uint8)
        for i in range(48):
            arr[i] = (state >> (3 * i)) & mask_3

        if one_hot:
            return mx.eye(6, dtype=dtype)[arr].reshape(-1)
        return arr.astype(dtype)

    def is_solved(self) -> bool:
        '''Check if the cube is solved.

        Returns
        -------
        solved : bool
            True if self is in the solved state, False otherwise.
        '''

        return self.state == self._SOLVED_STATE

    def reset(self):
        '''Reset the cube to the solved state.'''

        self.state = self._SOLVED_STATE

    @classmethod
    def get_move_sequence(cls, cubes: Optional[list['RubiksCube']]) -> list[str] | None:
        '''Get a sequence of moves from a list of cubes.

        Parameters
        ----------
        cubes : Optional[list[RubiksCube]]
            The list of cubes from which to determine moves.

        Returns
        -------
        moves : list[str]
            A list of moves corresponding to the cube states. If no move sequence exists for the cube states,
            None is returned.
        '''

        if cubes is None:
            return None

        if len(cubes) < 2:
            return []

        moves = []
        state = cubes[0]
        for cube in cubes[1:]:
            if state == cube:
                continue
            for move in cls._MOVES:
                state._move(move)
                if state == cube:
                    moves.append(move)
                    break
                state._move(cls.inverse(move))
            else:
                return None

        return moves

    def copy(self) -> 'RubiksCube':
        '''Return a copy of self.

        Returns
        -------
        RubiksCube
            A copy of self.
        '''

        return self.__copy__()

    def __hash__(self) -> int:
        '''Modulo hash to keep the result within 64 bits.

        Returns
        -------
        int
            The hash of the cube.
        '''

        return self.state % 18446744073709551557  # 2**64 - 59

    def __copy__(self) -> 'RubiksCube':
        '''Return a copy of self.

        Returns
        -------
        RubiksCube
            A copy of self.
        '''

        return self.__class__(self.state)

    def __eq__(self, other: 'RubiksCube') -> bool:
        '''Test 2 cubes for equality.

        Parameters
        ----------
        other : RubiksCube
            The cube to test for equality with self.

        Returns
        -------
        bool
            The result of the equality test.
        '''

        return self.state == other.state

    def __str__(self) -> str:
        '''Get a string representation of self.

        Returns
        -------
        cube : string
                       21  18  15
                       0       12
                       3   6   9

        117 114 111    45  42  39    69  66  63    93  90  87
        96      108    24      36    48      60    72      84
        99  102 105    27  30  33    51  54  57    75  78  81

                       141 138 135
                       120     132
                       123 126 129
        '''

        colors = {
            0b000: '🟥',
            0b001: '⬜️',
            0b010: '🟩',
            0b011: '🟨',
            0b100: '🟦',
            0b101: '🟧',
        }
        ansi_by_label = {
            'U': '\033[31m',
            'F': '\033[37m',
            'R': '\033[32m',
            'B': '\033[33m',
            'L': '\033[34m',
            'D': '\033[38;5;214m',
        }
        full_width_unicode = {'U': '\uff35', 'F': '\uff26', 'R': '\uff32', 'B': '\uff22', 'L': '\uff2c', 'D': '\uff24'}

        faces = {}
        for i, face in enumerate('UFRBLD'):
            shift = i * 24
            faces[face] = [
                f'{colors[extract_3(self.state, shift + 21)]}'
                f'{colors[extract_3(self.state, shift + 18)]}'
                f'{colors[extract_3(self.state, shift + 15)]}',
                f'{colors[extract_3(self.state, shift + 0)]}'
                f'\033[1m{ansi_by_label[face]}{full_width_unicode[face]}\033[0m'
                f'{colors[extract_3(self.state, shift + 12)]}',
                f'{colors[extract_3(self.state, shift + 3)]}'
                f'{colors[extract_3(self.state, shift + 6)]}'
                f'{colors[extract_3(self.state, shift + 9)]}',
            ]

        return f'''
       {faces['U'][0]}
       {faces['U'][1]}
       {faces['U'][2]}

{faces['L'][0]} {faces['F'][0]} {faces['R'][0]} {faces['B'][0]}
{faces['L'][1]} {faces['F'][1]} {faces['R'][1]} {faces['B'][1]}
{faces['L'][2]} {faces['F'][2]} {faces['R'][2]} {faces['B'][2]}

       {faces['D'][0]}
       {faces['D'][1]}
       {faces['D'][2]}
'''

    def __repr__(self) -> str:
        return f'<Cube: {hash(self)}>'


if __name__ == '__main__':
    RubiksCube.from_santa(
        'A;A;E;C;F;F;C;C;C;D;B;D;D;B;A;B;F;F;A;E;B;D;E;B;D;B;B;A;B;D;F;D;F;F;A;A;E;A;F;C;C;C;C;D;F;C;D;E;E;A;E;B;E;E'
    )
