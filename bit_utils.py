_3_BIT_MASK = 0x7
_6_BIT_MASK = 0x3F
_9_BIT_MASK = 0x1FF
_24_BIT_MASK = 0xFFFFFF


def left_circular_shift24(x: int, start: int, shift: int):
    '''Perform a circular shift by `shift` bits on a 24-bit subset of `x`
    beginning at bit position `start`.
    '''

    subfield = (x >> start) & _24_BIT_MASK
    rotated = ((subfield << shift) | (subfield >> (24 - shift))) & _24_BIT_MASK
    return (x & ~(_24_BIT_MASK << start)) | (rotated << start)


# def swap_3_bits(x: int, s1: int, s2: int):
#     '''Swap 3 bit subfields beginning at `s1` and `s2` in `x`.

#     Parameters
#     ----------
#     x : int
#         The integer.
#     s1 : int
#         The start position of the first set of bits to swap.
#     s2 : int
#         The start position of the second set of bits to swap.

#     Returns
#     -------
#     int
#         The integer with the bit subfields swapped.
#     '''

#     f1 = (x >> s1) & _3_BIT_MASK
#     f2 = (x >> s2) & _3_BIT_MASK

#     x &= ~((_3_BIT_MASK << s1) | (_3_BIT_MASK << s2))
#     return x | (f1 << s2) | (f2 << s1)


# def swap_6_bits(x: int, s1: int, s2: int):
#     '''Swap 6 bit subfields beginning at `s1` and `s2` in `x`.

#     Parameters
#     ----------
#     x : int
#         The integer.
#     s1 : int
#         The start position of the first set of bits to swap.
#     s2 : int
#         The start position of the second set of bits to swap.

#     Returns
#     -------
#     int
#         The integer with the bit subfields swapped.
#     '''

#     f1 = (x >> s1) & _6_BIT_MASK
#     f2 = (x >> s2) & _6_BIT_MASK

#     x &= ~((_6_BIT_MASK << s1) | (_6_BIT_MASK << s2))
#     return x | (f1 << s2) | (f2 << s1)


# def swap_9_bits(x: int, s1: int, s2: int):
#     '''Swap 9 bit subfields beginning at `s1` and `s2` in `x`.

#     Parameters
#     ----------
#     x : int
#         The integer.
#     s1 : int
#         The start position of the first set of bits to swap.
#     s2 : int
#         The start position of the second set of bits to swap.

#     Returns
#     -------
#     int
#         The integer with the bit subfields swapped.
#     '''

#     f1 = (x >> s1) & _9_BIT_MASK
#     f2 = (x >> s2) & _9_BIT_MASK

#     x &= ~((_9_BIT_MASK << s1) | (_9_BIT_MASK << s2))
#     return x | (f1 << s2) | (f2 << s1)


def swap_9_bits_twice(x: int, s11: int, s12: int, s21: int, s22: int) -> int:
    '''Swap two sets of 9 bit subfields in `x`.

    Parameters
    ----------
    x : int
        The integer.
    s11, s12 : int
        The start positions of the first set of bits to swap.
    s21, s22 : int
        The start positions of the second set of bits to swap.

    Returns
    -------
    int
        The integer with the bit subfields swapped.
    '''

    f11 = (x >> s11) & _9_BIT_MASK
    f12 = (x >> s12) & _9_BIT_MASK
    f21 = (x >> s21) & _9_BIT_MASK
    f22 = (x >> s22) & _9_BIT_MASK

    x &= ~((_9_BIT_MASK << s11) | (_9_BIT_MASK << s12) | (_9_BIT_MASK << s21) | (_9_BIT_MASK << s22))
    return x | (f11 << s12) | (f12 << s11) | (f21 << s22) | (f22 << s21)


def swap_9_6_3_bits(x: int, s91: int, s92: int, s61: int, s62: int, s31: int, s32: int) -> int:
    '''Swap 9 bit, 6 bit, and 3 bit subfields in `x`.

    Parameters
    ----------
    x : int
        The integer.
    s91, s92 : int
        The start positions of the sets of 9 bits to swap.
    s61, s62 : int
        The start positions of the sets of 6 bits to swap.
    s31, s32 : int
        The start positions of the sets of 3 bits to swap.

    Returns
    -------
    int
        The integer with the bit subfields swapped.
    '''

    f91 = (x >> s91) & _9_BIT_MASK
    f92 = (x >> s92) & _9_BIT_MASK
    f61 = (x >> s61) & _6_BIT_MASK
    f62 = (x >> s62) & _6_BIT_MASK
    f31 = (x >> s31) & _3_BIT_MASK
    f32 = (x >> s32) & _3_BIT_MASK

    x &= ~(
        (_9_BIT_MASK << s91)
        | (_9_BIT_MASK << s92)
        | (_6_BIT_MASK << s61)
        | (_6_BIT_MASK << s62)
        | (_3_BIT_MASK << s31)
        | (_3_BIT_MASK << s32)
    )
    return x | (f91 << s92) | (f92 << s91) | (f61 << s62) | (f62 << s61) | (f31 << s32) | (f32 << s31)


def swap_6_3_bits_twice(
    x: int, s611: int, s612: int, s621: int, s622: int, s311: int, s312: int, s321: int, s322: int
) -> int:
    '''Swap two set of 6 bit and 3 bit subfields in `x`.

    Parameters
    ----------
    x : int
        The integer.
    s611, s612 : int
        The start positions of the first set of 6 bits to swap.
    s621, s622 : int
        The start positions of the second set of 6 bits to swap.
    s311, s312 : int
        The start positions of the first set of 3 bits to swap.
    s321, s322 : int
        The start positions of the second set of 3 bits to swap.

    Returns
    -------
    int
        The integer with the bit subfields swapped.
    '''

    f611 = (x >> s611) & _6_BIT_MASK
    f612 = (x >> s612) & _6_BIT_MASK
    f621 = (x >> s621) & _6_BIT_MASK
    f622 = (x >> s622) & _6_BIT_MASK
    f311 = (x >> s311) & _3_BIT_MASK
    f312 = (x >> s312) & _3_BIT_MASK
    f321 = (x >> s321) & _3_BIT_MASK
    f322 = (x >> s322) & _3_BIT_MASK

    x &= ~(
        (_6_BIT_MASK << s611)
        | (_6_BIT_MASK << s612)
        | (_6_BIT_MASK << s621)
        | (_6_BIT_MASK << s622)
        | (_3_BIT_MASK << s311)
        | (_3_BIT_MASK << s312)
        | (_3_BIT_MASK << s321)
        | (_3_BIT_MASK << s322)
    )
    return (
        x
        | (f611 << s612)
        | (f612 << s611)
        | (f621 << s622)
        | (f622 << s621)
        | (f311 << s312)
        | (f312 << s311)
        | (f321 << s322)
        | (f322 << s321)
    )


def cycle_3_bit_subfields(x: int, s1: int, s2: int, s3: int, s4: int) -> int:
    '''
    Cycle four 3-bit fields in a Python int.
    Bit 0 = least significant bit (LSB).

    Cycle direction:
        field0 -> field1
        field1 -> field2
        field2 -> field3
        field3 -> field0

    Parameters
    ----------
    x : int
        The integer.
    s1: int
        The start position of the first subfield.
    s2: int
        The start position of the second subfield.
    s3: int
        The start position of the third subfield.
    s4: int
        The start position of the fourth subfield.

    Returns
    -------
    int
        The integer with fields cycled.
    '''

    # Extract all 4 fields
    f0 = (x >> s1) & _3_BIT_MASK
    f1 = (x >> s2) & _3_BIT_MASK
    f2 = (x >> s3) & _3_BIT_MASK
    f3 = (x >> s4) & _3_BIT_MASK

    # Clear all 4 fields in x
    x &= ~((_3_BIT_MASK << s1) | (_3_BIT_MASK << s2) | (_3_BIT_MASK << s3) | (_3_BIT_MASK << s4))

    # Reinsert, cycled
    return x | (f3 << s1) | (f0 << s2) | (f1 << s3) | (f2 << s4)


# def cycle_6_bit_subfields(x: int, s1: int, s2: int, s3: int, s4: int) -> int:
#     '''
#     Cycle four 6-bit fields in a Python int.
#     Bit 0 = least significant bit (LSB).

#     Cycle direction:
#         field0 -> field1
#         field1 -> field2
#         field2 -> field3
#         field3 -> field0

#     Parameters
#     ----------
#     x : int
#         The integer.
#     s1: int
#         The start position of the first subfield.
#     s2: int
#         The start position of the second subfield.
#     s3: int
#         The start position of the third subfield.
#     s4: int
#         The start position of the fourth subfield.

#     Returns
#     -------
#     int
#         The integer with fields cycled.
#     '''

#     # Extract all 4 fields
#     f0 = (x >> s1) & _6_BIT_MASK
#     f1 = (x >> s2) & _6_BIT_MASK
#     f2 = (x >> s3) & _6_BIT_MASK
#     f3 = (x >> s4) & _6_BIT_MASK

#     # Clear all 4 fields in x
#     x &= ~((_6_BIT_MASK << s1) | (_6_BIT_MASK << s2) | (_6_BIT_MASK << s3) | (_6_BIT_MASK << s4))

#     # Reinsert, cycled
#     return x | (f3 << s1) | (f0 << s2) | (f1 << s3) | (f2 << s4)


def cycle_9_bit_subfields(x: int, s1: int, s2: int, s3: int, s4: int) -> int:
    '''
    Cycle four 9-bit fields in a Python int.
    Bit 0 = least significant bit (LSB).

    Cycle direction:
        field0 -> field1
        field1 -> field2
        field2 -> field3
        field3 -> field0

    Parameters
    ----------
    x : int
        The integer.
    s1: int
        The start position of the first subfield.
    s2: int
        The start position of the second subfield.
    s3: int
        The start position of the third subfield.
    s4: int
        The start position of the fourth subfield.

    Returns
    -------
    int
        The integer with fields cycled.
    '''

    # Extract all 4 fields
    f0 = (x >> s1) & _9_BIT_MASK
    f1 = (x >> s2) & _9_BIT_MASK
    f2 = (x >> s3) & _9_BIT_MASK
    f3 = (x >> s4) & _9_BIT_MASK

    # Clear all 4 fields in x
    x &= ~((_9_BIT_MASK << s1) | (_9_BIT_MASK << s2) | (_9_BIT_MASK << s3) | (_9_BIT_MASK << s4))

    # Reinsert, cycled
    return x | (f3 << s1) | (f0 << s2) | (f1 << s3) | (f2 << s4)


def cycle_6_and_3_bit_subfields(
    x: int, s61: int, s62: int, s63: int, s64: int, s31: int, s32: int, s33: int, s34: int
):
    '''
    Perform both 6-bit and 3-bit field cycling in a single operation.

    6-bit cycle: field0 -> field1 -> field2 -> field3 -> field0
    3-bit cycle: field0 -> field1 -> field2 -> field3 -> field0

    Parameters
    ----------
    x : int
        The integer to operate on
    s61, s62, s63, s64 : int
        Start positions of the four 6-bit fields
    s31, s32, s33, s34 : int
        Start positions of the four 3-bit fields

    Returns
    -------
    int
        The integer with both 6-bit and 3-bit fields cycled
    '''

    # Extract all 6-bit fields
    f6_0 = (x >> s61) & _6_BIT_MASK
    f6_1 = (x >> s62) & _6_BIT_MASK
    f6_2 = (x >> s63) & _6_BIT_MASK
    f6_3 = (x >> s64) & _6_BIT_MASK

    # Extract all 3-bit fields
    f3_0 = (x >> s31) & _3_BIT_MASK
    f3_1 = (x >> s32) & _3_BIT_MASK
    f3_2 = (x >> s33) & _3_BIT_MASK
    f3_3 = (x >> s34) & _3_BIT_MASK

    # Clear all 8 fields in one operation
    x &= ~(
        (_6_BIT_MASK << s61)
        | (_6_BIT_MASK << s62)
        | (_6_BIT_MASK << s63)
        | (_6_BIT_MASK << s64)
        | (_3_BIT_MASK << s31)
        | (_3_BIT_MASK << s32)
        | (_3_BIT_MASK << s33)
        | (_3_BIT_MASK << s34)
    )

    # Reinsert all cycled fields in one operation
    return (
        x
        | (f6_3 << s61)
        | (f6_0 << s62)
        | (f6_1 << s63)
        | (f6_2 << s64)
        | (f3_3 << s31)
        | (f3_0 << s32)
        | (f3_1 << s33)
        | (f3_2 << s34)
    )


def extract_3(x: int, start: int) -> int:
    '''Extract 3 bits from `x` beginning at position `start`.

    Parameters
    ----------
    x : int
        The integer.
    start : int
        The start position of the three bits to extract.

    Returns
    -------
    int
        The extracted 3 bits.
    '''

    return (x >> start) & _3_BIT_MASK
