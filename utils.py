from __future__ import annotations

from random import randint
from typing import Callable

import numpy as np


def get_universal_hash(m: int, p: int | None = None) -> Callable[[int], int]:
    """Selects a hash function at random.

    Parameters
    ----------
    m : int
        Number of buckets / bins to hash into.
    p : int | None
        Prime number larger than m.
        If not provided, will be sampled from the range (m, 5m).

    Returns
    -------
    hash_function : Callable[[int], int]
        Random hash function.
    """
    if p is None:
        p = get_random_prime(m)
    if p < m:
        raise ValueError("p must be >= m!")
    a, b = randint(1, p), randint(0, p)
    assert a != 0
    return lambda x: ((a*x + b) % p) % m


"""Inefficient prime-finding code"""


def is_prime(x: int) -> bool:
    """Naive primality test using trial division."""
    for i in range(2, int(np.sqrt(x))):
        if x % i == 0:
            return False
    return True


def get_random_prime(m: int):
    """Generates a random prime between m+1 and 5m."""
    # ~log(4m)/4m primes in range (m, 5m)
    x = np.random.randint(m+1, 5*m)
    while not is_prime(x):
        x = np.random.randint(m+1, 5*m)
    return x
