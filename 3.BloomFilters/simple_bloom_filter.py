from __future__ import annotations

from typing import Callable

from base import BloomFilterABC

from bitarray import bitarray


class SimpleBloomFilter(BloomFilterABC[int]):
    """Simple Bloom filter which stores a set of integers using a single bit array."""

    def __init__(
            self, capacity: int, hash_function: Callable[[int], int] | None = None
    ):
        """Constructs the simple Bloom filter.

        Parameters
        ----------
        capacity : int
            Total number of bits in the SimpleBloomFilter.
        hash_function : Callable[[int], int | None
            A single custom hash function. If None, defaults to built-in hash function.
        """
        super().__init__(capacity=capacity)
        self._hash_function = hash_function if hash_function is not None else hash

        # TODO: Create bit array
        self.hash_table = bitarray([False] * capacity)

    def _hash(self, item: int) -> int:
        """Hashes the item using the provided hash function."""
        return self._hash_function(item)

    def __contains__(self, item: int) -> bool:
        # TODO
        if self.hash_table[self.hash(item)]:
            return True
        else:
            return False

    def _add(self, item: int) -> bool:
        """Adds the item to the filter.

        Parameters
        ----------
        item : T
            Value to add to the filter.

        Returns
        -------
        set_new_bit : bool
           True iff the item's hash was not present in the filter previously (a new bit is set).
        """
        # TODO
        if not self.hash_table[self.hash(item)]:
            self.hash_table[self.hash(item)] = True
            return True
        else:
            return False


# Simple test code
if __name__ == "__main__":
    from random import randint

    n = 50
    items = [randint(0, 2 ** 31 - 1) for _ in range(n)]
    bf = SimpleBloomFilter(n)
    bf.add_all(*items)
    tests = [randint(0, 2 ** 31 - 1) for _ in range(10)]
    print(f"All items included: {all(i in bf for i in items)}")
    for t in tests:
        print(f"{t} in bf: {t in bf}")
