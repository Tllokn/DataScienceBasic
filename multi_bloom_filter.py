from __future__ import annotations

from typing import Callable, Sequence

from base import BloomFilterABC
from simple_bloom_filter import SimpleBloomFilter
from utils import get_universal_hash
from bitarray import bitarray


class MultiBloomFilter(BloomFilterABC[int]):
    """Bloom filter which stores a set of integers by aggregating multiple SimpleBloomFilters.

    Consists of k SimpleBloomFilter instances, all with the same capacity.
    Answers membership queries by searching for the element in each SimpleBloomFilter.
    Adds an element by adding it to each SimpleBloomFilter.

    Attributes
    ----------
    filters : list[SimpleBloomFilter]
        List of constituent SimpleBloomFilters.
    hash_tables: list[bitarray]
        List of hash tables
    """

    def __init__(
        self,
        capacity: int,
        k: int,
        hash_functions: Sequence[Callable[[int], int]] | None = None,
    ):
        super().__init__(capacity)
        m = capacity // k
        if hash_functions is None:
            hash_functions = [get_universal_hash(m) for _ in range(k)]
        assert len(hash_functions) == k
        self.filters: list[SimpleBloomFilter] = [
            SimpleBloomFilter(capacity=m, hash_function=h) for h in hash_functions
        ]
        # TODO: Add any other instance variables that you need


    @property
    def num_filled(self) -> int:
        """Number of slots filled summed over all filters"""
        return sum(f.num_filled for f in self.filters)

    def _hash(self, item: int):
        # Shouldn't need this function; can rely on SimpleBloomFilter._hash
        pass

    def __contains__(self, item: int) -> bool:
        # TODO
        for f in self.filters:
            if item not in f:
                return False
        return True

    def _add(self, item: int) -> bool:
        # TODO
        if item in self:
            return False
        for f in self.filters:
            f.add(item)
        return True


# Simple test code
if __name__ == "__main__":
    from random import randint

    n = 50
    k = 2
    items = [randint(0, 2 ** 31 - 1) for _ in range(n)]
    bf = MultiBloomFilter(n, k)
    bf.add_all(*items)
    tests = [randint(0, 2 ** 31 - 1) for _ in range(10)]
    print(f"All items included: {all(i in bf for i in items)}")
    for t in tests:
        print(f"{t} in bf: {t in bf}")
