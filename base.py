from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class BloomFilterABC(ABC, Generic[T]):
    """Abstract base class for Bloom filter implementations.

    Recall that a Bloom filter is a probabilistic set data structure
    with a __contains__ operation that can return false positives, but not false negatives,
    i.e. "probably in set" or "definitely not in set".

    Attributes
    ---------
    _capacity : int
        Total number of bits in the Bloom filter.
    _num_filled : int
        Number of bits set to 1 in the Bloom filter.
    _num_added : int
        Number of items added to the Bloom filter.
    """

    def __init__(self, capacity: int):
        """Constructs the Bloom filter."""
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer!")
        self._capacity = capacity
        self._num_filled = 0
        self._num_added = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_filled(self) -> int:
        return self._num_filled

    @property
    def num_empty(self) -> int:
        return self._capacity - self._num_filled

    @property
    def percent_filled(self) -> float:
        return self._num_filled / self._capacity * 100

    @property
    def num_added(self) -> int:
        return self._num_added

    @property
    def num_collisions(self) -> int:
        return self._num_added - self._num_filled

    @abstractmethod
    def __contains__(self, item: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _add(self, item: T) -> bool:
        raise NotImplementedError

    def add(self, item: T) -> None:
        """Adds a single item, keeping track of the number of bits filled and items added."""
        item_already_present = item in self
        new_bit_set = self._add(item)
        assert item in self
        self._num_added += 1
        if new_bit_set:
            assert not item_already_present
            self._num_filled += 1

    def add_all(self, *items: T) -> None:
        """Adds multiple items at once. """
        for item in items:
            self.add(item)

    @abstractmethod
    def _hash(self, item: T) -> int:
        """Hashes the item."""
        raise NotImplementedError

    def hash(self, item: T) -> int:
        return self._hash(item) % self.capacity

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"capacity={self.capacity}, "
            f"num_filled={self.num_filled}), "
            f"num_added={self.num_added})"
        )
