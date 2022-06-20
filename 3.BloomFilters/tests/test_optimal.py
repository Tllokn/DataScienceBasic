import pytest

import random

from collision_optimal_mbf import CollisionOptimalMBF
from multi_bloom_filter import MultiBloomFilter

ITERS = 500
MAX_K = 10
SLACK = 0.05


def test_optimal(input_optimal):
    n, capacity = input_optimal
    false_positives = [0, 0]

    for _ in range(ITERS):
        bf_opt = CollisionOptimalMBF(capacity, n)
        rand_k = random.randint(1, min(capacity, MAX_K))
        bf_rand = MultiBloomFilter(capacity, rand_k, None)

        for i, bf in enumerate((bf_opt, bf_rand)):
            items = [random.randint(0, 2 ** 31 - 1) for _ in range(n)]
            bf.add_all(*items)

            assert all(i in bf for i in items)

            items = set(items)

            test_int = random.randint(0, 2 ** 31 - 1)
            while test_int in items:
                test_int = random.randint(0, 2 ** 31 - 1)
            if test_int in bf:
                false_positives[i] += 1

    assert (false_positives[0] / ITERS) - SLACK <= (false_positives[1] / ITERS)

