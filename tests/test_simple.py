import pytest

import random

from simple_bloom_filter import SimpleBloomFilter

ITERS = 1000
SLACK = 0.05


def test_simple(input_simple):
    n, capacity, hash_function = input_simple
    false_positives = 0
    expected_fp_rate = 1 - (1 - 1 / capacity) ** n

    for _ in range(ITERS):
        bf = SimpleBloomFilter(capacity, hash_function)

        items = [random.randint(0, 2 ** 31 - 1) for _ in range(n)]
        bf.add_all(*items)

        assert all(i in bf for i in items)

        items = set(items)

        test_int = random.randint(0, 2 ** 31 - 1)
        while test_int in items:
            test_int = random.randint(0, 2 ** 31 - 1)
        if test_int in bf:
            false_positives += 1
    assert (
        expected_fp_rate - SLACK <= false_positives / ITERS <= expected_fp_rate + SLACK
    )

