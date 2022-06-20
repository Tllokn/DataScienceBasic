import pytest

import random

from multi_bloom_filter import MultiBloomFilter

ITERS = 1000
SLACK = 0.05


def test_multi(input_multi):
    n, capacity, k, hash_function = input_multi
    false_positives = 0
    expected_fp_rate = (1 - (1 - 1 / (capacity // k)) ** n) ** k

    for _ in range(ITERS):
        bf = MultiBloomFilter(capacity, k, hash_function)

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

