import pytest

import numpy as np
from main import ValidNumberTree


def generate_random_ranges(low=0, high=100, size=5):
    centers = np.random.uniform(low, high, size=size)
    distances = np.abs(np.random.normal(0, np.sqrt(high - low) / 2, size=size))
    float_ranges = np.vstack((centers - distances, centers + distances)).T
    ranges = np.clip(float_ranges, low, high)
    return ranges.astype(int)


@pytest.mark.parametrize(
    "ranges, query_range",
    [
        pytest.param(
            [
                [1, 90],
                [10, 15],
                [20, 25],
                [30, 35],
                [10, 40],
                [0, 5],
                [0, 2],
                [0, 3],
                [0, 1],
                [0, 6],
                [0, 5],
            ],
            (0, 100),
            id="Example",
        ),
        pytest.param(generate_random_ranges(0, 100, 5), (0, 100), id="Random1",),
        pytest.param(generate_random_ranges(0, 1000, 10), (0, 1000), id="Random2",),
        pytest.param(generate_random_ranges(0, 1000, 10), (250, 500), id="Random3",),
        pytest.param(generate_random_ranges(5, 5, 5), (0, 10), id="Random4",),
    ],
)
def test_tree(ranges, query_range):
    tree = ValidNumberTree()
    queries = [x for x in range(*query_range)]
    for lo, hi in ranges:
        tree.add_range(lo, hi)
    for q in queries:
        result = tree.is_valid(q)
        expected = any(lo <= q <= hi for lo, hi in ranges)
        assert result == expected
