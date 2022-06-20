import pytest

from collections import Counter
import numpy as np
from main import optimize_portfolio

ITERS = 10000
SLACK = 0.05


@pytest.mark.parametrize(
    "values,expected,ratio",
    [
        (
            np.array(
                [
                    [100, 99, 98, 101, 102],
                    [100, 95, 88, 96, 103],
                    [100, 103, 107, 106, 104],
                ]
            ),
            ([0, 2], 6.21),
            1,
        ),
        (np.array([[100, 65, 70, 50, 110], [100, 100, 60, 60, 110]]), ([1], 0.25), 1,),
    ],
)
def test_optimize_portfolio(values, expected, ratio):
    successes = 0
    expected_subset, expected_r_r = expected
    for _ in range(ITERS):
        subset, r_r = optimize_portfolio(values)
        if Counter(subset) == Counter(expected_subset) and np.isclose(
            r_r, expected_r_r
        ):
            successes += 1
    assert ratio * (1 - SLACK) < successes / ITERS < ratio * (1 + SLACK)

