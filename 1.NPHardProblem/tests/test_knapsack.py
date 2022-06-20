import pytest

from collections import Counter

import assignment


def test_knapsack(knapsack_inputs, expected_knapsacks):
    result = assignment.knapsack_01(*knapsack_inputs)
    assert Counter(result) in expected_knapsacks
