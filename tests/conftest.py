import pytest

import random


@pytest.fixture(scope="module", params=range(6))
def test_number(request):
    return request.param


@pytest.fixture(scope="module")
def input_simple(test_number):
    input = {
        0: (10, 1, None),
        1: (100, 100, None),
    }
    if test_number not in input:
        return (random.randint(1, 1000), random.randint(1, 1000), None)
    return input[test_number]


@pytest.fixture(scope="module")
def input_multi(test_number):
    input = {
        0: (10, 1, 1, None),
        1: (100, 50, 10, None),
    }
    if test_number not in input:
        n = random.randint(1, 100)
        k = random.randint(1, 10)
        capacity = random.randint(k, 100)
        return (
            n,
            capacity,
            k,
            None,
        )
    return input[test_number]


@pytest.fixture(scope="module")
def input_optimal(test_number):
    input = {
        0: (10, 2),
        1: (100, 100),
    }
    if test_number not in input:
        capacity = random.randint(1, 1000)
        n = random.randint(1, capacity // 2)
        return (n, capacity)
    return input[test_number]
