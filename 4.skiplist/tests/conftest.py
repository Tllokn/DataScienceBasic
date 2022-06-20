import pytest

import random

MAX_HEIGHT = 20
MAX_N = 500


@pytest.fixture(scope="module", params=range(5))
def test_number(request):
    return request.param


@pytest.fixture(scope="module")
def inputs(test_number):
    input = {
        0: (10, 0.5, 200),
        1: (1, 0, 500),
    }
    if test_number not in input:
        return (
            random.randint(1, MAX_HEIGHT),
            random.random(),
            random.randint(1, MAX_N),
        )
    return input[test_number]
