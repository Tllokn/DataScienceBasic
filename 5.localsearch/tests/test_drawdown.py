import pytest

import numpy as np
from main import compute_max_drawdown


@pytest.mark.parametrize(
    "values,expected",
    [
        pytest.param(
            np.array([50, 48, 51, 55, 57, 60, 58, 54, 56, 57]), 10.0, id="Example"
        ),
        pytest.param(np.array([100, 99, 98, 101, 102]), 2.0, id="Optimize1"),
        pytest.param(np.array([100, 95, 88, 96, 103]), 12.0, id="Optimize2"),
        pytest.param(
            np.array([100, 103, 107, 106, 104]), (3.0 / 107.0) * 100, id="Optimize3"
        ),
        pytest.param(np.array([100, 65, 70, 50, 110]), 50.0, id="Optimize4"),
        pytest.param(np.array([100, 110, 120, 130, 140, 150]), 0.0, id="Zero"),
    ],
)
def test_compute_max_drawdown(values, expected):
    result = compute_max_drawdown(values)
    assert np.isclose(result, expected)
