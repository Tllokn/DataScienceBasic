import pytest

import numpy as np
from main import compute_return


@pytest.mark.parametrize(
    "values,expected",
    [
        pytest.param(
            np.array([50, 48, 51, 55, 57, 60, 58, 54, 56, 57]), 14.0, id="Example"
        ),
        pytest.param(np.array([100, 99, 98, 101, 102]), 2.0, id="Optimize1"),
        pytest.param(np.array([100, 95, 88, 96, 103]), 3.0, id="Optimize2"),
        pytest.param(np.array([100, 103, 107, 106, 104]), 4.0, id="Optimize3"),
        pytest.param(np.array([100, 65, 70, 50, 110]), 10.0, id="Optimize4"),
        pytest.param(np.array([100, 65, 110, 150, 90]), -10.0, id="Negative"),
    ],
)
def test_compute_return(values, expected):
    result = compute_return(values)
    assert np.isclose(result, expected)

