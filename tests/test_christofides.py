import pytest

import main
import utils


def test_christofides(input_graphs, optimal_solutions):
    result = main.christofides(input_graphs)
    assert utils.check_cycle(
        input_graphs, result, optimum=optimal_solutions, approximation_ratio=3 / 2
    )

