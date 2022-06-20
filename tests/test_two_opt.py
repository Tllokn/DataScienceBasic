import pytest

import main
import utils


def test_two_opt(input_graphs, optimal_solutions):
    result = main.two_opt(input_graphs)
    assert utils.check_cycle(
        input_graphs,
        result,
        optimum=optimal_solutions,
        approximation_ratio=(len(input_graphs.nodes) / 2) ** 0.5,
    )

