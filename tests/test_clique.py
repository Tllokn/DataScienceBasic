import pytest

from collections import Counter

import assignment


def test_clique(clique_inputs, expected_cliques):
    result = assignment.max_weighted_clique(clique_inputs)
    assert Counter(result) in expected_cliques
