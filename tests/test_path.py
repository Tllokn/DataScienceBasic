import pytest

import assignment


def test_path(path_inputs, expected_paths):
    result = assignment.longest_path(path_inputs)
    assert result in expected_paths
