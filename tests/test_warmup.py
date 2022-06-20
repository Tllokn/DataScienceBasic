import pytest

from collections import Counter
import itertools
import warmup


def test_permutations(warmup_sets):
    results = [item for item in warmup.generate_permutations(warmup_sets)]
    expected = [item for item in itertools.permutations(warmup_sets)]
    assert Counter(results) == Counter(expected)


def test_subsets(warmup_sets, warmup_subset_ks):
    results = [item for item in warmup.generate_subsets(warmup_sets, warmup_subset_ks)]
    expected = [item for item in itertools.combinations(warmup_sets, warmup_subset_ks)]
    assert Counter(results) == Counter(expected)


def test_powerset(warmup_sets):
    results = [item for item in warmup.generate_powerset(warmup_sets)]
    expected = [
        item
        for k in range(len(warmup_sets) + 1)
        for item in itertools.combinations(warmup_sets, k)
    ]
    assert Counter(results) == Counter(expected)
