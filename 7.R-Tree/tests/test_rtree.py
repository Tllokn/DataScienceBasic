import pytest
from shapely.geometry import Point, Polygon

from rtree import RTree


@pytest.fixture(scope="module")
def example_rtree() -> RTree:
    points = [
        Point(0.79, 0.82),  # top right
        Point(0.49, 0.26),  # bottom left
        Point(0.01, 0.66),  # top left
        Point(0.47, 0.76),  # top left
        Point(0.37, 0.77),  # top left
        Point(0.27, 0.80),  # top left
        Point(0.73, 0.41),  # bottom right
        Point(0.54, 0.68),  # top right
        Point(0.19, 0.55),  # top left
        Point(0.80, 0.69),  # top right
        Point(0.84, 0.34),  # bottom right
    ]
    rtree = RTree(min_degree=2)
    for point in points:
        rtree.insert(point)
    return rtree


@pytest.mark.parametrize(
    "query,num_expected",
    [
        pytest.param(
            Polygon([(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)]), 1, id="bottom left"
        ),
        pytest.param(
            Polygon([(0.5, 0), (0.5, 0.5), (1, 0.5), (1, 0)]), 2, id="bottom right"
        ),
        pytest.param(
            Polygon([(0, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]), 5, id="top left"
        ),
        pytest.param(
            Polygon([(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]), 3, id="top right"
        ),
    ],
)
def test_rtree_search(example_rtree: RTree, query: Polygon, num_expected: int):
    results = list(example_rtree.search(query))
    print(*results)
    assert len(results) == num_expected
    assert all(query.contains(result) for result in results)
