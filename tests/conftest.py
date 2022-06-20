import pytest

import glob
import os

import tsplib95


@pytest.fixture(scope="module", params=glob.iglob(os.path.join("data", "*.tsp")))
def file_name(request):
    return request.param


@pytest.fixture(scope="module")
def input_graphs(file_name):
    return tsplib95.load(file_name).get_graph()


@pytest.fixture(scope="module")
def optimal_solutions(file_name):
    return {
        "bayg29.tsp": 1610,
        "berlin52.tsp": 7542,
        "bier127.tsp": 118282,
        "brazil58.tsp": 25395,
        "ch130.tsp": 6110,
        "ch150.tsp": 6528,
        "dantzig42.tsp": 699,
        "eil51.tsp": 426,
        "eil76.tsp": 538,
        "eil101.tsp": 629,
        "fri26.tsp": 937,
        "gr17.tsp": 2085,
        "gr21.tsp": 2707,
        "gr24.tsp": 1272,
        "gr48.tsp": 5046,
        "gr120.tsp": 6942,
        "hk48.tsp": 11461,
        "kroA100.tsp": 21282,
        "kroB100.tsp": 22141,
        "kroC100.tsp": 20749,
        "kroD100.tsp": 21294,
        "kroE100.tsp": 22068,
        "kroA150.tsp": 26524,
        "kroB150.tsp": 26130,
        "kroA200.tsp": 29368,
        "kroB200.tsp": 29437,
        "lin105.tsp": 14379,
        "pr76.tsp": 108159,
        "pr107.tsp": 44303,
        "pr124.tsp": 59030,
        "pr136.tsp": 96772,
        "pr144.tsp": 58537,
        "pr152.tsp": 73682,
        "rd100.tsp": 7910,
        "st70.tsp": 675,
    }[os.path.basename(file_name)]

