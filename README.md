# Intractability

In this assignment, you will write brute force algorithms for various NP-hard problems.
The goal is to develop a feel for how difficult these problems are to solve exactly,
and motivate the need for heuristics and approximation algorithms. 
Despite their intractability, brute force may still be the right choice for small cases.

## Dependencies

We will use the following packages:
- [NumPy](https://numpy.org/doc/stable/) for random number generation
- [NetworkX](https://networkx.org/) for working with graphs
- [Pytest](https://docs.pytest.org/en/stable/) for unit testing

Please install these packages, e.g. via `python3 -m pip install -r requirements.txt`.
You are responsible for managing your own environment, 
but please ask in Slack if you need help.

## Warmup

Please complete the three combinatorial functions in `warmup.py`. 
**You may not use any imports, including the standard library.** 

To reduce memory usage, implement each function as a 
[generator](https://wiki.python.org/moin/Generators).
This is more efficient because the full output is never held in memory.
If you tried to return a list instead, your machine would likely run out of memory on large inputs.
With generators, the code will still complete, though it may take a very long time.

You should also have your generators yield *tuples* representing each permutation, subset, or powerset element.
The immutability of tuples is useful for hashing or comparing for equality.

## Assignment

In `assignment.py`, please implement algorithms for the following three NP-hard problems:
1. [Weighted clique](https://en.wikipedia.org/wiki/Clique_problem)
2. [Longest path](https://en.wikipedia.org/wiki/Longest_path_problem)
3. [0-1 Knapsack](https://en.wikipedia.org/wiki/Knapsack_problem)

You may not use algorithms from NetworkX or modify the call signatures.
You are encouraged to use functions from `warmup.py` as needed.
It may be convenient to add helper methods for each problem - 
you should do this wherever necessary to keep your code clean and readable.

## Testing

A number of automated test cases are located in the `tests/` folder. 
To run all of them at once, run the `pytest` command in your terminal. 
To run a specific set of tests, e.g. `test_warmup.py` use `pytest tests/test_warmup.py`.

Some of the tests may be time-consuming (especially if your implementation is slow).
You will only be graded on theoretical (asymptotic) runtime.

*Disclaimer:* Tests are provided as a guide and sanity check only.
**Passing all tests is not a guarantee of full credit**.
We cannot anticipate every potential mistake; you may make a mistake that is not covered by the test cases.

You are also welcome (and highly encouraged) to write your own unit tests - you can follow the format of the existing tests.

To generate random inputs for these problems, you can use the functions from `utils.py`:
1. For weighted clique, use `gen_vtx_weighted_graph(n, p=0.5, min_wt=0, max_wt=1, directed=False)`. 
2. For longest path, use `gen_edge_weighted_graph(n, p=0.5, min_wt=0, max_wt=1, directed=False)`
3. For knapsack, use `gen_backpack_instance(n, capacity=10, min_wt=0, max_wt=1, min_val=0, max_val=1)`

You can pass a seed argument to any of these functions for reproducibility (e.g. `seed=42`).
