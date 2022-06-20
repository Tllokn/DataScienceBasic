from multi_bloom_filter import MultiBloomFilter
import math


class CollisionOptimalMBF(MultiBloomFilter):
    """A MultiBloomFilter that is optimized to minimize collision probability.

    This class takes a capacity and n (the expected number of items to add)
    as parameters and returns a multi-bloom-filter with k sub-filters,
    where k is chosen to minimize the likelihood of collisions.

    You must only implement code to find the optimal such k
    (given an overall capacity and expected number of items).

    Please list the rationale for your choice of k in the docstring.
    Floors and ceilings can be ignored in the analysis.
    """

    def __init__(self, capacity: int, n: int):
        k = self.get_optimal_k(capacity, n)
        super().__init__(capacity=capacity, k=k)

    @staticmethod
    def get_optimal_k(capacity: int, n: int) -> int:
        """
        TODO: Explain your approach in this docstring
        1.
        the probability that a certain bit is not set to 1 by a certain hash function
        during the insertion of an element is 1-1/m
        2.
        then the probability that the bit is not set to 1 by any of the hash functions is
        (1-1/m)^k

        ********: for large m, (1-1/m)^k = e^(-k/m)

        3.
        we have inserted n elements, the probability that a certain bit is still 0 is
        (1-1/m)^(kn)
        4.
        the probability that it is 1 is therefore
        1-(1-1/m)^(kn)
        5.
        False Positive Prob=(1-(1-1/m)^(kn))^k
        m=capacity//k

        ********: for large m, (1-1/m)^k = e^(-k/m)
        False Positive Prob=(1-e^(-kn/m))^k
        """
        # TODO
        opt_k=0
        opt_prob=1
        for k in range(1,n):
            # prob=(1-(1-1//(capacity//k))**(k*n))**k
            prob=(1-math.exp(-n*k*k/capacity))**k
            if prob<opt_prob:
                opt_prob=prob
                opt_k=k
        return opt_k


# Simple test code
if __name__ == "__main__":
    from random import randint

    n = 50
    capacity = 50
    items = [randint(0, 2 ** 31 - 1) for _ in range(n)]
    bf = CollisionOptimalMBF(capacity, n)
    bf.add_all(*items)
    tests = [randint(0, 2 ** 31 - 1) for _ in range(10)]
    print(f"All items included: {all(i in bf for i in items)}")
    for t in tests:
        print(f"{t} in bf: {t in bf}")
