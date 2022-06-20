import numpy as np
import matplotlib.pyplot as plt

from simple_bloom_filter import SimpleBloomFilter
from multi_bloom_filter import MultiBloomFilter
from collision_optimal_mbf import CollisionOptimalMBF

#size for check false positive
FPSize=100

"""
Test the performance of your Bloom Filters.

  - Add r elements uniformly at random from
   [1..N] to your filter.
  - return the bloom filter
"""


def bf_performance_helper(N: int, r: int, capacity: int, hash_func=None):
    S = np.random.choice(np.arange(N), size=r, replace=False)
    bf = SimpleBloomFilter(capacity=capacity, hash_function=hash_func)
    bf.add_all(*S)

    filled=bf.num_filled *100/capacity

    assert all(i in bf for i in S)

    items = set(S)

    false_positives=0
    for k in range(FPSize):
        test_int = np.random.randint(0, 2 ** 31 - 1)
        while test_int in items:
            test_int = np.random.randint(0, 2 ** 31 - 1)
        if test_int in bf:
            false_positives += 1
    false_positives=false_positives*100/FPSize

    return filled,false_positives


def mbf_performance_helper(N: int, r: int, capacity: int, k: int, hash_funcs=None):
    S = np.random.choice(np.arange(N), size=r, replace=False)
    bf = MultiBloomFilter(capacity=capacity, k=k, hash_functions=hash_funcs)
    bf.add_all(*S)

    filled = bf.num_filled * 100 / capacity

    assert all(i in bf for i in S)

    items = set(S)

    false_positives = 0
    for k in range(FPSize):
        test_int = np.random.randint(0, 2 ** 31 - 1)
        while test_int in items:
            test_int = np.random.randint(0, 2 ** 31 - 1)
        if test_int in bf:
            false_positives += 1
    false_positives = false_positives * 100 / FPSize

    return filled,false_positives


def co_mbf_performance_helper(N: int, r: int, capacity: int):
    S = np.random.choice(np.arange(N), size=r, replace=False)
    bf = CollisionOptimalMBF(capacity=capacity, n=r)
    bf.add_all(*S)

    filled = bf.num_filled * 100 / capacity

    assert all(i in bf for i in S)

    items = set(S)

    false_positives = 0
    for k in range(FPSize):
        test_int = np.random.randint(0, 2 ** 31 - 1)
        while test_int in items:
            test_int = np.random.randint(0, 2 ** 31 - 1)
        if test_int in bf:
            false_positives += 1
    false_positives = false_positives * 100 / FPSize

    return filled,false_positives


"""
Compare the collision probabilities for each type of filter over
different ranges of r. Each bloom filter should have (approximately) 
the same total capacity.
"""

capacity = 10000
N = 20000
step = 1000
rs = list(range(1000, N + 1, step))
print(rs)
# compute the (empirical) collision probability for each type
# of bloom filter, and plot the results (you may use matplotlib.pyplot for this)
# Use r for the x axis and collision probability for y.
# For the MultiBloomFilter, you may want to experiment with a few different values of k
# and include the results for each k value in your plot.

flbfs=[]
fpbfs=[]

flmbfs3=[]
fpmbfs3=[]

flmbfs5=[]
fpmbfs5=[]

flmbfs10=[]
fpmbfs10=[]

flcos=[]
fpcos=[]
for r in rs:
    fl_bf,fp_bf=bf_performance_helper(N,r,capacity)
    print(f" For r={r}, simple: filled rate: {fl_bf}, false positive: {fp_bf}")
    flbfs.append(fl_bf)
    fpbfs.append(fp_bf)

    fl_mbf, fp_mbf = mbf_performance_helper(N, r, capacity, 3)
    print(f" For r={r}, multi: filled rate: {fl_mbf}, false positive: {fp_mbf} k=3")
    flmbfs3.append(fl_mbf)
    fpmbfs3.append(fp_mbf)

    fl_mbf,fp_mbf=mbf_performance_helper(N,r,capacity,5)
    print(f" For r={r}, multi: filled rate: {fl_mbf}, false positive: {fp_mbf} k=5")
    flmbfs5.append(fl_mbf)
    fpmbfs5.append(fp_mbf)

    fl_mbf, fp_mbf = mbf_performance_helper(N, r, capacity, 10)
    print(f" For r={r}, multi: filled rate: {fl_mbf}, false positive: {fp_mbf} k=10")
    flmbfs10.append(fl_mbf)
    fpmbfs10.append(fp_mbf)

    fl_co,fp_co=co_mbf_performance_helper(N,r,capacity)
    print(f" For r={r}, collision: filled rate: {fl_co}, false positive: {fp_co}")
    flcos.append(fl_co)
    fpcos.append(fp_co)

plt.title("False Positive Rate v.s. Insert Size")
plt.xlabel("Insert Size")
plt.ylabel("False Positive Rate (%)")
plt.plot(rs, fpbfs, label = "SBF")
plt.plot(rs, fpmbfs3,label = "MBF k=3" )
plt.plot(rs, fpmbfs5,label = "MBF k=5" )
plt.plot(rs, fpmbfs10,label = "MBF k=10" )
plt.plot(rs, fpcos, label = "CO-MBF")
plt.legend()
plt.savefig("./collision_comparison.png")
plt.show()

# Leave a comment explaining your findings below
'''
With the increase of size of insert number, the False Positive Rate of multi_BF 
and simple_BF always increase. 
However, with the optimal k, collision optimal BF shows a much better result than 
normally multi_BF. which means the good choice of k is in great significance.
'''



"""
Repeat the experiment plotting the number of slots filled for each type of filter
"""
plt.title("Number Filled Rate v.s. Insert Size")
plt.xlabel("Insert Size")
plt.ylabel("Number Filled Rate (%)")
plt.plot(rs, flbfs, label = "SBF")
plt.plot(rs, flmbfs3,label = "MBF k=3" )
plt.plot(rs, flmbfs5,label = "MBF k=5" )
plt.plot(rs, flmbfs10,label = "MBF k=10" )
plt.plot(rs, flcos, label = "CO-MBF")
plt.legend()
plt.savefig("./fill_comparison.png")
plt.show()


# Leave a comment explaining your findings below -
# Is this what you expect after seeing the results from the first plot?
'''
There is a trend that the higher the slots filled the higher the false positive will be.
This is reasonable. However with a optimal k, the number filled rate is not always increase
with the increase of insert size. which shows great help by good choice of k.
'''