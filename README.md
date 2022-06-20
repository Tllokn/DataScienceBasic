# Bloom Filters

In this assignment, we will build up to a Bloom filter implementation 
which is optimized to minimize collisions.

Please complete all TODOs in the following files, in order:
- `simple_bloom_filter.py` contains the `SimpleBloomFilter` class,
which is a basic extension of the `BloomFilterABC` abstract base class
for integer elements. Unlike the standard Bloom filter implementation,
it only uses a single hash function to add and query for elements.
- `multi_bloom_filter.py` contains the `MultiBloomFilter` class, 
which consists of `k` `SimpleBloomFilter` sub-filters.
Elements are added to and queried from each sub-filter.
Queries return true if the element is found in every sub-filter.
The `capacity` provided upon initialization will be evenly distributed
among the sub-filters, which will each have capacity `capacity // k`.
- `collision_optimal_mbf.py` contains the `CollisionOptimalMBF` 
class, which is a `MultiBloomFilter` with `k` chosen to minimize 
collision probability.

After implementing these classes, complete the experiments in 
`performance.py` and analyze your results.
