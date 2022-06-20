# Tree Design

In `main.py`, design a modified RB tree which stores a collection of integers
and supports the following operations:

- `add_range(lo, hi)` marks all integers in the range `[lo, hi]` (inclusive) as valid
- `is_valid(x)` answers whether `x` is marked as valid

By default, all integers are invalid (if no ranges have been added).

Suppose that there are `n` operations of the first type 
and `k` of the second type. 
The two operations may be interleaved arbitrarily.
The total runtime of all operations should be `O(n log n + k log n)`.

You may use / modify the provided `RedBlackTree` class as needed.
You may also define a custom Node class or additional helper functions.
