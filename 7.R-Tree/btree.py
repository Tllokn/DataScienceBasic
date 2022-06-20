from __future__ import annotations

from functools import cached_property
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class BTreeNode(Generic[T]):
    """A simple B-Tree Node."""

    def __init__(self, t: int, keys: list[T] | None = None):
        self.keys: list[T] = [] if keys is None else keys
        self.children: list[BTreeNode[T]] = []
        # t is the order of the parent B-Tree.
        # Nodes need this value to define max size and splitting.
        self._t = t

    @property
    def num_keys(self):
        return len(self.keys)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_full(self):
        return self.num_keys == 2 * self._t - 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.keys}, children={len(self.children)})"

    def split(self, parent: BTreeNode, payload: T) -> BTreeNode:
        """Split a node and reassign keys/children."""
        left = self
        right = self.__class__(self._t)

        mid_point = self.num_keys // 2
        split_value = self.keys[mid_point]
        parent.add_key(split_value)

        # Add keys and children to appropriate nodes
        right.children = left.children[mid_point + 1 :]
        left.children = left.children[: mid_point + 1]
        right.keys = left.keys[mid_point + 1 :]
        left.keys = left.keys[:mid_point]

        parent.children = parent.add_child(right)
        return left if payload < split_value else right

    def add_key(self, key: T):
        """Add a key to a node. The node must have room for the key."""
        if self.is_full:
            raise RuntimeError("Cannot add key to full node")
        self.keys.append(key)
        self.keys.sort()

    def add_child(self, new_node: BTreeNode):
        """
        Add a child to a node. This will sort the node's children, allowing for children
        to be ordered even after middle nodes are split.

        returns: an order list of child nodes
        """
        i = len(self.children) - 1
        while i >= 0 and self.children[i].keys[0] > new_node.keys[0]:
            i -= 1
        return self.children[: i + 1] + [new_node] + self.children[i + 1 :]


class BTree(Generic[T]):
    """CLRS-style B-tree implementation.

    Adapted from https://gist.github.com/mateor/885eb950df7231f178a5

    This implementation omits true leaves (no keys and no children).
    Instead, the nodes in the layer above are treated as leaves (no children, but at least 1 key).

    It should also support duplicate key values, although this is untested.

    Notes
    -----
    A B-tree of min-degree t >= 2 satisfies the following properties:
    - Internal nodes (excluding the root) have at least t children.
    - The root has at least 2 children
    - Every node has at most 2t children.
    - A non-leaf node with k children contains k-1 keys.
    It follows that internal nodes (except the root) contain between t-1 and 2t-1 keys.
    - All leaves appear on the same level and have no keys.
    Reference: https://sites.radford.edu/~nokie/classes/360/trees.b.tree.html

    Attributes
    ----------
    min_degree : int
        Minimum degree of the B-tree.
        Defaults to 2, which results in the 2-3-4 tree.
    root : BTreeNode
        Root node of the B-tree.
    """

    def __init__(self, min_degree: int = 2, node_cls: type[BTreeNode] = BTreeNode):
        """Creates an empty B-tree with the given min-degree."""
        if min_degree <= 1:
            raise ValueError("B-tree must have min-degree of at least 2.")
        self.min_degree = min_degree
        self.node_cls = node_cls
        self.root = node_cls(min_degree)

    @cached_property
    def max_degree(self) -> int:
        return 2 * self.min_degree

    @cached_property
    def min_keys(self) -> int:
        return self.min_degree - 1

    @cached_property
    def max_keys(self) -> int:
        return self.max_degree - 1

    @property
    def is_valid(self) -> bool:
        """Checks that all B-tree properties are satisfied"""
        return self._is_valid_helper()

    def _is_valid_helper(self, start_node: BTreeNode | None = None) -> bool:
        if start_node is None:
            root = self.root
            assert (
                root.is_leaf or len(root.children) >= 2
            ), "Root must have at least 2 children"
            start_node = root
        elif start_node.is_leaf:
            return True

        # print("Checking validity at", start_node)

        keys, children = start_node.keys, start_node.children
        num_children = len(children)
        num_children_valid = self.min_degree <= num_children <= self.max_degree
        num_children_valid |= start_node is self.root
        assert num_children_valid, "Num children not between t and 2t"
        num_keys_valid = start_node.is_leaf or num_children == len(keys) + 1
        assert num_keys_valid, "Num keys not equal to num children minus one"
        keys_sorted = sorted(keys) == keys
        assert keys_sorted, "Keys not sorted"
        bst_property_satisfied = (
            left.keys[-1] <= key <= right.keys[0]
            for left, key, right in zip(children, keys, children[1:])
        )
        assert bst_property_satisfied, "BST property is not satisfied"
        all_children_valid = all(self._is_valid_helper(child) for child in children)
        assert all_children_valid, "Children are not valid"

        return True

    def insert(self, payload: T):
        """Insert a new key of value payload into the B-Tree."""
        node = self.root
        # Root is handled explicitly since it requires creating 2 new nodes instead of the usual one.
        if node.is_full:
            new_root = self.node_cls(self.min_degree)
            new_root.children.append(self.root)
            # node is being set to the node containing the ranges we want for payload insertion.
            node = node.split(new_root, payload)
            self.root = new_root
        while not node.is_leaf:
            i = node.num_keys - 1
            while i > 0 and payload < node.keys[i]:
                i -= 1
            if payload > node.keys[i]:
                i += 1

            next = node.children[i]
            if next.is_full:
                node = next.split(node, payload)
            else:
                node = next
        # Since we split all full nodes on the way down, we can simply insert the payload in the leaf.
        node.add_key(payload)

    def search(self, value, start_node: BTreeNode | None = None):
        """Return True if the B-Tree contains a key that matches the value."""
        if start_node is None:
            start_node = self.root
        if value in start_node.keys:
            return True
        elif start_node.is_leaf:
            # If we are in a leaf, there is no more to check.
            return False
        else:
            i = 0
            while i < start_node.num_keys and value > start_node.keys[i]:
                i += 1
            return self.search(value, start_node.children[i])

    def in_order(self, node: BTreeNode | None = None) -> Iterable[T]:
        if node is None:
            node = self.root
        if node.is_leaf:
            yield from node.keys
        else:
            for child, key in zip(node.children, node.keys):
                yield from self.in_order(node=child)
                yield key
            yield from self.in_order(node=node.children[-1])

    def display(self, level=0, node: BTreeNode | None = None):
        if node is None:
            node = self.root

        print(f"{'|   ' * level}|-- {node}")
        for child in node.children:
            self.display(level=level + 1, node=child)


if __name__ == "__main__":

    test_keys = [20, 10, 5, 17, 22, 24, 29, 25]

    B = BTree(min_degree=3)

    for key in test_keys:
        print("Inserting", key)
        B.insert(key)
        B.display()
        assert B.is_valid
