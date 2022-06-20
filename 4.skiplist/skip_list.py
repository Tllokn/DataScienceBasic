from __future__ import annotations

from math import inf

import numpy as np

from linked_list import LinkedList, LinkedListNode


class SkipList:
    """Skip list data structure, implemented as a 2D linked list."""

    def __init__(self, height: int, p: float):
        """Constructs the skip list.

        Parameters
        ----------
        height : int
            The number of layers/rows in the skip list.
        p : float
            The probability that an element will also appear in the layer above.
        """
        assert height > 0
        self.p: float = p
        self.height: int = height
        self.structure: LinkedList[float] = LinkedList(
            *self.init_first_and_last_cols()
        )

        # The number of unique values in the skip list
        self.num_vals: int = 0
        # The number of nodes in the skip list
        self.num_nodes: int = 0

    def init_first_and_last_cols(
            self,
    ) -> tuple[LinkedListNode[float], LinkedListNode[float]]:
        """Initialize the leftmost and rightmost columns of the skip list.

        Returns
        -------
        tuple[LinkedListNode[float], LinkedListNode[float]]
            The list nodes corresponding to the head and the tail of the linked list
            (the nodes in the upper-left and upper-right corners).
        """
        # leftmost column should consist of nodes with value -inf
        left_col = [LinkedListNode(-inf) for _ in range(self.height)]
        for a, b in zip(left_col[:-1], left_col[1:]):
            self.attach_rows([a], [b])

        # rightmost column should consist of nodes with value inf
        right_col = [LinkedListNode(inf) for _ in range(self.height)]
        for a, b in zip(right_col[:-1], right_col[1:]):
            self.attach_rows([a], [b])

        self.attach_columns(left_col, right_col)

        # return the heads to be used as the head and tail of the 2D linked list
        return left_col[0], right_col[0]

    @staticmethod
    def attach_rows(
            above_row: list[LinkedListNode[float]],
            below_row: list[LinkedListNode[float]],
    ) -> None:
        """Given two lists of LinkedListNodes, attach the two rows so that
        the ith node on the top points down to the ith node on the bottom and
        vice versa.

        Parameters
        ----------
        above_row : list[LinkedListNode[float]]
            The upper row to attach.
        below_row : list[LinkedListNode[float]]
            The lower row to attach.
        """

        assert len(above_row) == len(below_row)
        for above, below in zip(above_row, below_row):
            above.below = below
            below.above = above

    @staticmethod
    def attach_columns(
            left_col: list[LinkedListNode[float]],
            right_col: list[LinkedListNode[float]],
    ) -> None:
        """Given two lists of LinkedListNodes, attach the two columns so that
        the ith node on the left points right to the ith node on the right and
        vice versa.

        Parameters
        ----------
        left_col : list[LinkedListNode[float]]
            The left column to attach.
        right_col : list[LinkedListNode[float]]
            The right column to attach.
        """
        assert len(left_col) == len(right_col)
        for (left, right) in zip(left_col, right_col):
            left.right = right
            right.left = left

    def row_insert(
            self,
            to_insert: LinkedListNode[float],
            node_left: LinkedListNode[float],
    ) -> None:
        """Insert a node to the right of node_left. Does not update the node's
        above or below elements.

        Parameters
        ----------
        to_insert : LinkedListNode[float]
            The node to insert.
        node_left : LinkedListNode[float]
            The node that will be directly to the left of the inserted node.
        """
        right = node_left.right
        self.attach_columns([node_left], [to_insert])
        if right:
            self.attach_columns([to_insert], [right])

    def row_delete(self, node: LinkedListNode[float]) -> None:
        """Delete a node from a given row. Does not update the node's
        above or below elements.

        Parameters
        ----------
        node : LinkedListNode[float]
            The node that will be deleted.
        """
        left, right = node.left, node.right
        assert left is not None and right is not None
        left.right = right
        right.left = left

    def search(self, val: float) -> list[LinkedListNode[float]]:
        """Search for the value in the skip list and return the largest
        value node in each layer/row whose value is less than or equal to val.

        Example:

        -inf      17                                  inf
        -inf      17      25                      55  inf
        -inf      17      25  31                  55  inf
        -inf  12  17      25  31  38      44      55  inf
        -inf  12  17  20  25  31  38  39  44  50  55  inf

        Running self.search(50) with the above skip list would return a list of
        five nodes, one from each layer, containing [17, 25, 31, 44, 50] respectively.

        Parameters
        ----------
        val : float
            The value to search for.

        Returns
        -------
        list[LinkedListNode[float]]
            A list of nodes where each node contains the largest value <= val in a given row.
        """
        path = []
        curr_node = self.structure.head
        curr_row = 0
        while curr_row < self.height:
            if curr_node.right.data > val:
                path.append(curr_node)
                curr_row += 1
                curr_node = curr_node.below
            else:
                # path.append(curr_node.right)
                curr_node=curr_node.right

        return path

    def find(self, val: float) -> LinkedListNode[float] | None:
        """Locate the node in the bottom layer of the skip list containing the
        desired value. If no such node exists, return None.

        Parameters
        ----------
        val : float
            The value to find in the skip list.

        Returns
        -------
        LinkedListNode[float] | None
            The node in the bottom layer of the skip list that contains val.
            None if no such node exists.
        """
        path = self.search(val)
        if path[-1].data == val:
            return path[-1]
        else:
            return None

    def add(self, val: float) -> None:
        """Add a new value to the skip list and update the data structure's
        internal counts (self.num_vals and self.num_nodes).

        Parameters
        ----------
        val : float
            The value to add to the skip list.
        """
        insert_positions = self.search(val)
        # such value already exist
        # if val == insert_positions[-1].data:
        #     return
        # add new
        self.num_vals += 1
        curr_row = self.height - 1

        newnode = None
        up = True

        while curr_row >= 0 and up:
            if newnode is None:
                newnode = LinkedListNode(val)
                # print(f"Create new node with value:{val}")
            up_node = LinkedListNode(val)
            l = insert_positions.pop(-1)

            self.row_insert(newnode,l)
            # r = l.right
            self.num_nodes += 1

            # l.right = newnode
            # newnode.left = l
            #
            # r.left = newnode
            # newnode.right = r

            up = np.random.choice([True, False], p=[self.p, 1 - self.p])

            # move up, find next place to insert x
            if up:
                # print("Move Up")
                # print(f"current row {curr_row}")
                curr_row -= 1
                newnode.above = up_node
                up_node.below = newnode
                newnode = up_node
            else:
                # print(f"finish insert {val}")
                return

    def remove(self, val: float) -> None:
        """Remove all nodes with the given value from the skip list and update
        the data structure's internal counts (self.num_vals and self.num_nodes).

        Parameters
        ----------
        val : float
            The value to remove from the skip list.
        """
        if val not in self:
            return
        path = self.search(val)
        self.num_vals -= 1

        # for node in path:
        #     if node.data==val:
        #         self.num_nodes -= 1
        #         self.row_delete(node)


        for node in path:
            # print(node.data)
            if node.data == val:
                # first show up
                curr_node = node
                break

        while curr_node is not None and curr_node.data==val:
            # l = curr_node.left
            # r = curr_node.right
            #
            # l.right = r
            # r.left = l

            self.row_delete(curr_node)

            if curr_node.right.data == val:
                curr_node=curr_node.right
            else:
                curr_node = curr_node.below
            self.num_nodes -= 1

    def __contains__(self, val: float) -> bool:
        """Check to see if a value is in the skip list.

        Parameters
        ----------
        val : float
            The value to find in the skip list.

        Returns
        -------
        bool
            Whether or not the item is in the skip list.
        """
        if self.find(val) is not None:
            return True
        else:
            return False
