from __future__ import annotations

from rb_tree import Node, RedBlackTree

class ValidNumberTree(RedBlackTree):
    def __init__(self):
        super().__init__(node_cls=Node, tnull_val=0)

    def display_in_order(self):
        self.in_order(display=True)

    def add_range(self, lo: int, hi: int):
        """Marks the given range as valid.

        Parameters
        ----------
        lo : int
            Start of range.
        hi : int
            End of range.
        """
        # TODO
        key=[lo,hi]
        self.insert(key)


    def is_valid(self, x: int) -> bool:
        """Checks if x is valid.

        Parameters
        ----------
        x : int
            Search value.

        Returns
        -------
        valid : bool
            Boolean indicating whether or not x is valid.
        """
        # TODO
        r=self.root
        while r!=self.TNULL:
            if r.data[0]<= x and x<= r.data[1]:
                return True
            elif x<r.data[0]:
                r=r.left
            else:
                r=r.right

        return False



