"""
RB tree implementation
Modified from https://algorithmtutor.com/Data-Structures/Tree/Red-Black-Trees/
"""

import sys

# class Interval:
#
#     def __init__(self,low=0,high=0):
#         self.low=low
#         self.high=high
#
#     def __eq__(self, other):
#         # Weak equality check
#         if not isinstance(other, Interval):
#             return False
#         return (other.low == self.low and other.high==self.high)


class Node:
    """Data structure that represents a node in the tree"""
    _id = 0

    def __init__(self, interval):
        self.data = interval  # holds the key
        self.parent = None  # pointer to the parent
        self.left = None  # pointer to left child
        self.right = None  # pointer to right child
        self.color = 1  # 1 . Red, 0 . Black
        self.id = Node._id
        Node._id += 1

    def __eq__(self, other):
        # Weak equality check
        if not isinstance(other, Node):
            return False
        return other.id == self.id

    def __ne__(self, other):
        # Weak inequality check
        if not isinstance(other, Node):
            return True
        return other.id != self.id


class RedBlackTree:
    """Red-black tree implementation"""

    def __init__(self, node_cls=Node, tnull_val=0):
        self.node_cls = node_cls
        self.TNULL = self.node_cls(tnull_val)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL

    def __pre_order_helper(self, node, display=False):
        if node == self.TNULL:
            return
        if display:
            sys.stdout.write(node.data + " ")
        self.__pre_order_helper(node.left, display=display)
        self.__pre_order_helper(node.right, display=display)

    def __in_order_helper(self, node, display=False):
        if node == self.TNULL:
            return
        self.__in_order_helper(node.left, display=display)
        if display:
            sys.stdout.write(node.data + " ")
        self.__in_order_helper(node.right, display=display)

    def __post_order_helper(self, node, display=False):
        if node == self.TNULL:
            return
        self.__post_order_helper(node.left, display=display)
        self.__post_order_helper(node.right, display=display)
        if display:
            sys.stdout.write(node.data + " ")

    def __search_tree_helper(self, node, key):
        if node == self.TNULL or key == node.data:
            return node

        if key[0] < node.data[0] :
            return self.__search_tree_helper(node.left, key)
        return self.__search_tree_helper(node.right, key)

    # fix the rb tree modified by the delete operation
    def __fix_delete(self, x):
        while x != self.root and x.color == 0:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == 1:
                    # case 3.1
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    s = x.parent.right

                if s.left.color == 0 and s.right.color == 0:
                    # case 3.2
                    s.color = 1
                    x = x.parent
                else:
                    if s.right.color == 0:
                        # case 3.3
                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s = x.parent.right

                    # case 3.4
                    s.color = x.parent.color
                    x.parent.color = 0
                    s.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 1:
                    # case 3.1
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    s = x.parent.left

                if s.left.color == 0 and s.right.color == 0:
                    # case 3.2
                    s.color = 1
                    x = x.parent
                else:
                    if s.left.color == 0:
                        # case 3.3
                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s = x.parent.left

                        # case 3.4
                    s.color = x.parent.color
                    x.parent.color = 0
                    s.left.color = 0
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 0

    def __rb_transplant(self, x, y):
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            assert x == x.parent.right
            x.parent.right = y
        y.parent = x.parent

    def __delete_node_helper(self, node, key):
        # find the node containing key
        z = self.TNULL
        while node != self.TNULL:
            if node.data == key:
                z = node

            if node.data[0] <= key[0]:
                node = node.right
            else:
                node = node.left

        if z == self.TNULL:
            raise ValueError(f"Couldn't find key {key} in the tree")

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif z.right == self.TNULL:
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == 0:
            self.__fix_delete(x)

    # fix the red-black tree
    def __fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # uncle
                if u.color == 1:
                    # case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # case 3.2.2
                        k = k.parent
                        self.right_rotate(k)
                    # case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # uncle

                if u.color == 1:
                    # mirror case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # mirror case 3.2.2
                        k = k.parent
                        self.left_rotate(k)
                    # mirror case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    def __print_helper(self, node, indent, last):
        # print the tree structure on the screen
        if node != self.TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            s_color = "RED" if node.color == 1 else "BLACK"
            print(str(node.data) + "(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    # Pre-Order traversal
    # Node.Left Subtree.Right Subtree
    def pre_order(self):
        self.__pre_order_helper(self.root)

    # In-Order traversal
    # left Subtree . Node . Right Subtree
    def in_order(self, display=False):
        self.__in_order_helper(self.root, display=display)
        if display:
            sys.stdout.write('\n')

    # Post-Order traversal
    # Left Subtree . Right Subtree . Node
    def post_order(self, display=False):
        self.__post_order_helper(self.root, display=display)

    # search the tree for the key k
    # and return the corresponding node
    def search_tree(self, k):
        return self.__search_tree_helper(self.root, k)

    # find the node with the minimum key
    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    # find the node with the maximum key
    def maximum(self, node):
        while node.right != self.TNULL:
            node = node.right
        return node

    # find the successor of a given node
    def successor(self, x):
        # if the right subtree is not None,
        # the successor is the leftmost node in the
        # right subtree
        if x.right != self.TNULL:
            return self.minimum(x.right)

        if not x.parent:
            return None

        # else it is the lowest ancestor of x whose
        # left child is also an ancestor of x.
        y = x.parent
        while y and y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    # find the predecessor of a given node
    def predecessor(self, x):
        # if the left subtree is not None,
        # the predecessor is the rightmost node in the
        # left subtree
        if x.left != self.TNULL:
            return self.maximum(x.left)

        if not x.parent:
            return None

        y = x.parent
        while y and y != self.TNULL and x == y.left:
            x = y
            y = y.parent

        return y

    # rotate left at node x
    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    # rotate right at node x
    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    # insert the key to the tree in its appropriate position
    # and fix the tree
    def insert(self, key):
        # Ordinary Binary Search Insertion
        node = self.node_cls(key)
        node.parent = None
        node.data = key
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = 1  # new node must be red

        y = None
        x = self.root

        flag = True

        while x != self.TNULL:
            y = x
            #node range contain x range
            if node.data[0]<x.data[0] and node.data[1]>x.data[1]:
                x.data=node.data
                flag=False
                break
            elif node.data[0] < x.data[0]:
                x = x.left
            elif node.data[1] > x.data[1]:
                x = x.right
            else:
                #x range contain node range
                flag=False
                break

        if flag:
            # y is parent of x
            node.parent = y
            if y is None:
                self.root = node
            elif node.data[0] < y.data[0]:
                y.left = node
            else:
                y.right = node

        # if new node is a root node, simply return
        if node.parent is None:
            node.color = 0
            return

        # if the grandparent is None, simply return
        if node.parent.parent is None:
            return

        # Fix the tree
        self.__fix_insert(node)

    def get_root(self):
        return self.root

    # delete the node from the tree
    def delete(self, key):
        self.__delete_node_helper(self.root, key)

    # print the tree structure on the screen
    def pretty_print(self):
        self.__print_helper(self.root, "", True)


if __name__ == "__main__":
    import numpy as np

    keys = np.random.randint(0, 10000, size=40)
    bst = RedBlackTree()
    added = [k for k in keys]
    for key in keys:
        bst.insert(key)
    bst.pretty_print()
    removed = np.random.choice(keys, size=20, replace=False)
    print(removed)
    for key in removed:
        bst.delete(key)
    for key in removed:
        bst.insert(key)
