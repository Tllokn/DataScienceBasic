from __future__ import annotations

from functools import cached_property
from typing import Iterable

from shapely.geometry.base import BaseGeometry


class Node:
    def __init__(self, min_degree: int, parent: InternalNode | None = None):
        self.min_degree = min_degree
        self.parent = parent
        self.bbox = BaseGeometry()

    def update_ancestors(self):
        """Update the bounding boxes of each ancestor to reflect changes to self.bbox"""
        # TODO
        parent = self.parent
        while parent is not None:
            parent.bbox = parent.bbox.union(self.bbox).envelope
            parent = parent.parent

    @property
    def max_degree(self) -> int:
        return 2 * self.min_degree

    @property
    def is_leaf(self) -> bool:
        raise NotImplementedError

    @property
    def is_full(self) -> bool:
        raise NotImplementedError

    def split(self, payload):
        raise NotImplementedError


class LeafNode(Node):
    """A leaf node stores some geometries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometries: list[BaseGeometry] = []

    @cached_property
    def is_leaf(self) -> bool:
        return True

    @property
    def is_full(self) -> bool:
        return len(self.geometries) >= self.max_degree

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(str(geom) for geom in self.geometries)})"

    def add(self, payload: BaseGeometry):
        """Adds the given geometry to this leaf node."""
        if self.is_full:
            self.split(payload)
        else:
            self.geometries.append(payload)
        self.bbox = self.bbox.union(payload).envelope
        self.update_ancestors()

    def split(self, payload: BaseGeometry):
        """Splits this leaf node into two new leaf nodes."""
        # TODO
        # Find the two geometries which are farthest apart (including the payload).
        all_geo = self.geometries.copy()
        all_geo.append(payload)
        distance = -1.0
        far1=BaseGeometry()
        far2=BaseGeometry()
        for i in all_geo:
            for j in all_geo:
                dis = i.distance(j)
                if dis > distance:
                    distance = dis
                    far1 = i
                    far2 = j

        # Create two new leaf nodes, seeded with these geometries.
        new_node1 = LeafNode(min_degree=self.min_degree)
        new_node2 = LeafNode(min_degree=self.min_degree)

        new_node1.add(far1)
        new_node2.add(far2)

        all_geo.remove(far1)
        all_geo.remove(far2)

        # Split the remaining geometries between these two nodes.
        # Assign each to the node requiring the minimum enlargement to accommodate it.
        # Ensure that both new nodes have at least self.min_degree geometries.
        for re in all_geo:
            change1 = new_node1.bbox.union(re).envelope.area - new_node1.bbox.area
            change2 = new_node2.bbox.union(re).envelope.area - new_node2.bbox.area
            if len(new_node1.geometries) > self.min_degree:
                new_node2.add(re)
            elif len(new_node2.geometries) > self.min_degree:
                new_node1.add(re)
            elif change1 < change2:
                new_node1.add(re)
            else:
                new_node2.add(re)
        # Remove old node from parent and add new nodes.
        new_node1.parent = self.parent
        new_node2.parent = self.parent
        # print(type(self.parent))
        self.parent.remove(self)
        self.parent.add(new_node1)
        self.parent.add(new_node2)



class InternalNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children: list[Node] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bbox={self.bbox}, children={len(self.children)})"

    @cached_property
    def is_leaf(self) -> bool:
        return False

    @property
    def is_full(self) -> bool:
        return len(self.children) >= self.max_degree

    def add(self, payload: Node):
        """Adds the given node to this internal node."""
        if self.is_full:
            self.split(payload)
        else:
            self.children.append(payload)
        self.bbox = self.bbox.union(payload.bbox).envelope
        self.update_ancestors()

    def remove(self, child: Node):
        self.children.remove(child)

    def split(self, payload: Node):
        """Splits this internal node into two new internal nodes."""
        # TODO
        # Find the two nodes which are farthest apart (including the payload).
        all_child = self.children.copy()
        all_child.append(payload)
        distance = -1.0
        far1=None
        far2=None
        for i in all_child:
            for j in all_child:
                dis = i.bbox.distance(j.bbox)
                if dis > distance:
                    distance = dis
                    far1 = i
                    far2 = j

        # Create two new internal nodes, seeded with these children.
        new_node1 = InternalNode(min_degree=self.min_degree)
        new_node2 = InternalNode(min_degree=self.min_degree)

        new_node1.add(far1)
        new_node2.add(far2)

        all_child.remove(far1)
        all_child.remove(far2)

        # Split the remaining children between these two nodes.
        # Assign each child to the node requiring the minimum enlargement to accommodate it.
        # Ensure that both new nodes have at least self.min_degree children.
        for re in all_child:
            change1 = new_node1.bbox.union(re.bbox).area - new_node1.bbox.area
            change2 = new_node2.bbox.union(re.bbox).area - new_node2.bbox.area
            if len(new_node1.children) > self.min_degree:
                new_node2.add(re)
            elif len(new_node2.children) > self.min_degree:
                new_node1.add(re)
            elif change1 < change2:
                new_node1.add(re)
            else:
                new_node2.add(re)
        # Remove old node from parent and add new nodes.
        new_node1.parent = self.parent
        new_node2.parent = self.parent
        self.parent.add(new_node1)
        self.parent.add(new_node2)
        self.parent.remove(self)


class RTree:
    def __init__(self, min_degree: int):
        self.min_degree = min_degree
        self.root: Node = LeafNode(min_degree=min_degree)

    def insert(self, payload: BaseGeometry):
        """Inserts the given geometry into the tree."""
        # TODO

        # If root is full, create a new root
        if self.root.is_full:
            old_root = self.root
            self.root = InternalNode(min_degree=self.min_degree)
            # Add the old root as a child
            old_root.parent = self.root
            self.root.add(old_root)
        # Split the old root using the payload (wrapped in a new leaf node if needed)


        # Otherwise, descend until reaching a leaf
        # At each level, select the child that requires the minimum enlargement to cover the payload
        target_node=self.root
        while not target_node.is_leaf:
            change=float('inf')
            for ele in target_node.children:
                curr_change=ele.bbox.union(payload).envelope.area-ele.bbox.area
                if curr_change < change:
                    change=curr_change
                    close=ele
            target_node=close

        # Once you reach a leaf, add the payload geometry
        target_node.add(payload)

    def search(self, query: BaseGeometry, start_node: Node | None = None) -> Iterable[BaseGeometry]:
        """Searches the tree for geometries contained by the query geometry."""
        if start_node is None:
            start_node = self.root

        res = []

        # TODO: Search recursively. Only check subtrees which intersect the query region.
        # When reaching a leaf, yield any geometry contained in the query region.

        if start_node.is_leaf:
            for region in start_node.geometries:
                if query.contains(region):
                    res.append(region)

        if not start_node.is_leaf:
            for child in start_node.children:
                if query.intersects(child.bbox):
                    res.extend(self.search(query,child))

        return res


    def display(self, level=0, node: Node | None = None):
        """Prints the tree structure to the command line."""
        if node is None:
            node = self.root

        print(f"{'|   ' * level}|-- {node}")
        if not node.is_leaf:
            for child in node.children:
                self.display(level=level + 1, node=child)


if __name__ == "__main__":
    import random

    from shapely.geometry import Point, Polygon

    random.seed(6)

    rtree = RTree(min_degree=2)
    num_points = 12
    for _ in range(num_points):
        x = round(random.random(), 2)
        y = round(random.random(), 2)
        pt = Point(x, y)
        rtree.insert(pt)
        rtree.display()

    polygon = Polygon([(1, 1), (1, 0.5), (0.5, 0.5), (0.5, 1)])
    results = list(rtree.search(polygon))
    assert all(polygon.contains(result) for result in results)
    import pprint

    pprint.pprint([str(res) for res in results])
