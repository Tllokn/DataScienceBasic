from btree import BTree


def test_btree():
    test_keys = [20, 10, 5, 17, 22, 24, 29, 25]
    bt = BTree(min_degree=2)
    for key in test_keys:
        bt.insert(key)
        assert bt.is_valid

    assert list(bt.in_order()) == sorted(test_keys)
