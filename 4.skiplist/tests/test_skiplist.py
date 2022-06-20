import random
from skip_list import SkipList


def test_skiplist(inputs):
    height, p, n_items = inputs

    # Make a skip list with specified height and bubble-up probability
    skip_list = SkipList(height=height, p=p)
    items_to_add = [random.randint(0, 2 ** (height + 1)) for _ in range(n_items)]
    items_to_add = list(set(items_to_add))

    # Add all the items
    for item in items_to_add:
        skip_list.add(item)
        assert item in skip_list

    assert skip_list.num_vals == len(set(items_to_add))
    assert skip_list.num_nodes >= len(set(items_to_add))

    num_added = skip_list.num_vals

    for i, item_to_remove in enumerate(items_to_add):
        skip_list.remove(item_to_remove)
        assert item_to_remove not in skip_list
        assert skip_list.num_vals == num_added - i - 1
        for remaining_item in items_to_add[i + 1 :]:
            assert remaining_item in skip_list

    assert skip_list.num_nodes == 0
    assert skip_list.num_vals == 0
