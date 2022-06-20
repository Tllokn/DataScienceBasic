import numpy as np

from skip_list import SkipList

if __name__ == '__main__':
    height, p = 10, 0.5
    n_items = 200

    # Make a skip list with specified height and bubble-up probability
    skip_list = SkipList(height=height, p=p)
    print("Construct Complete")
    items_to_add = np.random.randint(0, 2 ** (height + 1), n_items)
    items_to_add = [x for x in set(items_to_add)]

    # Add all the items
    for item in items_to_add:
        # print(f"add {item}")
        skip_list.add(item)
        # print("finish insert")
        assert item in skip_list

    assert skip_list.num_vals == len(set(items_to_add))

    print("skip list num_vals:", skip_list.num_vals)
    print("skip list num_nodes:", skip_list.num_nodes)
    ratio = skip_list.num_nodes / skip_list.num_vals
    print("ratio:", ratio)
    print("actual num_vals:", len(items_to_add))

    num_added = skip_list.num_vals

    for i, item_to_remove in enumerate(items_to_add):
        skip_list.remove(item_to_remove)
        print("removed item:", item_to_remove)
        assert item_to_remove not in skip_list
        print("skip list num_vals:", skip_list.num_vals)
        print("skip list num nodes:", skip_list.num_nodes)
        assert skip_list.num_vals == num_added - i - 1
        for remaining_item in items_to_add[i + 1 :]:
            assert remaining_item in skip_list
