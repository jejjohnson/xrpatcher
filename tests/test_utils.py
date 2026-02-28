import pytest
import xarray as xr

from xrpatcher._src.utils import (
    check_lists_equal,
    check_lists_subset,
    get_dims_xrda,
    get_patches_size,
    get_slices,
    update_dict_keys,
)


@pytest.mark.parametrize(
    "list1,list2",
    [
        ([1, 2, 3, 5], [1, 3, 2, 5]),
        ([1, 3, 2, 5], [1, 3, 2, 5]),
        ([5, 3, 2, 1], [1, 3, 2, 5]),
    ],
)
def test_check_lists_equal_true(list1, list2):
    check_lists_equal(list1, list2)
    check_lists_equal(list2, list1)


@pytest.mark.parametrize(
    "list1,list2",
    [
        ([1, 2, 3, 5], [1, 3, 2]),
        ([1, 2, 3, 5], [1, 3, 2, 6]),
        ([1, 2, 3, 5], [1, 2, 3, 4]),
    ],
)
def test_check_lists_equal_false(list1, list2):
    with pytest.raises(AssertionError):
        check_lists_equal(list1, list2)
        check_lists_equal(list2, list1)


@pytest.mark.parametrize(
    "list1,list2",
    [
        ([1, 2, 3, 5], []),
        (
            [1, 2, 3, 5],
            [
                1,
            ],
        ),
        ([1, 2, 3, 5], [1, 2]),
        ([1, 2, 3, 5], [1, 2, 3]),
        ([1, 2, 3, 5], [1, 2, 3, 5]),
    ],
)
def test_check_lists_subset_true(list1, list2):
    check_lists_subset(list2, list1)


@pytest.mark.parametrize(
    "list1,list2",
    [
        ([1, 2, 3, 5], [6]),
        ([1, 2, 3, 5], [1, 6]),
        ([1, 2, 3, 5], [1, 2, 6]),
        ([1, 2, 3, 5], [1, 2, 3, 4]),
    ],
)
def test_check_lists_subset_false(list1, list2):
    with pytest.raises(AssertionError):
        check_lists_subset(list2, list1)


@pytest.mark.parametrize(
    "source,new,correct",
    [
        ({"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"y": 1, "x": 1}, {"x": 1, "y": 1}),
        ({"x": 10, "y": 100}, {"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"x": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"y": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"x": 10}, {"x": 10, "y": 1}),
        ({"x": 50, "y": 100}, {"x": 10}, {"x": 10, "y": 1}),
    ],
)
def test_update_dict_keys_not_default(source, new, correct):
    new_update = update_dict_keys(source, new, False)

    assert new_update == correct


@pytest.mark.parametrize(
    "source,new,correct",
    [
        ({"x": 1, "y": 1}, {"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"y": 1, "x": 1}, {"x": 1, "y": 1}),
        ({"x": 10, "y": 100}, {"x": 1, "y": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"x": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"y": 1}, {"x": 1, "y": 1}),
        ({"x": 1, "y": 1}, {"x": 10}, {"x": 10, "y": 1}),
        ({"x": 50, "y": 100}, {"x": 10}, {"x": 10, "y": 100}),
    ],
)
def test_update_dict_keys_default(source, new, correct):
    new_update = update_dict_keys(source, new, True)

    assert new_update == correct


@pytest.mark.parametrize(
    "dims,patches,strides,correct",
    [
        ({"x": 10}, {}, {}, {"x": 1}),
        ({"x": 10}, {"x": 1}, {}, {"x": 10}),
        ({"x": 10}, {}, {"x": 1}, {"x": 1}),
        ({"x": 10}, {"x": 2}, {}, {"x": 9}),
        ({"x": 10}, {"x": 3}, {}, {"x": 8}),
        ({"x": 10}, {"x": 4}, {}, {"x": 7}),
        ({"x": 10}, {}, {"x": 2}, {"x": 1}),
        ({"x": 10}, {}, {"x": 3}, {"x": 1}),
        ({"x": 10}, {}, {"x": 4}, {"x": 1}),
        ({"x": 10}, {"x": 2}, {"x": 2}, {"x": 5}),
        ({"x": 10}, {"x": 3}, {"x": 3}, {"x": 3}),
    ],
)
def test_get_patches_size(dims, patches, strides, correct):
    dims_size, patches, strides = get_patches_size(dims, patches, strides)

    msg = f"Dims: {dims} | Patches: {patches} | Strides: {strides} | Dims: {dims_size}"
    assert dims_size == correct, msg


@pytest.mark.parametrize(
    "dims,patches,strides,correct",
    [
        ({"x": 10, "y": 20}, {"x": 4, "y": 5}, {"x": 2, "y": 5}, {"x": 4, "y": 4}),
        ({"x": 10, "y": 20}, {"x": 2, "y": 4}, {"x": 2, "y": 4}, {"x": 5, "y": 5}),
        ({"x": 12, "y": 8}, {"x": 4, "y": 4}, {"x": 4, "y": 2}, {"x": 3, "y": 3}),
    ],
)
def test_get_patches_size_multi_dim(dims, patches, strides, correct):
    dims_size, _, _ = get_patches_size(dims, patches, strides)
    assert dims_size == correct


def test_get_dims_xrda():
    import numpy as np
    from collections import OrderedDict

    da = xr.DataArray(
        np.ones((5, 10, 3)),
        dims=["x", "y", "z"],
        coords={"x": range(5), "y": range(10), "z": range(3)},
    )
    result = get_dims_xrda(da)
    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == ["x", "y", "z"]
    assert result["x"] == 5
    assert result["y"] == 10
    assert result["z"] == 3


@pytest.mark.parametrize(
    "idx,da_size,patches,strides,expected",
    [
        (
            0,
            {"x": 5},
            {"x": 3},
            {"x": 2},
            {"x": slice(0, 3)},
        ),
        (
            1,
            {"x": 5},
            {"x": 3},
            {"x": 2},
            {"x": slice(2, 5)},
        ),
        (
            0,
            {"x": 3, "y": 4},
            {"x": 2, "y": 2},
            {"x": 1, "y": 1},
            {"x": slice(0, 2), "y": slice(0, 2)},
        ),
        (
            1,
            {"x": 3, "y": 4},
            {"x": 2, "y": 2},
            {"x": 1, "y": 1},
            {"x": slice(0, 2), "y": slice(1, 3)},
        ),
    ],
)
def test_get_slices_basic(idx, da_size, patches, strides, expected):
    result = get_slices(idx=idx, da_size=da_size, patches=patches, strides=strides)
    assert result == expected


def test_get_slices_last_item():
    import numpy as np

    # da_size is the number of patches per dimension (not raw data size)
    # 10 patches in x, 5 patches in y → 50 total patches, last valid index is 49
    da_size = {"x": 10, "y": 5}
    patches = {"x": 3, "y": 2}
    strides = {"x": 2, "y": 1}
    last_idx = int(np.prod(list(da_size.values()))) - 1  # 49

    result = get_slices(idx=last_idx, da_size=da_size, patches=patches, strides=strides)
    # unravel_index(49, (10, 5)) == (9, 4)
    assert result == {"x": slice(18, 21), "y": slice(4, 6)}
