"""Edge case and data-integrity tests for XRDAPatcher."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from xarray_dataclasses import Coord, Data, Name, asdataarray

from xrpatcher._src.base import XRDAPatcher


RNG = np.random.RandomState(seed=42)

X = Literal["x"]


@dataclass
class Variable1D:
    data: Data[tuple[X], np.float32]
    x: Coord[X, np.float32] = 0
    name: Name[str] = "var"


def _make_da(size: int):
    coord = np.arange(0, size, 1)
    data = RNG.randn(size).astype(np.float32)
    da = Variable1D(data=data, x=coord)
    return asdataarray(da)


def test_patch_equals_full_data_size():
    """Single patch covering the entire array → len==1 and reconstruction matches."""
    da = _make_da(10)
    patcher = XRDAPatcher(
        da=da, patches={"x": 10}, strides={"x": 10}, check_full_scan=True
    )
    assert len(patcher) == 1

    items = list(map(lambda x: x.data, patcher))
    rec = patcher.reconstruct(items, dims_labels=["x"], weight=None)
    np.testing.assert_array_almost_equal(rec.data, da.data)


def test_stride_equals_one():
    """Stride of 1 creates maximum overlap; verify count and reconstruction."""
    size, patch = 10, 4
    da = _make_da(size)
    patcher = XRDAPatcher(
        da=da, patches={"x": patch}, strides={"x": 1}, check_full_scan=True
    )
    expected_count = size - patch + 1  # (10 - 4) // 1 + 1 == 7
    assert len(patcher) == expected_count

    items = list(map(lambda x: x.data, patcher))
    rec = patcher.reconstruct(items, dims_labels=["x"], weight=None)
    np.testing.assert_array_almost_equal(rec.data, da.data)


def test_data_integrity_patches_match_original():
    """Each patch's data matches da.isel(...) for the corresponding slice."""
    size, patch, stride = 20, 4, 4
    da = _make_da(size)
    patcher = XRDAPatcher(
        da=da, patches={"x": patch}, strides={"x": stride}, check_full_scan=True
    )

    for i in range(len(patcher)):
        start = i * stride
        expected = da.isel(x=slice(start, start + patch))
        np.testing.assert_array_equal(patcher[i].data, expected.data)
