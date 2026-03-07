"""Tests for indexing and iteration behaviour of XRDAPatcher."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import xarray as xr
from xarray_dataclasses import Coord, Data, Name, asdataarray

from xrpatcher._src.base import XRDAPatcher


X = Literal["x"]


@dataclass
class Variable1D:
    data: Data[tuple[X], np.float32]
    x: Coord[X, np.float32] = 0
    name: Name[str] = "var"


@pytest.fixture
def patcher_1d():
    coord = np.arange(0, 20, 1)
    data = np.arange(20, dtype=np.float32)
    da = Variable1D(data=data, x=coord)
    da = asdataarray(da)
    return XRDAPatcher(da=da, patches={"x": 4}, strides={"x": 4}, check_full_scan=True)


def test_getitem_out_of_bounds(patcher_1d):
    """Accessing patcher[len(patcher)] should raise ValueError."""
    with pytest.raises((ValueError, IndexError)):
        _ = patcher_1d[len(patcher_1d)]


def test_iter_yields_correct_count_and_order(patcher_1d):
    """Iterating yields exactly len(patcher) items that match patcher[i]."""
    items = list(patcher_1d)
    assert len(items) == len(patcher_1d)
    for i, item in enumerate(items):
        xr.testing.assert_identical(item, patcher_1d[i])


def test_iter_is_repeatable(patcher_1d):
    """Iterating twice gives the same results."""
    items1 = list(patcher_1d)
    items2 = list(patcher_1d)
    assert len(items1) == len(items2)
    for i1, i2 in zip(items1, items2, strict=True):
        xr.testing.assert_identical(i1, i2)


def test_getitem_uses_cache_when_enabled():
    """Repeated access returns the cached patch instance when cache=True."""
    coord = np.arange(0, 20, 1)
    data = np.arange(20, dtype=np.float32)
    da = Variable1D(data=data, x=coord)
    da = asdataarray(da)
    patcher = XRDAPatcher(
        da=da,
        patches={"x": 4},
        strides={"x": 4},
        check_full_scan=True,
        cache=True,
    )

    first = patcher[1]
    second = patcher[1]

    assert first is second


def test_clear_cache_forces_patch_reload(monkeypatch):
    """clear_cache() removes cached patches so the next access re-slices the data."""
    coord = np.arange(0, 20, 1)
    data = np.arange(20, dtype=np.float32)
    da = Variable1D(data=data, x=coord)
    da = asdataarray(da)
    patcher = XRDAPatcher(
        da=da,
        patches={"x": 4},
        strides={"x": 4},
        check_full_scan=True,
        cache=True,
    )

    calls = 0
    original_isel = xr.DataArray.isel

    def spy_isel(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_isel(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "isel", spy_isel)

    first = patcher[2]
    cached = patcher[2]
    patcher.clear_cache()
    second = patcher[2]

    assert first is cached
    assert first is not second
    assert calls == 2


def test_preload_loads_patch_once_on_first_access(monkeypatch):
    """preload=True loads a patch once on first access before cache hits."""
    coord = np.arange(0, 20, 1)
    data = np.arange(20, dtype=np.float32)
    da = Variable1D(data=data, x=coord)
    da = asdataarray(da)
    patcher = XRDAPatcher(
        da=da,
        patches={"x": 4},
        strides={"x": 4},
        check_full_scan=True,
        cache=True,
        preload=True,
    )

    calls = 0
    original_load = xr.DataArray.load

    def spy_load(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", spy_load)

    first = patcher[0]
    second = patcher[0]

    assert first is second
    assert calls == 1
