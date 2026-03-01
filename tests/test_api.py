"""Public API smoke tests for XRDAPatcher."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
from xarray_dataclasses import Coord, Data, Name, asdataarray

from xrpatcher._src.base import XRDAPatcher


X = Literal["x"]
Y = Literal["y"]


@dataclass
class Variable1D:
    data: Data[tuple[X], np.float32]
    x: Coord[X, np.float32] = 0
    name: Name[str] = "var"


@dataclass
class Variable2D:
    data: Data[tuple[X, Y], np.float32]
    x: Coord[X, np.float32] = 0
    y: Coord[Y, np.float32] = 0
    name: Name[str] = "var"


@pytest.fixture
def patcher_1d():
    coord = np.arange(0, 20, 1)
    data = np.ones(len(coord), dtype=np.float32)
    da = Variable1D(data=data, x=coord)
    da = asdataarray(da)
    return XRDAPatcher(da=da, patches={"x": 5}, strides={"x": 5}, check_full_scan=True)


@pytest.fixture
def patcher_2d():
    x = np.arange(0, 12, 1)
    y = np.arange(0, 12, 1)
    data = np.ones((len(x), len(y)), dtype=np.float32)
    da = Variable2D(data=data, x=x, y=y)
    da = asdataarray(da)
    return XRDAPatcher(
        da=da,
        patches={"x": 4, "y": 4},
        strides={"x": 4, "y": 4},
        check_full_scan=True,
    )


def test_repr_contains_expected_strings(patcher_1d):
    """repr(patcher) contains 'XArray Patcher', 'Patches', 'Strides'."""
    r = repr(patcher_1d)
    assert "XArray Patcher" in r
    assert "Patches" in r
    assert "Strides" in r


def test_str_contains_expected_strings(patcher_1d):
    """str(patcher) contains 'XArray Patcher', 'Patches', 'Strides'."""
    s = str(patcher_1d)
    assert "XArray Patcher" in s
    assert "Patches" in s
    assert "Strides" in s


def test_coord_names_property(patcher_2d):
    """patcher.coord_names returns the correct list of dimension names."""
    assert patcher_2d.coord_names == ["x", "y"]


def test_get_coords_returns_correct_datasets(patcher_2d):
    """get_coords() returns a list of length len(patcher), each with correct dim keys."""
    coords = patcher_2d.get_coords()
    assert len(coords) == len(patcher_2d)
    expected_keys = set(patcher_2d.patches.keys())
    for coord in coords:
        assert set(coord.dims.keys()) == expected_keys
