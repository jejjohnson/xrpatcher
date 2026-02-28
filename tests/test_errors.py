"""Tests for error paths in XRDAPatcher."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
from xarray_dataclasses import Coord, Data, Name, asdataarray

from xrpatcher._src.base import XRDAPatcher
from xrpatcher._src.exceptions import IncompleteScanConfiguration


X = Literal["x"]


@dataclass
class Variable1D:
    data: Data[tuple[X], np.float32]
    x: Coord[X, np.float32] = 0
    name: Name[str] = "var"


@pytest.fixture
def da_1d():
    coord = np.arange(0, 20, 1)
    data = np.ones(len(coord))
    var = Variable1D(data=data, x=coord)
    return asdataarray(var)


def test_incomplete_scan_raises(da_1d):
    """IncompleteScanConfiguration is raised when patch/stride don't evenly cover domain."""
    # (20 - 3) % 2 == 1 != 0 → incomplete scan
    with pytest.raises(IncompleteScanConfiguration):
        XRDAPatcher(
            da=da_1d,
            patches={"x": 3},
            strides={"x": 2},
            check_full_scan=True,
        )


def test_incomplete_scan_no_raise_when_disabled(da_1d):
    """Same config does NOT raise when check_full_scan=False."""
    patcher = XRDAPatcher(
        da=da_1d,
        patches={"x": 3},
        strides={"x": 2},
        check_full_scan=False,
    )
    assert len(patcher) > 0


def test_reconstruct_wrong_dim_labels_count(da_1d):
    """Too many dims_labels raises AssertionError."""
    patcher = XRDAPatcher(da=da_1d, patches={"x": 4}, strides={"x": 4})
    items = list(map(lambda x: x.data, patcher))  # each item has shape (4,)

    with pytest.raises(AssertionError):
        # 3 labels for a 1-D item → mismatch
        patcher.reconstruct(items, dims_labels=["x", "y", "z"])


def test_reconstruct_no_matching_coords(da_1d):
    """dims_labels with no overlap to patcher coords raises AssertionError."""
    patcher = XRDAPatcher(da=da_1d, patches={"x": 4}, strides={"x": 4})
    items = list(map(lambda x: x.data, patcher))

    with pytest.raises(AssertionError):
        # "z" is not a coordinate in the patcher (which has "x")
        patcher.reconstruct(items, dims_labels=["z"])


def test_reconstruct_wrong_weight_shape(da_1d):
    """Weight with wrong number of dimensions raises AssertionError."""
    patcher = XRDAPatcher(da=da_1d, patches={"x": 4}, strides={"x": 4})
    items = list(map(lambda x: x.data, patcher))

    # patch is 1-D (size 4) but weight is 2-D → len mismatch
    wrong_weight = np.ones((4, 4))
    with pytest.raises(AssertionError):
        patcher.reconstruct(items, dims_labels=["x"], weight=wrong_weight)


def test_domain_limits_invalid_key_raises(da_1d):
    """domain_limits with a key not in the DataArray raises AssertionError."""
    with pytest.raises(AssertionError):
        XRDAPatcher(
            da=da_1d,
            patches={"x": 4},
            strides={"x": 4},
            domain_limits={"z": slice(0, 5)},  # "z" doesn't exist in da_1d
        )
