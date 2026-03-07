from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import xarray as xr
from einops import repeat
from xarray_dataclasses import Coord, Data, Name, asdataarray

from xrpatcher._src.base import XRDAPatcher


RNG = np.random.RandomState(seed=123)

X = Literal["x"]
Y = Literal["y"]
Z = Literal["z"]


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


@dataclass
class Variable3D:
    data: Data[tuple[X, Y, Z], np.float32]
    x: Coord[X, np.float32] = 0
    y: Coord[Y, np.float32] = 0
    z: Coord[Z, np.float32] = 0
    name: Name[str] = "var"


@pytest.fixture
def axis_1d():
    return np.arange(-10, 10, 1)


@pytest.fixture
def axis_2d(axis_1d):
    axis2 = np.arange(-20, 20, 1)
    return axis_1d, axis2


@pytest.fixture
def axis_3d(axis_2d):
    axis1, axis2 = axis_2d
    axis3 = np.arange(-30, 30, 1)
    return axis1, axis2, axis3


@pytest.fixture
def variable_1d(axis_1d):
    ones = np.ones((len(axis_1d),))
    var = Variable1D(ones, axis_1d)
    return asdataarray(var)


@pytest.fixture
def variable_2d(axis_2d):
    axis1, axis2 = axis_2d
    ones = np.ones((len(axis1), len(axis2)))
    var = Variable2D(ones, axis1, axis2)
    return asdataarray(var)


@pytest.fixture
def variable_3d(axis_3d):
    axis1, axis2, axis3 = axis_3d
    ones = np.ones((len(axis1), len(axis2), len(axis3)))
    var = Variable3D(ones, axis1, axis2, axis3)
    return asdataarray(var)


@pytest.mark.parametrize(
    "patch,stride,domain_limits,datasize",
    [
        (None, None, None, 1),
        (1, None, None, 20),
        (1, 1, None, 20),
        (1, None, {"x": slice(-5, 5)}, 11),
        (5, None, None, 16),
        (4, 2, None, 9),
        (5, 2, {"x": slice(-5, 5)}, 4),
    ],
)
def test_xrda_patcher_1d(variable_1d, patch, stride, domain_limits, datasize):
    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None
    check_full_scan = True

    ds = XRDAPatcher(
        da=variable_1d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
    )

    msg = f"Patches: {ds.patches} | Strides: {ds.strides} | Dims: {ds.da_size}"
    assert ds.strides == ({"x": 1} if strides is None else strides), msg
    assert ds.patches == ({"x": 20} if patches is None else patches), msg
    assert ds.da_size == {"x": datasize}, msg
    assert ds[0].shape == (
        (patch,) if patch is not None else tuple(ds.patches.values())
    ), msg
    assert len(ds) == datasize


# Use a small 1D array to keep reconstruct tests fast (fewer patches = fewer iterations)
_COORD_1D = np.arange(1, 61, 1)  # 60 elements


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        (30, None),
        (30, 5),
        (20, 20),
    ],
)
def test_xrda_patcher_1d_reconstruct(patch, stride):
    data = RNG.randn(*_COORD_1D.shape)
    da = Variable1D(data=data, x=_COORD_1D, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))

    # No Weight | No Label
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=None, weight=None)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple("x")

    # Weight | Exact Label
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x"], weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x"}
    assert rec_da.dims == tuple("x")


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        (30, None),
        (30, 5),
        (20, 20),
    ],
)
def test_xrda_patcher_1d_reconstruct_latent(patch, stride):
    data = RNG.randn(*_COORD_1D.shape)
    da = Variable1D(data=data, x=_COORD_1D, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch} if patch is not None else None
    strides = {"x": stride} if stride is not None else None

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))
    all_items = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_items))

    # No Weight | No Label
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=None, weight=None)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x"}
    assert set(rec_da.dims) == {"x", "v1"}

    # No Weight | Exact Label with extra dim
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x", "z"], weight=None)
    np.testing.assert_array_almost_equal(rec_da.isel(z=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x"}
    assert set(rec_da.dims) == {"x", "z"}


@pytest.mark.parametrize(
    "patch,stride,domain_limits,datasize",
    [
        (None, None, None, (1, 1)),
        ((1, 1), None, None, (20, 40)),
        ((1, 1), (1, 1), None, (20, 40)),
        ((1, 1), None, {"x": slice(-5, 5)}, (11, 40)),
        ((1, 1), None, {"y": slice(-10, 10)}, (20, 21)),
        ((1, 1), None, {"x": slice(-5, 5), "y": slice(-10, 10)}, (11, 21)),
        ((5, 1), None, None, (16, 40)),
        ((1, 5), None, None, (20, 36)),
        ((5, 5), None, None, (16, 36)),
        ((10, 20), (2, 1), None, (6, 21)),
        ((10, 20), (1, 2), None, (11, 11)),
        ((10, 4), (2, 4), None, (6, 10)),
    ],
)
def test_xrda_patcher_2d(variable_2d, patch, stride, domain_limits, datasize):
    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None
    check_full_scan = True

    ds = XRDAPatcher(
        da=variable_2d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
    )

    msg = f"Patches: {ds.patches} | Strides: {ds.strides} | Dims: {ds.da_size}"
    assert ds.strides == ({"x": 1, "y": 1} if strides is None else strides), msg
    assert ds.patches == ({"x": 20, "y": 40} if patches is None else patches), msg
    assert ds.da_size == {"x": datasize[0], "y": datasize[1]}, msg
    assert ds[0].shape == (
        (patch[0], patch[1]) if patch is not None else tuple(ds.patches.values())
    ), msg
    assert len(ds) == np.prod(list(datasize))


# Use small square 2D arrays to keep reconstruct tests fast.
# Square arrays ensure mixed-label projection tests work regardless of dimension order.
# Axis size (12) must be divisible by patch sizes used below.
_AXIS_2D = np.arange(10, 22, 1)  # 12 elements (used for both x and y)


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        ((4, 4), (4, 4)),
        ((4, 4), (2, 2)),
    ],
)
def test_xrda_patcher_2d_reconstruct(patch, stride):
    # data shape is (x, y); use square arrays so mixed-label cases are unambiguous
    data = RNG.randn(_AXIS_2D.shape[0], _AXIS_2D.shape[0])
    da = Variable2D(data=data, x=_AXIS_2D, y=_AXIS_2D, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))

    # No Weight | No Label
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=None, weight=None)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["x", "y"])

    # Weight | Exact Label
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x", "y"], weight=weight)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["x", "y"])

    # Weight | Mixed Label (x only)
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x"], weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "v1"])

    # No Weight | Mixed Label (y only)
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["y"], weight=None)
    assert list(rec_da.coords.keys()) == ["y"]
    assert rec_da.dims == tuple(["y", "v1"])


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        ((4, 4), (4, 4)),
        ((4, 4), (2, 2)),
    ],
)
def test_xrda_patcher_2d_reconstruct_latent(patch, stride):
    data = RNG.randn(_AXIS_2D.shape[0], _AXIS_2D.shape[0])
    da = Variable2D(data=data, x=_AXIS_2D, y=_AXIS_2D, name="ssh")
    da = asdataarray(da)

    patches = {"x": patch[0], "y": patch[1]} if patch is not None else None
    strides = {"x": stride[0], "y": stride[1]} if stride is not None else None

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))
    all_items = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_items))

    # No Weight | No Label — auto-adds extra dim
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=None, weight=None)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["x", "y", "v1"])

    # No Weight | Exact Label with named extra dim
    rec_da = xrda_batcher.reconstruct(
        all_items, dims_labels=["x", "y", "z"], weight=None
    )
    np.testing.assert_array_almost_equal(rec_da.isel(z=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["x", "y", "z"])

    # Weight | Mixed Label (x only)
    weight = np.ones((xrda_batcher.patches["x"],))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x"], weight=weight)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == tuple(["x", "v1", "v2"])

    # Reshaped items: y z x layout
    all_items_yzx = list(map(lambda x: repeat(x, "x y z -> y z x"), all_items))
    weight = np.ones((xrda_batcher.patches["y"], xrda_batcher.patches["x"]))
    rec_da = xrda_batcher.reconstruct(
        all_items_yzx, dims_labels=["y", "z", "x"], weight=weight
    )
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["y", "z", "x"])


# Use small cube 3D arrays to keep reconstruct tests fast.
# Axis size (8) must be divisible by patch sizes used below.
_AXIS_3D = np.arange(0, 8, 1)  # 8 elements (used for x, y, and z)


@pytest.mark.parametrize(
    "patch,stride,domain_limits,datasize",
    [
        (None, None, None, (1, 1, 1)),
        ((1, 1, 1), None, None, (20, 40, 60)),
        ((1, 1, 1), (1, 1, 1), None, (20, 40, 60)),
        ((5, 5, 5), None, None, (16, 36, 56)),
        ((5, 5, 5), (5, 5, 5), None, (4, 8, 12)),
        ((10, 8, 6), (2, 4, 6), None, (6, 9, 10)),
        ((5, 5, 5), (5, 5, 5), {"x": slice(-5, 4)}, (2, 8, 12)),
    ],
)
def test_xrda_patcher_3d(variable_3d, patch, stride, domain_limits, datasize):
    patches = (
        {"x": patch[0], "y": patch[1], "z": patch[2]} if patch is not None else None
    )
    strides = (
        {"x": stride[0], "y": stride[1], "z": stride[2]} if stride is not None else None
    )
    check_full_scan = True

    ds = XRDAPatcher(
        da=variable_3d,
        patches=patches,
        strides=strides,
        domain_limits=domain_limits,
        check_full_scan=check_full_scan,
    )

    msg = f"Patches: {ds.patches} | Strides: {ds.strides} | Dims: {ds.da_size}"
    assert ds.strides == ({"x": 1, "y": 1, "z": 1} if strides is None else strides), msg
    assert ds.patches == (
        {"x": 20, "y": 40, "z": 60} if patches is None else patches
    ), msg
    assert ds.da_size == {"x": datasize[0], "y": datasize[1], "z": datasize[2]}, msg
    assert ds[0].shape == (
        (patch[0], patch[1], patch[2])
        if patch is not None
        else tuple(ds.patches.values())
    ), msg
    assert len(ds) == np.prod(list(datasize))


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        ((4, 4, 4), (4, 4, 4)),
        ((4, 4, 4), (2, 2, 2)),
    ],
)
def test_xrda_patcher_3d_reconstruct(patch, stride):
    data = RNG.randn(_AXIS_3D.shape[0], _AXIS_3D.shape[0], _AXIS_3D.shape[0])
    da = Variable3D(data=data, x=_AXIS_3D, y=_AXIS_3D, z=_AXIS_3D, name="ssh")
    da = asdataarray(da)

    patches = (
        {"x": patch[0], "y": patch[1], "z": patch[2]} if patch is not None else None
    )
    strides = (
        {"x": stride[0], "y": stride[1], "z": stride[2]} if stride is not None else None
    )

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))

    # No Weight | No Label
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=None, weight=None)
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y", "z"}
    assert rec_da.dims == tuple(["x", "y", "z"])

    # Weight | Exact Label
    weight = np.ones(
        (
            xrda_batcher.patches["x"],
            xrda_batcher.patches["y"],
            xrda_batcher.patches["z"],
        )
    )
    rec_da = xrda_batcher.reconstruct(
        all_items, dims_labels=["x", "y", "z"], weight=weight
    )
    np.testing.assert_array_almost_equal(rec_da.data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y", "z"}
    assert rec_da.dims == tuple(["x", "y", "z"])

    # Weight | Mixed Label (x, y only)
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x", "y"], weight=weight)
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["x", "y", "v1"])

    # No Weight | Mixed Label (z only)
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["z"], weight=None)
    assert list(rec_da.coords.keys()) == ["z"]
    assert rec_da.dims == tuple(["z", "v1", "v2"])


@pytest.mark.parametrize(
    "patch, stride",
    [
        (None, None),
        ((4, 4, 4), (4, 4, 4)),
        ((4, 4, 4), (2, 2, 2)),
    ],
)
def test_xrda_patcher_3d_reconstruct_latent(patch, stride):
    data = RNG.randn(_AXIS_3D.shape[0], _AXIS_3D.shape[0], _AXIS_3D.shape[0])
    da = Variable3D(data=data, x=_AXIS_3D, y=_AXIS_3D, z=_AXIS_3D, name="ssh")
    da = asdataarray(da)

    patches = (
        {"x": patch[0], "y": patch[1], "z": patch[2]} if patch is not None else None
    )
    strides = (
        {"x": stride[0], "y": stride[1], "z": stride[2]} if stride is not None else None
    )

    xrda_batcher = XRDAPatcher(
        da=da, patches=patches, strides=strides, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))
    all_items = list(map(lambda x: repeat(x, "... -> ... N", N=5), all_items))

    # No Weight | No Label — auto-adds extra dim
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=None, weight=None)
    np.testing.assert_array_almost_equal(rec_da.isel(v1=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y", "z"}
    assert rec_da.dims == tuple(["x", "y", "z", "v1"])

    # No Weight | Exact Label with named extra dim
    rec_da = xrda_batcher.reconstruct(
        all_items, dims_labels=["x", "y", "z", "w"], weight=None
    )
    np.testing.assert_array_almost_equal(rec_da.isel(w=0).data, xrda_batcher.da)
    assert set(rec_da.coords.keys()) == {"x", "y", "z"}
    assert rec_da.dims == tuple(["x", "y", "z", "w"])

    # Weight | Mixed Label (x, y only)
    weight = np.ones((xrda_batcher.patches["x"], xrda_batcher.patches["y"]))
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x", "y"], weight=weight)
    assert set(rec_da.coords.keys()) == {"x", "y"}
    assert rec_da.dims == tuple(["x", "y", "v1", "v2"])


def test_reconstruct_with_nonuniform_weight():
    """Non-uniform weight should still reconstruct correctly for real patch data."""
    coord = np.arange(1, 13, 1)  # 12 elements
    data = RNG.randn(12)
    da = Variable1D(data=data, x=coord, name="ssh")
    da = asdataarray(da)

    # Overlapping patches: 5 patches total
    xrda_batcher = XRDAPatcher(
        da=da, patches={"x": 4}, strides={"x": 2}, check_full_scan=True
    )
    all_items = list(map(lambda x: x.data, xrda_batcher))

    # Non-uniform (triangular) weight — heavier in the centre
    weight = np.array([1.0, 3.0, 3.0, 1.0])
    rec_da = xrda_batcher.reconstruct(all_items, dims_labels=["x"], weight=weight)

    # With real patch values each point's weighted average equals its original value
    np.testing.assert_array_almost_equal(rec_da.data, da.data)
    assert list(rec_da.coords.keys()) == ["x"]
    assert rec_da.dims == ("x",)


def test_reconstruct_bypasses_cache_and_preload_for_internal_coords(monkeypatch):
    """reconstruct() should not preload/cache patches just to recover coordinates."""
    coord = np.arange(1, 13, 1)
    data = RNG.randn(12)
    da = Variable1D(data=data, x=coord, name="ssh")
    da = asdataarray(da)

    uncached_patcher = XRDAPatcher(
        da=da, patches={"x": 4}, strides={"x": 2}, check_full_scan=True
    )
    cached_patcher = XRDAPatcher(
        da=da,
        patches={"x": 4},
        strides={"x": 2},
        check_full_scan=True,
        cache=True,
        preload=True,
    )
    all_items = list(map(lambda x: x.data, uncached_patcher))

    calls = 0
    original_load = xr.DataArray.load

    def spy_load(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", spy_load)

    rec_da = cached_patcher.reconstruct(all_items, dims_labels=["x"], weight=None)

    np.testing.assert_array_almost_equal(rec_da.data, da.data)
    assert calls == 0

    _ = cached_patcher[0]

    assert calls == 1
