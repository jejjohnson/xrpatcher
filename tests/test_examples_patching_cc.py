"""Tests for the patterns demonstrated in ``examples/patching_cc.py``.

These tests cover the six use-cases shown in the example:

- Case I:   1-D time-series patching (no overlap and with overlap)
- Case II:  2-D grid patching
- Case III: 3-D volume patching
- Case IV:  2-D+T spatio-temporal patching
- Case V:   Reconstruction with multiple output variables (latent expansion)
- Case VI:  Selective-dimension reconstruction

``xarray_dataclasses`` and ``einops`` are treated as optional test
dependencies.  If either is not installed the tests are automatically skipped.
"""

import typing as tp
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from xrpatcher._src.base import XRDAPatcher


# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

einops = pytest.importorskip("einops", reason="einops is not installed")
xrdataclass = pytest.importorskip(
    "xarray_dataclasses", reason="xarray_dataclasses is not installed"
)


# ---------------------------------------------------------------------------
# Type literals used by xarray_dataclasses
# ---------------------------------------------------------------------------

TIME = tp.Literal["time"]
X = tp.Literal["x"]
Y = tp.Literal["y"]
Z = tp.Literal["z"]


# ---------------------------------------------------------------------------
# xarray_dataclasses schemas (mirrors patching_cc.py)
# ---------------------------------------------------------------------------


@dataclass
class TimeAxis:
    data: xrdataclass.Data[TIME, tp.Literal["datetime64[ns]"]]
    name: xrdataclass.Name[str] = "time"
    long_name: xrdataclass.Attr[str] = "Date"


@dataclass
class Variable1D:
    data: xrdataclass.Data[tuple[TIME], np.float32]
    time: xrdataclass.Coordof[TimeAxis] = 0
    name: xrdataclass.Attr[str] = "var"


@dataclass
class XAxis:
    data: xrdataclass.Data[X, np.float32]
    name: xrdataclass.Name[str] = "x"


@dataclass
class YAxis:
    data: xrdataclass.Data[Y, np.float32]
    name: xrdataclass.Name[str] = "y"


@dataclass
class ZAxis:
    data: xrdataclass.Data[Z, np.float32]
    name: xrdataclass.Name[str] = "z"


@dataclass
class Variable2D:
    data: xrdataclass.Data[tuple[X, Y], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    name: xrdataclass.Attr[str] = "var"


@dataclass
class Variable3D:
    data: xrdataclass.Data[tuple[X, Y, Z], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    z: xrdataclass.Coordof[ZAxis] = 0
    name: xrdataclass.Attr[str] = "var"


@dataclass
class Variable2DT:
    data: xrdataclass.Data[tuple[TIME, X, Y], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    time: xrdataclass.Coordof[TimeAxis] = 0
    name: xrdataclass.Attr[str] = "var"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def da_1d():
    """360-step 1-D time-series ``DataArray`` (mirrors Case I in the example)."""
    t = np.arange(1, 360 + 1, 1)
    ts = np.sin(t).astype(np.float32)
    spec = Variable1D(data=ts, time=t, name="var")
    return xrdataclass.asdataarray(spec)


@pytest.fixture
def da_2d():
    """128x128 2-D spatial ``DataArray`` (mirrors Case II in the example)."""
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-2, 2, 128)
    rng = np.random.RandomState(seed=123)
    data = rng.randn(len(x), len(y)).astype(np.float32)
    grid = Variable2D(data=data, x=x, y=y, name="var")
    return xrdataclass.asdataarray(grid)


@pytest.fixture
def da_3d():
    """128x128x128 3-D volume ``DataArray`` (mirrors Case III in the example)."""
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-2, 2, 32)
    z = np.linspace(-5, 5, 32)
    rng = np.random.RandomState(seed=123)
    data = rng.randn(len(x), len(y), len(z)).astype(np.float32)
    grid = Variable3D(data=data, x=x, y=y, z=z, name="var")
    return xrdataclass.asdataarray(grid)


@pytest.fixture
def da_2dt():
    """Small 2-D+T spatio-temporal ``DataArray`` (mirrors Case IV in the example)."""
    x = np.linspace(-1, 1, 40)
    y = np.linspace(-2, 2, 40)
    t = np.arange(1, 21, 1)  # 20 time steps
    rng = np.random.RandomState(seed=123)
    data = rng.randn(len(t), len(x), len(y)).astype(np.float32)
    grid = Variable2DT(data=data, x=x, y=y, time=t, name="var")
    return xrdataclass.asdataarray(grid)


# ---------------------------------------------------------------------------
# Case I: 1-D time-series patching
# ---------------------------------------------------------------------------


def test_1d_no_overlap_patch_count(da_1d):
    """Non-overlapping 1-D patches produce the expected number of items."""
    patcher = XRDAPatcher(
        da=da_1d,
        patches={"time": 30},
        strides={"time": 30},
        check_full_scan=True,
    )
    # 360 / 30 = 12 non-overlapping patches
    assert len(patcher) == 12


def test_1d_overlap_patch_count(da_1d):
    """Overlapping 1-D patches (stride < patch) produce the expected count."""
    patcher = XRDAPatcher(
        da=da_1d,
        patches={"time": 30},
        strides={"time": 15},
        check_full_scan=True,
    )
    # (360 - 30) / 15 + 1 = 23
    assert len(patcher) == 23


def test_1d_patch_shape(da_1d):
    """Each 1-D patch has the correct shape."""
    patcher = XRDAPatcher(
        da=da_1d,
        patches={"time": 30},
        strides={"time": 30},
        check_full_scan=True,
    )
    patch = patcher[0].load()
    assert patch.sizes["time"] == 30


# ---------------------------------------------------------------------------
# Case II: 2-D grid patching
# ---------------------------------------------------------------------------


def test_2d_no_overlap_patch_count(da_2d):
    """Non-overlapping 2-D patches produce the expected number of items."""
    patcher = XRDAPatcher(
        da=da_2d,
        patches={"x": 8, "y": 8},
        strides={"x": 8, "y": 8},
        check_full_scan=True,
    )
    # 128/8 * 128/8 = 256 patches
    assert len(patcher) == 256


def test_2d_patch_shape(da_2d):
    """Each 2-D patch has the correct spatial shape."""
    patcher = XRDAPatcher(
        da=da_2d,
        patches={"x": 8, "y": 8},
        strides={"x": 8, "y": 8},
        check_full_scan=True,
    )
    patch = patcher[0].load()
    assert patch.sizes["x"] == 8
    assert patch.sizes["y"] == 8


def test_2d_overlap_patch_count(da_2d):
    """Overlapping 2-D patches produce a larger dataset than non-overlapping."""
    patcher_no_overlap = XRDAPatcher(
        da=da_2d,
        patches={"x": 8, "y": 8},
        strides={"x": 8, "y": 8},
        check_full_scan=True,
    )
    patcher_overlap = XRDAPatcher(
        da=da_2d,
        patches={"x": 8, "y": 8},
        strides={"x": 2, "y": 2},
        check_full_scan=True,
    )
    assert len(patcher_overlap) > len(patcher_no_overlap)


# ---------------------------------------------------------------------------
# Case III: 3-D volume patching
# ---------------------------------------------------------------------------


def test_3d_no_overlap_patch_count(da_3d):
    """Non-overlapping 3-D patches produce the expected number of items."""
    patcher = XRDAPatcher(
        da=da_3d,
        patches={"x": 8, "y": 8, "z": 8},
        strides={"x": 8, "y": 8, "z": 8},
        check_full_scan=True,
    )
    # (32/8)^3 = 64 patches
    assert len(patcher) == 64


def test_3d_patch_shape(da_3d):
    """Each 3-D patch has the correct volumetric shape."""
    patcher = XRDAPatcher(
        da=da_3d,
        patches={"x": 8, "y": 8, "z": 8},
        strides={"x": 8, "y": 8, "z": 8},
        check_full_scan=True,
    )
    patch = patcher[0].load()
    assert patch.sizes["x"] == 8
    assert patch.sizes["y"] == 8
    assert patch.sizes["z"] == 8


# ---------------------------------------------------------------------------
# Case IV: 2-D+T spatio-temporal patching
# ---------------------------------------------------------------------------


def test_2dt_patch_shape(da_2dt):
    """Each 2-D+T patch has the correct spatio-temporal shape."""
    patcher = XRDAPatcher(
        da=da_2dt,
        patches={"x": 8, "y": 8, "time": 4},
        strides={"x": 4, "y": 4, "time": 2},
        check_full_scan=True,
    )
    assert len(patcher) > 0
    patch = patcher[0].load()
    assert patch.sizes["x"] == 8
    assert patch.sizes["y"] == 8
    assert patch.sizes["time"] == 4


# ---------------------------------------------------------------------------
# Case V: Reconstruction with latent expansion (multiple output variables)
# ---------------------------------------------------------------------------


def test_latent_reconstruction_shape(da_1d):
    """Reconstruction with a latent N-dimension expands the output correctly."""
    patches = {"time": 30}
    strides = {"time": 30}
    patcher = XRDAPatcher(
        da=da_1d,
        patches=patches,
        strides=strides,
        check_full_scan=True,
    )

    # Expand each patch with a latent dimension N=5 (mirrors Case V)
    all_batches = list(map(lambda x: x.data, patcher))
    all_batches_latent = list(
        map(lambda x: einops.repeat(x, "... -> ... N", N=5), all_batches)
    )

    weight = np.ones((patches["time"],), dtype=np.float32)
    rec_da = patcher.reconstruct(
        all_batches_latent,
        dims_labels=["time", "z"],
        weight=weight,
    )

    # Output should have shape (360, 5)
    assert isinstance(rec_da, xr.DataArray)
    assert rec_da.shape == (360, 5)


# ---------------------------------------------------------------------------
# Case VI: Selective-dimension reconstruction
# ---------------------------------------------------------------------------


def test_selective_time_reconstruction(da_2dt):
    """Reconstructing only the time axis produces a 1-D time series."""
    patches = {"x": 8, "y": 8, "time": 4}
    strides = {"x": 4, "y": 4, "time": 2}
    patcher = XRDAPatcher(
        da=da_2dt,
        patches=patches,
        strides=strides,
        check_full_scan=True,
    )

    # Take mean over spatial dims to produce a per-patch time series
    all_batches = list(map(lambda p: p.mean(dim=["x", "y"]).data, patcher))

    weight = np.ones((patches["time"],), dtype=np.float32)
    rec_da = patcher.reconstruct(
        all_batches,
        dims_labels=["time"],
        weight=weight,
    )

    assert isinstance(rec_da, xr.DataArray)
    assert "time" in rec_da.dims
    assert rec_da.sizes["time"] == da_2dt.sizes["time"]


def test_selective_spatial_reconstruction(da_2dt):
    """Reconstructing only the spatial axes produces a 2-D spatial field."""
    patches = {"x": 8, "y": 8, "time": 4}
    strides = {"x": 4, "y": 4, "time": 2}
    patcher = XRDAPatcher(
        da=da_2dt,
        patches=patches,
        strides=strides,
        check_full_scan=True,
    )

    # Take mean over the time dimension to produce a per-patch spatial field
    all_batches = list(map(lambda p: p.mean(dim=["time"]).data, patcher))

    weight = np.ones((patches["x"], patches["y"]), dtype=np.float32)
    rec_da = patcher.reconstruct(
        all_batches,
        dims_labels=["x", "y"],
        weight=weight,
    )

    assert isinstance(rec_da, xr.DataArray)
    assert rec_da.shape == (da_2dt.sizes["x"], da_2dt.sizes["y"])
