"""Tests for the patterns demonstrated in ``examples/pytorch_integration.py``
and ``examples/xrpatcher_concat_torch_dataloading.py``.

These tests cover:

- ``XrTorchDataset`` wrapping ``XRDAPatcher`` for 1-D and 2-D data
- ``torch.utils.data.DataLoader`` iteration over patches
- ``reconstruct_from_batches`` producing a correctly shaped ``xr.DataArray``
- ``XrConcatDataset`` for multi-domain patching and joint reconstruction

``torch`` is treated as an optional test dependency.  If it is not installed
the tests are automatically skipped.
"""

import itertools

import numpy as np
import pytest
import xarray as xr

from xrpatcher._src.base import XRDAPatcher


# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch is not installed")


# ---------------------------------------------------------------------------
# Dataset wrappers (mirrors pytorch_integration.py and
# xrpatcher_concat_torch_dataloading.py)
# ---------------------------------------------------------------------------


class XrTorchDataset(torch.utils.data.Dataset):
    """Thin PyTorch ``Dataset`` wrapper around an ``XRDAPatcher``.

    Args:
        patcher: The patcher object to draw patches from.
        item_postpro: Optional callable applied to each raw patch array.
    """

    def __init__(self, patcher: XRDAPatcher, item_postpro=None):
        self.patcher = patcher
        self.postpro = item_postpro

    def __getitem__(self, idx):
        item = self.patcher[idx].load().values
        if self.postpro:
            item = self.postpro(item)
        return item

    def reconstruct_from_batches(self, batches, **rec_kws):
        """Reconstruct from an iterable of batches collected from the loader."""
        return self.patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __len__(self):
        return len(self.patcher)


class XrConcatDataset(torch.utils.data.ConcatDataset):
    """Concatenated dataset over multiple ``XrTorchDataset`` instances.

    Supports per-domain reconstruction from a shared loader's batch list.
    """

    def __init__(self, *dses: XrTorchDataset):
        super().__init__(dses)

    def reconstruct_from_batches(self, batches, weight=None):
        """Reconstruct each sub-dataset in order from the combined batch list."""
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.patcher.reconstruct(ds_items, weight=weight))
        return rec_das


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_da(n_time: int = 200, seed: int = 0) -> xr.DataArray:
    """Create a synthetic 1-variable 1-D ``DataArray``."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((1, n_time)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("variable", "time"),
        coords={"variable": ["air"], "time": np.arange(n_time)},
    )


def _make_2d_da(
    nx: int = 100, ny: int = 100, n_var: int = 2, seed: int = 0
) -> xr.DataArray:
    """Create a synthetic 2-variable 2-D ``DataArray``."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_var, nx, ny)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("variable", "latitude", "longitude"),
        coords={
            "variable": ["u", "v"],
            "latitude": np.linspace(-90, 90, nx),
            "longitude": np.linspace(-180, 180, ny),
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patcher_1d():
    """``XRDAPatcher`` over a 1-D time-series."""
    da = _make_1d_da(n_time=200)
    return XRDAPatcher(
        da=da,
        patches={"time": 40},
        strides={"time": 20},
        check_full_scan=True,
    )


@pytest.fixture
def torch_ds_1d(patcher_1d):
    """``XrTorchDataset`` wrapping the 1-D patcher."""
    return XrTorchDataset(patcher_1d)


@pytest.fixture
def patcher_2d():
    """``XRDAPatcher`` over a 2-variable 2-D spatial field."""
    da = _make_2d_da(nx=100, ny=100)
    return XRDAPatcher(
        da=da,
        patches={"latitude": 50, "longitude": 50},
        strides={"latitude": 25, "longitude": 25},
        check_full_scan=True,
    )


@pytest.fixture
def torch_ds_2d(patcher_2d):
    """``XrTorchDataset`` wrapping the 2-D patcher."""
    return XrTorchDataset(patcher_2d)


# ---------------------------------------------------------------------------
# Tests: XrTorchDataset (1-D)
# ---------------------------------------------------------------------------


def test_torch_ds_1d_len(torch_ds_1d, patcher_1d):
    """Dataset length matches the underlying patcher."""
    assert len(torch_ds_1d) == len(patcher_1d)


def test_torch_ds_1d_item_shape(torch_ds_1d):
    """Single item from the 1-D dataset has the expected shape."""
    item = torch_ds_1d[0]
    assert item.shape == (1, 40)  # (variables, time_patch)


def test_torch_ds_1d_dataloader_batch_shape(torch_ds_1d):
    """DataLoader batches from the 1-D dataset have the expected shape."""
    loader = torch.utils.data.DataLoader(torch_ds_1d, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.shape[1:] == (1, 40)  # (B, variables, time)
    assert batch.shape[0] <= 4


def test_torch_ds_1d_reconstruct_shape(torch_ds_1d, patcher_1d):
    """Reconstruction from all 1-D batches returns the original time shape."""
    loader = torch.utils.data.DataLoader(torch_ds_1d, batch_size=4, shuffle=False)
    batches = list(loader)
    rec = torch_ds_1d.reconstruct_from_batches(batches)
    assert isinstance(rec, xr.DataArray)
    assert rec.sizes["time"] == patcher_1d.da.sizes["time"]


# ---------------------------------------------------------------------------
# Tests: XrTorchDataset (2-D)
# ---------------------------------------------------------------------------


def test_torch_ds_2d_len(torch_ds_2d, patcher_2d):
    """2-D dataset length matches the underlying patcher."""
    assert len(torch_ds_2d) == len(patcher_2d)


def test_torch_ds_2d_item_shape(torch_ds_2d):
    """Single item from the 2-D dataset has the expected shape."""
    item = torch_ds_2d[0]
    # (variables, lat_patch, lon_patch)
    assert item.shape == (2, 50, 50)


def test_torch_ds_2d_dataloader_batch_shape(torch_ds_2d):
    """DataLoader batches from the 2-D dataset have the expected shape."""
    loader = torch.utils.data.DataLoader(torch_ds_2d, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.shape[1:] == (2, 50, 50)


def test_torch_ds_2d_reconstruct_shape(torch_ds_2d, patcher_2d):
    """Reconstruction from 2-D batches returns the original spatial shape."""
    loader = torch.utils.data.DataLoader(torch_ds_2d, batch_size=4, shuffle=False)
    batches = list(loader)
    # Use the first variable only as a scalar-field prediction
    rec = torch_ds_2d.reconstruct_from_batches(
        (batch[:, 0] for batch in batches),
        dims_labels=["latitude", "longitude"],
    )
    assert isinstance(rec, xr.DataArray)
    da = patcher_2d.da
    assert rec.shape == (da.sizes["latitude"], da.sizes["longitude"])


def test_torch_ds_2d_reconstruct_with_weight(torch_ds_2d, patcher_2d):
    """Weighted reconstruction from 2-D batches completes without error."""
    loader = torch.utils.data.DataLoader(torch_ds_2d, batch_size=4, shuffle=False)
    batches = list(loader)

    weight = np.ones((50, 50), dtype=np.float32)
    weight[:5] = 0.0
    weight[-5:] = 0.0
    weight[:, :5] = 0.0
    weight[:, -5:] = 0.0

    rec = torch_ds_2d.reconstruct_from_batches(
        (batch[:, 0] for batch in batches),
        dims_labels=["latitude", "longitude"],
        weight=weight,
    )
    da = patcher_2d.da
    assert rec.shape == (da.sizes["latitude"], da.sizes["longitude"])


# ---------------------------------------------------------------------------
# Tests: XrConcatDataset (xrpatcher_concat_torch_dataloading.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def concat_dataset():
    """Two ``XrTorchDataset`` instances combined via ``XrConcatDataset``."""
    rng = np.random.default_rng(42)
    data = xr.DataArray(
        rng.standard_normal(100).astype(np.float32),
        dims=("time",),
        coords={"time": np.arange(100)},
    )
    p1 = XRDAPatcher(
        data, patches={"time": 20}, strides={"time": 10}, check_full_scan=True
    )
    p2 = XRDAPatcher(
        data, patches={"time": 20}, strides={"time": 10}, check_full_scan=True
    )
    ds1 = XrTorchDataset(p1)
    ds2 = XrTorchDataset(p2)
    return XrConcatDataset(ds1, ds2)


def test_concat_dataset_len(concat_dataset):
    """Combined dataset length equals the sum of the sub-dataset lengths."""
    sub_lens = [len(ds) for ds in concat_dataset.datasets]
    assert len(concat_dataset) == sum(sub_lens)


def test_concat_dataset_item_shape(concat_dataset):
    """Items from the concat dataset have the expected patch shape."""
    item = concat_dataset[0]
    assert item.shape == (20,)


def test_concat_dataset_reconstruct_returns_two_arrays(concat_dataset):
    """Reconstruct from the combined loader returns one DataArray per sub-dataset."""
    loader = torch.utils.data.DataLoader(concat_dataset, batch_size=4, shuffle=False)
    batches = list(loader)
    results = concat_dataset.reconstruct_from_batches(batches)
    assert len(results) == 2
    for rec in results:
        assert isinstance(rec, xr.DataArray)
        assert rec.sizes["time"] == 100
