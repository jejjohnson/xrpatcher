"""End-to-end integration tests for the grain + xrpatcher example pipeline.

These tests verify that:
- patch extraction via ``XRDAPatcher`` works as expected,
- ``grain.MapDataset`` can wrap patcher indices and map a fetch function,
- the ``.batch()`` transform produces correctly shaped NumPy arrays,
- ``patcher.reconstruct()`` returns an ``xr.DataArray`` with the correct
  shape after processing all batched predictions.

``grain`` and ``jax`` are treated as optional test dependencies.  Tests that
require them are skipped individually when the packages are not installed.
"""

import numpy as np
import pytest
import xarray as xr

from xrpatcher._src.base import XRDAPatcher


# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import grain as _grain_mod
except ImportError:  # pragma: no cover - behavior tested via skipping
    _grain_mod = None

try:
    import jax as _jax_mod
except ImportError:  # pragma: no cover - behavior tested via skipping
    _jax_mod = None


class _OptionalDependencyProxy:
    """Proxy that skips the current test on first attribute access."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str):
        pytest.skip(f"{self._name} is not installed")


grain = _grain_mod if _grain_mod is not None else _OptionalDependencyProxy("grain")
jax = _jax_mod if _jax_mod is not None else _OptionalDependencyProxy("jax")
jnp = _jax_mod.numpy if _jax_mod is not None else _OptionalDependencyProxy("jax.numpy")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_multi_channel_da(
    n_channels: int = 3,
    nx: int = 64,
    ny: int = 64,
    seed: int = 0,
) -> xr.DataArray:
    """Create a synthetic multi-channel ``DataArray`` for testing.

    Args:
        n_channels: Number of channels (variable dimension).
        nx: Size of the x dimension.
        ny: Size of the y dimension.
        seed: Random seed for reproducibility.

    Returns:
        DataArray with dimensions ``(channel, x, y)`` and float32 values.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_channels, nx, ny)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("channel", "x", "y"),
        coords={
            "channel": np.arange(n_channels),
            "x": np.linspace(-1, 1, nx),
            "y": np.linspace(-1, 1, ny),
        },
    )


@pytest.fixture
def da_3ch():
    """A 3-channel 64x64 ``DataArray`` ready for patchifying."""
    return _make_multi_channel_da(n_channels=3, nx=64, ny=64)


@pytest.fixture
def patcher_3ch(da_3ch):
    """``XRDAPatcher`` with 16x16 patches and 8x8 strides on ``da_3ch``."""
    return XRDAPatcher(
        da=da_3ch,
        patches={"x": 16, "y": 16},
        strides={"x": 8, "y": 8},
        check_full_scan=True,
    )


# ---------------------------------------------------------------------------
# Tests: grain MapDataset construction
# ---------------------------------------------------------------------------


def test_grain_map_dataset_length(patcher_3ch):
    """Grain MapDataset wrapping patch indices has the same length as the patcher."""
    grain_ds = grain.MapDataset.source(list(range(len(patcher_3ch))))
    assert len(grain_ds) == len(patcher_3ch)


def test_grain_fetch_patch_shape(patcher_3ch):
    """Fetching a patch via grain map returns an array with the correct shape."""

    def fetch(idx: int) -> dict:
        patch = patcher_3ch[idx].load()
        return {"x": patch.values.astype(np.float32), "idx": np.int32(idx)}

    grain_ds = grain.MapDataset.source(list(range(len(patcher_3ch)))).map(fetch)

    record = grain_ds[0]
    assert record["x"].shape == (3, 16, 16), (
        f"Expected shape (3, 16, 16), got {record['x'].shape}"
    )
    assert record["idx"] == 0


def test_grain_batch_shape(patcher_3ch):
    """Batching via ``.batch()`` stacks patches along a new leading axis."""

    def fetch(idx: int) -> np.ndarray:
        return patcher_3ch[idx].load().values.astype(np.float32)

    batch_size = 4
    batched = (
        grain.MapDataset.source(list(range(len(patcher_3ch))))
        .map(fetch)
        .batch(batch_size=batch_size)
    )

    first_batch = batched[0]
    assert first_batch.shape == (batch_size, 3, 16, 16), (
        f"Expected shape ({batch_size}, 3, 16, 16), got {first_batch.shape}"
    )


# ---------------------------------------------------------------------------
# Tests: JAX model step on batches
# ---------------------------------------------------------------------------


def test_jax_model_step_output_shape(patcher_3ch):
    """A JAX model step applied to a batch produces the expected output shape."""

    def fetch(idx: int) -> np.ndarray:
        return patcher_3ch[idx].load().values.astype(np.float32)

    batch_size = 4
    batched = (
        grain.MapDataset.source(list(range(len(patcher_3ch))))
        .map(fetch)
        .batch(batch_size=batch_size)
    )

    x_batch = batched[0]  # (B, C, H, W)
    # Simulate a simple JAX model: mean squared activation over channels.
    jx = jnp.array(x_batch)
    pred = jnp.mean(jx**2, axis=1)  # -> (B, H, W)
    pred_np = np.asarray(pred)

    assert pred_np.shape == (batch_size, 16, 16), (
        f"Expected shape ({batch_size}, 16, 16), got {pred_np.shape}"
    )


# ---------------------------------------------------------------------------
# Tests: end-to-end reconstruct from grain-batched predictions
# ---------------------------------------------------------------------------


def test_reconstruct_from_grain_batches(patcher_3ch):
    """Full round-trip: grain batches -> JAX model step -> reconstruct."""

    def fetch(idx: int) -> np.ndarray:
        return patcher_3ch[idx].load().values.astype(np.float32)

    batch_size = 4
    batched = (
        grain.MapDataset.source(list(range(len(patcher_3ch))))
        .map(fetch)
        .batch(batch_size=batch_size)
    )

    # Collect per-batch predictions using a simple JAX "model".
    all_preds: list[np.ndarray] = []
    for i in range(len(batched)):
        x = jnp.array(batched[i])
        pred = np.asarray(jnp.mean(x**2, axis=1))  # (B, H, W)
        all_preds.append(pred)

    # Flatten list-of-batches into a list of individual (H, W) arrays.
    flat_preds = [item for batch in all_preds for item in batch]

    # Reconstruct must yield a DataArray with the original spatial dimensions.
    reconstructed = patcher_3ch.reconstruct(
        flat_preds,
        dims_labels=["x", "y"],
    )

    da = patcher_3ch.da
    assert isinstance(reconstructed, xr.DataArray)
    assert reconstructed.shape == (da.sizes["x"], da.sizes["y"]), (
        f"Reconstructed shape {reconstructed.shape} does not match "
        f"original spatial shape ({da.sizes['x']}, {da.sizes['y']})"
    )


def test_reconstruct_with_edge_weight(patcher_3ch):
    """Reconstruction with an edge-tapered weight completes without error."""

    def fetch(idx: int) -> np.ndarray:
        return patcher_3ch[idx].load().values.astype(np.float32)

    batch_size = 8
    batched = (
        grain.MapDataset.source(list(range(len(patcher_3ch))))
        .map(fetch)
        .batch(batch_size=batch_size)
    )

    weight = np.ones((16, 16), dtype=np.float32)
    border = 2
    weight[:border] = 0.0
    weight[-border:] = 0.0
    weight[:, :border] = 0.0
    weight[:, -border:] = 0.0

    all_preds: list[np.ndarray] = []
    for i in range(len(batched)):
        x = jnp.array(batched[i])
        pred = np.asarray(jnp.mean(x**2, axis=1))
        all_preds.append(pred)

    flat_preds = [item for batch in all_preds for item in batch]

    reconstructed = patcher_3ch.reconstruct(
        flat_preds,
        dims_labels=["x", "y"],
        weight=weight,
    )

    da = patcher_3ch.da
    assert reconstructed.shape == (da.sizes["x"], da.sizes["y"])


# ---------------------------------------------------------------------------
# Tests: physical + coordinate transforms (mirrors the example logic)
# ---------------------------------------------------------------------------


def test_add_physical_variables():
    """Wind speed and direction are computed correctly from u/v components."""
    ds = xr.Dataset(
        {
            "u": xr.DataArray(np.array([3.0, 0.0]), dims="x"),
            "v": xr.DataArray(np.array([4.0, 1.0]), dims="x"),
        }
    )
    ds["wind_speed"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds["wind_direction"] = np.arctan2(ds["v"], ds["u"])

    np.testing.assert_allclose(ds["wind_speed"].values, [5.0, 1.0])
    np.testing.assert_allclose(
        ds["wind_direction"].values,
        np.arctan2([4.0, 1.0], [3.0, 0.0]),
    )


def test_standardization_statistics():
    """After standardization the per-channel mean is ~0 and std is ~1."""
    rng = np.random.default_rng(42)
    data_vals = rng.standard_normal((3, 64, 64)).astype(np.float32)
    da = xr.DataArray(
        data_vals,
        dims=("variable", "x", "y"),
    )
    mean = da.mean(dim=("x", "y"))
    std = da.std(dim=("x", "y"))
    da_std = (da - mean) / (std + 1e-6)

    np.testing.assert_allclose(
        da_std.mean(dim=("x", "y")).values,
        np.zeros(3),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        da_std.std(dim=("x", "y")).values,
        np.ones(3),
        atol=1e-3,
    )
