# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/grain_integration.py)

# %% [markdown]
# # XRPatcher + Google Grain Integration
#
# This example demonstrates how to combine:
#
# - `xrpatcher` for coordinate-aware patch extraction and reconstruction
# - `grain` (google/grain) for dataset transforms, batching, and data loading
# - `jax` for accelerator-ready array operations on batches
#
# ## Pipeline overview
#
# 1. Load a geophysical `xarray.Dataset`
# 2. Add derived physical variables (wind speed, wind direction)
# 3. Apply coordinate transforms (sort, add radian auxiliary coordinates)
# 4. Stack variables into a channel dimension
# 5. Standardize each channel independently
# 6. Patchify the stacked array with `xrpatcher`
# 7. Build a `grain.MapDataset` backed by patch indices
# 8. Batch patches using `grain`'s native `.batch()` method
# 9. Apply a JAX model step on each batch
# 10. Reconstruct the patch-wise predictions back onto the original coordinates

# %%
# Install dependencies (Colab / first-run setup)
try:
    import grain
except ImportError:
    import subprocess

    subprocess.run(
        [
            "pip",
            "install",
            "grain",
            "jax",
            "xrpatcher",
            "xarray[io]",
            "matplotlib",
            "numpy",
            "pooch",
        ],
        check=True,
    )

# %%
import grain
import jax.numpy as jnp
import numpy as np
import xarray as xr

from xrpatcher import XRDAPatcher


# %% [markdown]
# ## 1. Load a demo geophysical dataset
#
# We use the ERA-Interim reanalysis tutorial dataset bundled with `xarray`.
# It provides zonal wind (`u`), meridional wind (`v`), and geopotential (`z`)
# on a global lat/lon grid.

# %%
raw = xr.tutorial.load_dataset("eraint_uvz")

# Select a single month and pressure level for a 2-D spatial example.
ds = raw[["u", "v", "z"]].isel(month=0, level=0)

# Sort coordinates so that latitude increases from south to north.
ds = ds.sortby("latitude").sortby("longitude")

print(ds)


# %% [markdown]
# ## 2. Add physical variables
#
# We derive two meteorologically meaningful quantities from the wind components:
#
# - **wind speed** = √(u² + v²)  — scalar magnitude of the wind vector
# - **wind direction** = atan2(v, u)  — angle of the wind vector (radians)


# %%
def add_physical_variables(ds: xr.Dataset) -> xr.Dataset:
    """Compute wind speed and direction from u/v components.

    Args:
        ds: Dataset with variables ``u`` (zonal wind) and ``v`` (meridional wind).

    Returns:
        Dataset with two additional variables: ``wind_speed`` and
        ``wind_direction``.
    """
    ds = ds.copy()
    ds["wind_speed"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds["wind_direction"] = np.arctan2(ds["v"], ds["u"])
    return ds


ds = add_physical_variables(ds)

print("Variables:", list(ds.data_vars))


# %% [markdown]
# ## 3. Coordinate transforms
#
# We attach auxiliary coordinates that can serve as additional geospatial
# features in a model:
#
# - `latitude_rad` / `longitude_rad` — raw angles in radians
# - `cos_lat` — cosine of latitude, useful for area-weighted operations


# %%
def add_coordinate_features(ds: xr.Dataset) -> xr.Dataset:
    """Attach auxiliary coordinate features derived from lat/lon.

    Adds radian-valued latitude and longitude as non-index coordinates,
    as well as the cosine of latitude.

    Args:
        ds: Dataset with ``latitude`` and ``longitude`` coordinate dimensions.

    Returns:
        Dataset with three additional (non-index) auxiliary coordinates:
        ``latitude_rad``, ``longitude_rad``, and ``cos_lat``.
    """
    lat_rad = np.deg2rad(ds.latitude.data)
    lon_rad = np.deg2rad(ds.longitude.data)
    return ds.assign_coords(
        latitude_rad=("latitude", lat_rad),
        longitude_rad=("longitude", lon_rad),
        cos_lat=("latitude", np.cos(lat_rad)),
    )


ds = add_coordinate_features(ds)

print("Coordinates:", list(ds.coords))


# %% [markdown]
# ## 4. Stack variables into a channel dimension
#
# We combine all physical variables into a single `DataArray` with shape
# `(variable, latitude, longitude)`.  This is the standard multi-channel
# layout expected by most deep-learning pipelines.

# %%
VARIABLES = ["u", "v", "z", "wind_speed", "wind_direction"]

data = (
    ds[VARIABLES]
    .to_array("variable")
    .transpose("variable", "latitude", "longitude")
    .load()
)

print(data)


# %% [markdown]
# ## 5. Standardize each channel independently
#
# We normalize over the spatial dimensions so that each channel has approximately
# zero mean and unit variance.

# %%
channel_mean = data.mean(dim=("latitude", "longitude"))
channel_std = data.std(dim=("latitude", "longitude"))
data_std = (data - channel_mean) / (channel_std + 1e-6)

print("Per-channel mean after standardization (should be ~0):")
print(data_std.mean(dim=("latitude", "longitude")).values.round(4))
print("Per-channel std after standardization (should be ~1):")
print(data_std.std(dim=("latitude", "longitude")).values.round(4))


# %% [markdown]
# ## 6. Patchify with `xrpatcher`
#
# `XRDAPatcher` splits the array into overlapping spatial patches.
# Each patch retains the original coordinate metadata, which we use later
# during reconstruction.

# %%
PATCHES = {"latitude": 32, "longitude": 32}
STRIDES = {"latitude": 16, "longitude": 16}

patcher = XRDAPatcher(
    da=data_std,
    patches=PATCHES,
    strides=STRIDES,
    check_full_scan=False,  # allow non-aligned grid edges
)

print(patcher)
print(f"Total patches: {len(patcher)}")

# Inspect the first patch to verify the coordinate structure is preserved.
patch0 = patcher[0].load()
print(f"\nFirst patch shape  : {patch0.shape}")
lat_min = float(patch0.latitude.min())
lat_max = float(patch0.latitude.max())
lon_min = float(patch0.longitude.min())
lon_max = float(patch0.longitude.max())
print(f"Latitude  range    : {lat_min:.2f} - {lat_max:.2f}")
print(f"Longitude range    : {lon_min:.2f} - {lon_max:.2f}")


# %% [markdown]
# ## 7. Build a `grain.MapDataset` backed by patch indices
#
# `grain.MapDataset.source(indices)` creates a dataset from a plain Python
# list of integer indices.  We then `.map()` a function that retrieves the
# actual patch data from the `XRDAPatcher`.
#
# Each record returned from `fetch_patch` is a dictionary containing:
# - `"x"` — the patch data as a `float32` NumPy array of shape `(C, H, W)`
# - `"lat"` — 2-D latitude grid broadcast to `(H, W)`
# - `"lon"` — 2-D longitude grid broadcast to `(H, W)`
# - `"patch_index"` — scalar integer index for bookkeeping


# %%
def fetch_patch(idx: int) -> dict:
    """Retrieve a single patch from the patcher and package it as a dict.

    Args:
        idx: Integer index into the patcher.

    Returns:
        Dictionary with keys ``x``, ``lat``, ``lon``, and ``patch_index``.
    """
    patch = patcher[idx].load()

    lat = patch.latitude.values  # shape (H,)
    lon = patch.longitude.values  # shape (W,)

    # Broadcast 1-D coordinate arrays to 2-D spatial grids.
    h, w = patch.shape[-2], patch.shape[-1]
    lat2d = np.broadcast_to(lat[:, None], (h, w)).astype(np.float32)
    lon2d = np.broadcast_to(lon[None, :], (h, w)).astype(np.float32)

    return {
        "x": patch.values.astype(np.float32),  # (C, H, W)
        "lat": lat2d,  # (H, W)
        "lon": lon2d,  # (H, W)
        "patch_index": np.int32(idx),
    }


# Build the grain dataset pipeline:
#   source(indices) -> map(fetch_patch) -> batch(batch_size)
grain_ds = (
    grain.MapDataset.source(list(range(len(patcher))))
    .map(fetch_patch)
    .batch(batch_size=8)
)

first_batch = grain_ds[0]

print(f"Number of batches  : {len(grain_ds)}")
print(f"Batch x shape      : {first_batch['x'].shape}")  # (B, C, H, W)
print(f"Batch lat shape    : {first_batch['lat'].shape}")  # (B, H, W)
print(f"Batch lon shape    : {first_batch['lon'].shape}")  # (B, H, W)


# %% [markdown]
# ## 8. Apply a JAX model step on each batch
#
# We treat `grain` as the data pipeline and `jax` as the compute backend.
# The "model" here is a simple energy proxy that compresses the channel
# dimension:
#
#   **prediction** = mean(x² over channels)
#
# This produces a patch-sized scalar field — the kind of output a real
# regression model might return.


# %%
def model_step(x_batch: np.ndarray) -> np.ndarray:
    """Apply a simple JAX model step to a batch of patches.

    Computes the mean channel-wise squared activation as a scalar proxy
    output of shape ``(B, H, W)``.

    Args:
        x_batch: NumPy array of shape ``(B, C, H, W)``.

    Returns:
        NumPy array of shape ``(B, H, W)``.
    """
    jx = jnp.array(x_batch)  # move to JAX device
    pred = jnp.mean(jx**2, axis=1)  # channel reduction -> (B, H, W)
    return np.asarray(pred)  # back to NumPy for xrpatcher


all_preds: list[np.ndarray] = []

for batch_idx in range(len(grain_ds)):
    batch = grain_ds[batch_idx]
    pred = model_step(batch["x"])  # (B, H, W)
    all_preds.append(pred)

print(f"Processed {len(all_preds)} batches")
print(f"Single prediction batch shape: {all_preds[0].shape}")


# %% [markdown]
# ## 9. Reconstruct predictions back onto the original coordinates
#
# `patcher.reconstruct()` stitches per-patch predictions back into a
# full-resolution `DataArray` aligned on the original lat/lon grid.
#
# We use an edge-tapered weight to discount patch border pixels, which
# reduces seam artefacts in the overlapping regions.

# %%
# Build an edge-tapered weight: border pixels receive zero weight.
lat_p, lon_p = PATCHES["latitude"], PATCHES["longitude"]
rec_weight = np.ones((lat_p, lon_p), dtype=np.float32)
border = 4  # pixels to zero out at each edge
rec_weight[:border] = 0.0
rec_weight[-border:] = 0.0
rec_weight[:, :border] = 0.0
rec_weight[:, -border:] = 0.0

# Flatten the list of (B, H, W) arrays into individual (H, W) items.
flat_preds = [item for batch in all_preds for item in batch]

reconstructed = patcher.reconstruct(
    flat_preds,
    dims_labels=["latitude", "longitude"],
    weight=rec_weight,
)

print(reconstructed)
print(f"\nReconstructed shape: {reconstructed.shape}")


# %% [markdown]
# ## 10. Visualise the pipeline results

# %%
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Original wind speed field
    data.sel(variable="wind_speed").plot(ax=axes[0], cmap="viridis")
    axes[0].set_title("Wind speed (m/s)")

    # Standardised wind speed
    data_std.sel(variable="wind_speed").plot(ax=axes[1], cmap="RdBu_r")
    axes[1].set_title("Standardised wind speed")

    # First patch, first channel (u)
    axes[2].imshow(first_batch["x"][0, 0], origin="lower", cmap="RdBu_r")
    axes[2].set_title("First batch, patch 0, channel u")

    # Reconstructed prediction field
    reconstructed.plot(ax=axes[3], cmap="magma")
    axes[3].set_title("Reconstructed prediction")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not available — skipping visualisation.")


# %% [markdown]
# ## Summary
#
# | Step | Tool |
# |------|------|
# | Coordinate-aware patch extraction | `xrpatcher.XRDAPatcher` |
# | Dataset composition and batching | `grain.MapDataset` |
# | Accelerated model step | `jax` / `jax.numpy` |
# | Reconstruction to original grid | `xrpatcher.XRDAPatcher.reconstruct` |
#
# **Key takeaways**:
#
# - `xrpatcher` handles all the coordinate bookkeeping; patches always carry
#   their original lat/lon metadata.
# - `grain.MapDataset.source(indices).map(fn).batch(B)` gives a clean,
#   composable data pipeline with no custom boilerplate.
# - JAX and NumPy interoperate transparently via `jnp.array()` /
#   `np.asarray()`, so the prediction loop is framework-agnostic.
# - `patcher.reconstruct()` accepts plain NumPy arrays and returns an
#   `xarray.DataArray` aligned to the original grid.
