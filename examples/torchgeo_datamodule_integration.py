# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/torchgeo_datamodule_integration.py)

# %% [markdown]
# # TorchGeo DataModule Integration Demo
#
# This tutorial shows how to bridge a TorchGeo `GeoDataModule` with `xrpatcher`.
# The main idea is:
#
# 1. Ask a TorchGeo `DataModule` for a batch with geospatial metadata.
# 2. Convert each sample into an `xarray.DataArray` with `x`/`y` coordinates.
# 3. Patchify the sample with `xrpatcher`.
# 4. Wrap the patches in a PyTorch `Dataset`/`DataLoader`.
# 5. Reconstruct a georeferenced prediction map from patch-wise outputs.
#
# The code uses `InriaAerialImageLabelingDataModule` when TorchGeo and local data
# are available. Otherwise it falls back to a synthetic TorchGeo-like batch so
# the workflow remains runnable end to end.

# %%
# Install dependencies (Colab)
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if importlib.util.find_spec("xrpatcher") is None:
    subprocess.run(["pip", "install", "xrpatcher"], check=True)
if importlib.util.find_spec("torch") is None:
    subprocess.run(
        [
            "pip",
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        check=True,
    )

# %%
import collections
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from xrpatcher import XRDAPatcher


try:
    from torchgeo.datamodules import InriaAerialImageLabelingDataModule
except ImportError:
    InriaAerialImageLabelingDataModule = None


BoundingBox = collections.namedtuple(
    "BoundingBox", ["minx", "maxx", "miny", "maxy", "mint", "maxt"]
)


# %% [markdown]
# ## Torch dataset wrapper for `xrpatcher`
#
# This follows the same pattern as the existing PyTorch example, but it returns a
# dictionary with image and mask tensors so it looks more like a TorchGeo batch.


# %%
class XrTorchGeoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_patcher: XRDAPatcher,
        mask_patcher: XRDAPatcher | None = None,
    ):
        if mask_patcher is not None and len(image_patcher) != len(mask_patcher):
            msg = "Image and mask patchers must expose the same number of patches."
            raise ValueError(msg)

        self.image_patcher = image_patcher
        self.mask_patcher = mask_patcher

    def __getitem__(self, idx):
        sample = {
            "image": torch.as_tensor(self.image_patcher[idx].load().values.copy()),
        }
        if self.mask_patcher is not None:
            sample["mask"] = torch.as_tensor(
                self.mask_patcher[idx].load().values.copy()
            )
        return sample

    def reconstruct_from_batches(self, batches, **rec_kws):
        return self.image_patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __len__(self):
        return len(self.image_patcher)


# %% [markdown]
# ## Helpers for converting TorchGeo samples to `xarray`
#
# TorchGeo batches are dictionaries with tensor data plus geospatial metadata such
# as `bbox` and `crs`. We convert each sample independently because samples in the
# same batch can come from different locations.


# %%
def pixel_centers(start: float, stop: float, size: int) -> np.ndarray:
    step = (stop - start) / size
    return start + step * (0.5 + np.arange(size))


def sample_to_dataarray(
    image: torch.Tensor,
    bbox: BoundingBox,
    crs: str,
) -> xr.DataArray:
    image_np = image.detach().cpu().numpy()
    y_coords = pixel_centers(float(bbox.maxy), float(bbox.miny), image_np.shape[-2])
    x_coords = pixel_centers(float(bbox.minx), float(bbox.maxx), image_np.shape[-1])

    return xr.DataArray(
        image_np,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(image_np.shape[0]),
            "y": y_coords,
            "x": x_coords,
        },
        attrs={"crs": crs},
    )


def mask_to_dataarray(mask: torch.Tensor, image_da: xr.DataArray) -> xr.DataArray:
    mask_np = mask.detach().cpu().numpy()
    return xr.DataArray(
        mask_np,
        dims=("y", "x"),
        coords={"y": image_da.y, "x": image_da.x},
        attrs=image_da.attrs,
    )


def batch_to_xarray_samples(
    batch: dict[str, object],
) -> list[tuple[xr.DataArray, xr.DataArray]]:
    images = batch["image"]
    masks = batch["mask"]
    bboxes = batch["bbox"]
    crs = batch.get("crs", "unknown")

    if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
        msg = "Expected a TorchGeo-like batch with tensor image and mask entries."
        raise TypeError(msg)
    if not isinstance(bboxes, list | tuple):
        msg = "Expected bbox metadata to be a sequence of bounding boxes."
        raise TypeError(msg)
    crs_items = [crs] * len(images) if isinstance(crs, str) else list(crs)

    samples = []
    for image, mask, bbox, crs_item in zip(
        images, masks, bboxes, crs_items, strict=True
    ):
        image_da = sample_to_dataarray(image, bbox, str(crs_item))
        samples.append((image_da, mask_to_dataarray(mask, image_da)))

    return samples


def synthetic_torchgeo_batch(
    batch_size: int = 2,
    patch_size: int = 256,
) -> dict[str, object]:
    base_x = np.linspace(-1, 1, patch_size)
    base_y = np.linspace(-1, 1, patch_size)
    yy, xx = np.meshgrid(base_y, base_x, indexing="ij")

    images = []
    masks = []
    bboxes = []
    for idx in range(batch_size):
        shift = 0.1 * idx
        red = np.exp(-((xx - shift) ** 2 + yy**2) / 0.2)
        green = np.sin(np.pi * (xx + shift)) * np.cos(np.pi * yy)
        blue = np.cos(2 * np.pi * yy) + xx
        image = np.stack([red, green, blue]).astype(np.float32)
        mask = ((red + green) > 0.4).astype(np.int64)

        images.append(image)
        masks.append(mask)
        bboxes.append(
            BoundingBox(
                500_000.0 + 128.0 * idx,
                500_256.0 + 128.0 * idx,
                4_100_000.0,
                4_100_256.0,
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
            )
        )

    return {
        "image": torch.from_numpy(np.stack(images)),
        "mask": torch.from_numpy(np.stack(masks)),
        "bbox": bboxes,
        "crs": ["EPSG:32631"] * batch_size,
    }


def load_torchgeo_batch(
    root: str | os.PathLike[str] | None = None,
    batch_size: int = 2,
    patch_size: int = 256,
) -> tuple[dict[str, object], str]:
    if InriaAerialImageLabelingDataModule is None:
        return synthetic_torchgeo_batch(batch_size=batch_size, patch_size=patch_size), (
            "synthetic fallback (install torchgeo to use a real DataModule)"
        )

    if root is None:
        return synthetic_torchgeo_batch(batch_size=batch_size, patch_size=patch_size), (
            "synthetic fallback (set TORCHGEO_INRIA_ROOT to a local Inria dataset)"
        )

    data_root = Path(root)
    if not data_root.exists():
        return synthetic_torchgeo_batch(batch_size=batch_size, patch_size=patch_size), (
            f"synthetic fallback ({data_root} does not exist)"
        )

    datamodule = InriaAerialImageLabelingDataModule(
        root=str(data_root),
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=0,
    )
    datamodule.setup(stage="fit")
    return next(iter(datamodule.train_dataloader())), "torchgeo datamodule"


def as_rgb(image_da: xr.DataArray) -> np.ndarray:
    rgb = image_da.values
    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    elif rgb.shape[0] == 2:
        rgb = np.concatenate([rgb, rgb[:1]], axis=0)
    else:
        rgb = rgb[:3]

    rgb = np.moveaxis(rgb, 0, -1)
    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()
    return rgb


# %% [markdown]
# ## 1. Get a batch from TorchGeo
#
# If you have the Inria Aerial Image Labeling dataset locally, point the
# `TORCHGEO_INRIA_ROOT` environment variable at it before running this cell.
# Otherwise the tutorial uses a synthetic batch with the same keys as TorchGeo:
# `image`, `mask`, `bbox`, and `crs`.

# %%
data_root = os.environ.get("TORCHGEO_INRIA_ROOT")
geo_batch, batch_source = load_torchgeo_batch(root=data_root)

print(f"Loaded batch from: {batch_source}")
print(f"Batch keys: {sorted(geo_batch)}")
print(f"Image batch shape: {geo_batch['image'].shape}")
print(f"Mask batch shape: {geo_batch['mask'].shape}")

# %% [markdown]
# ## 2. Convert each TorchGeo sample into `xarray`
#
# Each TorchGeo sample has its own bounding box, so we convert them one by one.

# %%
xr_samples = batch_to_xarray_samples(geo_batch)
image_da, mask_da = xr_samples[0]

print(image_da)
print(mask_da)

# %% [markdown]
# ## 3. Patchify the georeferenced sample with `xrpatcher`
#
# Here we use overlapping spatial patches. The `band` dimension is kept intact
# while `xrpatcher` splits only along the `y` and `x` dimensions.

# %%
patches = {"y": 128, "x": 128}
strides = {"y": 64, "x": 64}

image_patcher = XRDAPatcher(
    da=image_da,
    patches=patches,
    strides=strides,
    check_full_scan=True,
)
mask_patcher = XRDAPatcher(
    da=mask_da,
    patches=patches,
    strides=strides,
    check_full_scan=True,
)

torch_ds = XrTorchGeoDataset(image_patcher=image_patcher, mask_patcher=mask_patcher)
dataloader = torch.utils.data.DataLoader(torch_ds, batch_size=4, shuffle=False)

first_patch = torch_ds[0]
print(f"Number of patches: {len(torch_ds)}")
print(f"Patch image shape: {first_patch['image'].shape}")
print(f"Patch mask shape: {first_patch['mask'].shape}")

# %% [markdown]
# ## 4. Reconstruct a georeferenced prediction map from patch batches
#
# In practice this generator would yield model outputs. For the demo, we use the
# first image channel as a simple proxy prediction and reconstruct it on the
# original coordinates.

# %%
rec_weight = np.ones((128, 128), dtype=np.float32)
border_epsilon = 1e-6
rec_weight[:8] = border_epsilon
rec_weight[-8:] = border_epsilon
rec_weight[:, :8] = border_epsilon
rec_weight[:, -8:] = border_epsilon

probability_map = torch_ds.reconstruct_from_batches(
    (torch.sigmoid(batch["image"][:, 0]).cpu().numpy() for batch in dataloader),
    dims_labels=["y", "x"],
    weight=rec_weight,
)

print(probability_map)

# %% [markdown]
# ## 5. Visualize the end-to-end workflow
#
# The four panels below show the original TorchGeo sample, one xrpatcher patch,
# the corresponding target mask, and the reconstructed patch-wise output.

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(as_rgb(image_da), origin="lower")
axes[0].set_title("TorchGeo sample")

axes[1].imshow(first_patch["image"][0], cmap="viridis", origin="lower")
axes[1].set_title("One xrpatcher patch")

mask_da.plot(ax=axes[2], add_colorbar=False, cmap="Greens")
axes[2].set_title("Target mask")

probability_map.plot(ax=axes[3], add_colorbar=False, cmap="magma")
axes[3].set_title("Reconstructed output")

for ax in axes:
    ax.set_xticks([], labels=None)
    ax.set_yticks([], labels=None)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaways
#
# * TorchGeo `DataModule`s already solve the problem of sampling georeferenced
#   batches and keeping the spatial metadata around.
# * Converting each sample to `xarray` makes those coordinates explicit for
#   `xrpatcher`.
# * `XRDAPatcher` then gives you fine-grained patch extraction, overlap control,
#   and reconstruction on top of the original TorchGeo sample geometry.
