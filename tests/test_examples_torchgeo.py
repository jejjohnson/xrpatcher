"""Tests for the patterns demonstrated in
``examples/torchgeo_datamodule_integration.py``.

These tests cover:

- ``pixel_centers`` coordinate helper
- ``sample_to_dataarray`` - converting a tensor + bounding box to ``xr.DataArray``
- ``mask_to_dataarray`` - converting a mask tensor to ``xr.DataArray``
- ``batch_to_xarray_samples`` - unpacking a TorchGeo-like batch dict
- ``synthetic_torchgeo_batch`` - synthetic batch generation
- ``XrTorchGeoDataset`` - patch extraction and reconstruction from a
  georeferenced sample

``torch`` is treated as an optional test dependency.  ``torchgeo`` itself is
also optional; the tests use the same synthetic-batch fallback that the
example uses when the real dataset is unavailable.
"""

import collections
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
# BoundingBox (mirrors the namedtuple defined in the example)
# ---------------------------------------------------------------------------

BoundingBox = collections.namedtuple(
    "BoundingBox", ["minx", "maxx", "miny", "maxy", "mint", "maxt"]
)


# ---------------------------------------------------------------------------
# Helpers copied verbatim from the example (kept in sync with the example)
# ---------------------------------------------------------------------------


def pixel_centers(start: float, stop: float, size: int) -> np.ndarray:
    """Return pixel-centre coordinates for ``size`` pixels spanning start→stop.

    Args:
        start: Left/bottom edge of the first pixel.
        stop: Right/top edge of the last pixel.
        size: Number of pixels.

    Returns:
        1-D NumPy array of length ``size`` with pixel-centre values.
    """
    step = (stop - start) / size
    return start + step * (0.5 + np.arange(size))


def sample_to_dataarray(
    image: "torch.Tensor",
    bbox: BoundingBox,
    crs: str,
) -> xr.DataArray:
    """Convert a TorchGeo-style image tensor and bounding box to xarray.

    Args:
        image: Tensor of shape ``(bands, H, W)``.
        bbox: Bounding box with ``minx``, ``maxx``, ``miny``, ``maxy``.
        crs: CRS string stored in the DataArray attrs.

    Returns:
        ``DataArray`` with dimensions ``(band, y, x)`` and pixel-centre
        coordinates derived from ``bbox``.
    """
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


def mask_to_dataarray(
    mask: "torch.Tensor",
    image_da: xr.DataArray,
) -> xr.DataArray:
    """Convert a TorchGeo-style mask tensor to xarray, sharing coords with image.

    Args:
        mask: Tensor of shape ``(H, W)``.
        image_da: DataArray whose ``y`` and ``x`` coordinates are reused.

    Returns:
        ``DataArray`` with dimensions ``(y, x)`` and the same spatial
        coordinates as ``image_da``.
    """
    mask_np = mask.detach().cpu().numpy()
    return xr.DataArray(
        mask_np,
        dims=("y", "x"),
        coords={"y": image_da.y, "x": image_da.x},
        attrs=image_da.attrs,
    )


def batch_to_xarray_samples(
    batch: dict,
) -> list[tuple[xr.DataArray, xr.DataArray]]:
    """Unpack a TorchGeo-like batch dict into (image_da, mask_da) pairs.

    Args:
        batch: Dictionary with keys ``image`` (tensor), ``mask`` (tensor),
            ``bbox`` (list of BoundingBox), and optional ``crs``.

    Returns:
        List of ``(image_da, mask_da)`` tuples, one per sample in the batch.

    Raises:
        TypeError: If ``image`` or ``mask`` are not tensors, or if ``bbox``
            is not a sequence.
    """
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
    patch_size: int = 64,
) -> dict:
    """Create a synthetic TorchGeo-style batch for testing.

    Args:
        batch_size: Number of samples in the batch.
        patch_size: Spatial size (H = W = patch_size) of each sample.

    Returns:
        Dictionary with keys ``image``, ``mask``, ``bbox``, and ``crs``
        matching the TorchGeo batch contract.
    """
    base_x = np.linspace(-1, 1, patch_size)
    base_y = np.linspace(-1, 1, patch_size)
    yy, xx = np.meshgrid(base_y, base_x, indexing="ij")

    images, masks, bboxes = [], [], []
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
                500_064.0 + 128.0 * idx,
                4_100_000.0,
                4_100_064.0,
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


# ---------------------------------------------------------------------------
# XrTorchGeoDataset (mirrors torchgeo_datamodule_integration.py)
# ---------------------------------------------------------------------------


class XrTorchGeoDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that serves image and mask patches from ``XRDAPatcher``.

    Args:
        image_patcher: Patcher for the multi-band image DataArray.
        mask_patcher: Optional patcher for the corresponding mask DataArray.
            Must expose the same number of patches as ``image_patcher``.
    """

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
        """Reconstruct from an iterable of batches collected from the loader."""
        return self.image_patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __len__(self):
        return len(self.image_patcher)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def geo_batch():
    """A small 2-sample synthetic TorchGeo-like batch with 64x64 images."""
    return synthetic_torchgeo_batch(batch_size=2, patch_size=64)


@pytest.fixture
def image_da(geo_batch):
    """``xr.DataArray`` for the first sample in the batch."""
    samples = batch_to_xarray_samples(geo_batch)
    return samples[0][0]


@pytest.fixture
def mask_da(geo_batch):
    """``xr.DataArray`` for the first sample's mask in the batch."""
    samples = batch_to_xarray_samples(geo_batch)
    return samples[0][1]


@pytest.fixture
def geo_dataset(image_da, mask_da):
    """``XrTorchGeoDataset`` with 16x16 patches (no strict full-scan check)."""
    patches = {"y": 16, "x": 16}
    strides = {"y": 8, "x": 8}
    image_patcher = XRDAPatcher(
        da=image_da, patches=patches, strides=strides, check_full_scan=True
    )
    mask_patcher = XRDAPatcher(
        da=mask_da, patches=patches, strides=strides, check_full_scan=True
    )
    return XrTorchGeoDataset(image_patcher=image_patcher, mask_patcher=mask_patcher)


# ---------------------------------------------------------------------------
# Tests: pixel_centers
# ---------------------------------------------------------------------------


def test_pixel_centers_count():
    """pixel_centers returns an array with the requested number of elements."""
    centers = pixel_centers(0.0, 4.0, 4)
    assert len(centers) == 4


def test_pixel_centers_values():
    """pixel_centers values are correct midpoints."""
    centers = pixel_centers(0.0, 4.0, 4)
    np.testing.assert_allclose(centers, [0.5, 1.5, 2.5, 3.5])


def test_pixel_centers_does_not_include_endpoints():
    """First and last pixel centres are strictly inside the interval."""
    centers = pixel_centers(0.0, 1.0, 10)
    assert centers[0] > 0.0
    assert centers[-1] < 1.0


# ---------------------------------------------------------------------------
# Tests: sample_to_dataarray
# ---------------------------------------------------------------------------


def test_sample_to_dataarray_dims():
    """Converted DataArray has dimensions (band, y, x)."""
    image = torch.zeros(3, 32, 32)
    bbox = BoundingBox(0.0, 32.0, 0.0, 32.0, 0, 1)
    da = sample_to_dataarray(image, bbox, "EPSG:4326")
    assert da.dims == ("band", "y", "x")


def test_sample_to_dataarray_shape():
    """Converted DataArray shape matches the input tensor."""
    image = torch.zeros(3, 32, 32)
    bbox = BoundingBox(0.0, 32.0, 0.0, 32.0, 0, 1)
    da = sample_to_dataarray(image, bbox, "EPSG:4326")
    assert da.shape == (3, 32, 32)


def test_sample_to_dataarray_crs_attr():
    """CRS string is preserved in the DataArray attrs."""
    image = torch.zeros(2, 16, 16)
    bbox = BoundingBox(500_000.0, 500_256.0, 4_100_000.0, 4_100_256.0, 0, 1)
    da = sample_to_dataarray(image, bbox, "EPSG:32631")
    assert da.attrs["crs"] == "EPSG:32631"


def test_sample_to_dataarray_coord_count():
    """y and x coordinate arrays have the correct lengths."""
    h, w = 24, 48
    image = torch.zeros(1, h, w)
    bbox = BoundingBox(0.0, float(w), 0.0, float(h), 0, 1)
    da = sample_to_dataarray(image, bbox, "unknown")
    assert len(da.y) == h
    assert len(da.x) == w


# ---------------------------------------------------------------------------
# Tests: mask_to_dataarray
# ---------------------------------------------------------------------------


def test_mask_to_dataarray_dims():
    """Mask DataArray has dimensions (y, x)."""
    image = torch.zeros(3, 16, 16)
    bbox = BoundingBox(0.0, 16.0, 0.0, 16.0, 0, 1)
    image_da = sample_to_dataarray(image, bbox, "EPSG:4326")
    mask = torch.zeros(16, 16)
    mask_da = mask_to_dataarray(mask, image_da)
    assert mask_da.dims == ("y", "x")


def test_mask_to_dataarray_shares_coords():
    """Mask DataArray shares y and x coordinates with the image DataArray."""
    image = torch.zeros(3, 16, 16)
    bbox = BoundingBox(0.0, 16.0, 0.0, 16.0, 0, 1)
    image_da = sample_to_dataarray(image, bbox, "EPSG:4326")
    mask = torch.zeros(16, 16)
    mask_da = mask_to_dataarray(mask, image_da)
    np.testing.assert_array_equal(mask_da.y.values, image_da.y.values)
    np.testing.assert_array_equal(mask_da.x.values, image_da.x.values)


# ---------------------------------------------------------------------------
# Tests: batch_to_xarray_samples
# ---------------------------------------------------------------------------


def test_batch_to_xarray_samples_length(geo_batch):
    """Returns one (image_da, mask_da) pair per sample in the batch."""
    samples = batch_to_xarray_samples(geo_batch)
    assert len(samples) == len(geo_batch["image"])


def test_batch_to_xarray_samples_types(geo_batch):
    """Each pair contains an image and mask DataArray."""
    samples = batch_to_xarray_samples(geo_batch)
    for image_da, mask_da in samples:
        assert isinstance(image_da, xr.DataArray)
        assert isinstance(mask_da, xr.DataArray)


def test_batch_to_xarray_samples_image_shape(geo_batch):
    """Image DataArrays have the expected shape matching the input batch."""
    samples = batch_to_xarray_samples(geo_batch)
    _, h, w = geo_batch["image"].shape[1:]
    for image_da, _ in samples:
        assert image_da.shape == (3, h, w)


# ---------------------------------------------------------------------------
# Tests: synthetic_torchgeo_batch
# ---------------------------------------------------------------------------


def test_synthetic_batch_keys():
    """Synthetic batch contains the required TorchGeo keys."""
    batch = synthetic_torchgeo_batch(batch_size=2, patch_size=32)
    assert {"image", "mask", "bbox", "crs"} <= set(batch.keys())


def test_synthetic_batch_image_shape():
    """Synthetic batch image tensor has the correct shape."""
    batch = synthetic_torchgeo_batch(batch_size=3, patch_size=32)
    assert batch["image"].shape == (3, 3, 32, 32)


def test_synthetic_batch_mask_shape():
    """Synthetic batch mask tensor has the correct shape."""
    batch = synthetic_torchgeo_batch(batch_size=3, patch_size=32)
    assert batch["mask"].shape == (3, 32, 32)


def test_synthetic_batch_bbox_count():
    """Synthetic batch has one bounding box per sample."""
    batch = synthetic_torchgeo_batch(batch_size=4, patch_size=32)
    assert len(batch["bbox"]) == 4


# ---------------------------------------------------------------------------
# Tests: XrTorchGeoDataset
# ---------------------------------------------------------------------------


def test_geo_dataset_len(geo_dataset, image_da):
    """Dataset length matches the underlying image patcher."""
    from xrpatcher import XRDAPatcher

    patcher = XRDAPatcher(
        da=image_da,
        patches={"y": 16, "x": 16},
        strides={"y": 8, "x": 8},
        check_full_scan=True,
    )
    assert len(geo_dataset) == len(patcher)


def test_geo_dataset_item_has_image_and_mask(geo_dataset):
    """Each item from the dataset contains both 'image' and 'mask' tensors."""
    item = geo_dataset[0]
    assert "image" in item
    assert "mask" in item


def test_geo_dataset_item_image_shape(geo_dataset):
    """Image tensor in each item has the expected patch shape."""
    item = geo_dataset[0]
    # 3 bands, 16x16 patch
    assert item["image"].shape == (3, 16, 16)


def test_geo_dataset_item_mask_shape(geo_dataset):
    """Mask tensor in each item has the expected patch shape."""
    item = geo_dataset[0]
    assert item["mask"].shape == (16, 16)


def test_geo_dataset_mismatched_patchers_raises(image_da, mask_da):
    """Providing patchers with different lengths raises ``ValueError``."""
    image_patcher = XRDAPatcher(
        da=image_da,
        patches={"y": 16, "x": 16},
        strides={"y": 8, "x": 8},
        check_full_scan=True,
    )
    # Use a different patch/stride that produces a different number of patches
    mask_patcher = XRDAPatcher(
        da=mask_da,
        patches={"y": 32, "x": 32},
        strides={"y": 16, "x": 16},
        check_full_scan=True,
    )
    with pytest.raises(ValueError, match="same number of patches"):
        XrTorchGeoDataset(image_patcher=image_patcher, mask_patcher=mask_patcher)


def test_geo_dataset_reconstruct_shape(geo_dataset, image_da):
    """Reconstruction from all batches returns a DataArray with the correct shape."""
    loader = torch.utils.data.DataLoader(geo_dataset, batch_size=4, shuffle=False)

    rec_weight = np.ones((16, 16), dtype=np.float32)
    rec_weight[:2] = 0.0
    rec_weight[-2:] = 0.0
    rec_weight[:, :2] = 0.0
    rec_weight[:, -2:] = 0.0

    rec = geo_dataset.reconstruct_from_batches(
        (batch["image"][:, 0].cpu().numpy() for batch in loader),
        dims_labels=["y", "x"],
        weight=rec_weight,
    )

    assert isinstance(rec, xr.DataArray)
    assert rec.shape == (image_da.sizes["y"], image_da.sizes["x"])
