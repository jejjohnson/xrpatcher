# XRPatcher - A ML-Oriented Generic Patcher for `xarray` Data Structures

[**Installation**](#installation)
| [**Examples**](#examples)

[![Python Versions](https://img.shields.io/pypi/pyversions/xrpatcher)](https://pypi.org/project/xrpatcher/)
[![PyPI version](https://badge.fury.io/py/xrpatcher.svg)](https://badge.fury.io/py/xrpatcher)
![codestyle](https://img.shields.io/badge/codestyle-ruff-black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/pytorch_integration.py)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/jejjohnson/xrpatcher)
[![Tests](https://github.com/jejjohnson/xrpatcher/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/xrpatcher/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/xrpatcher/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/xrpatcher/actions/workflows/lint.yml)
[![Format](https://github.com/jejjohnson/xrpatcher/actions/workflows/format.yml/badge.svg)](https://github.com/jejjohnson/xrpatcher/actions/workflows/format.yml)
[![Typecheck](https://github.com/jejjohnson/xrpatcher/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/xrpatcher/actions/workflows/typecheck.yml)

* J. Emmanuel Johnson
* Quentin Febvre

## About

This is a lightweight package to create patch *items* from xarray data structures.
This makes it more compatible with machine learning datasets and dataloaders like PyTorch or TensorFlow.
The user simply needs to define the patch dimensions and the stride dimensions and you are good to go!
It also reconstructs (or unpatchifies) from arbitrary patches which allows for more robust inference procedures, e.g. to account for border effects from CNN models.


## ⏩ Examples<a id="examples"></a>

### Quick Example

```python
import xarray as xr
from xrpatcher import XRDAPatcher

# load demo dataset
data = xr.tutorial.load_dataset("eraint_uvz")

# extract demo dataarray
data = data.u[..., :240, :360]

# Instantiate the patching logic for training
patches = dict(longitude=30, latitude=30)
train_patcher = XRDAPatcher(
    da=data,
    patches=patches,
    strides=patches,        # No Overlap
    check_full_scan=True    # check no extra dimensions
)

# Instantiate the patching logic for testing
patches = dict(longitude=30, latitude=30)
strides = dict(longitude=5, latitude=5)

test_patcher = XRDAPatcher(
    da=data,
    patches=patches,
    strides=strides,        # Overlap
    check_full_scan=True    # check no extra dimensions
)
```

---
### Caching Repeated Patch Access

If the same patch index is accessed repeatedly, you can enable in-memory caching:

```python
patcher = XRDAPatcher(
    da=data,
    patches=patches,
    strides=strides,
    cache=True,
    preload=True,  # eagerly load cached patches into memory
)

patch = patcher[0]     # sliced and cached on first access
patch = patcher[0]     # returned from cache on later access
patcher.clear_cache()  # free cached patches
```

`preload=True` is useful when your `DataArray` may be backed by lazy arrays and you
want cached patches to be loaded into RAM when they are first accessed and cached.
If `preload=False`, the cache stores the original patch objects as-is.
`preload=True` requires `cache=True`.

You can compare cached and uncached repeated access with a simple benchmark:

```python
from time import perf_counter

uncached = XRDAPatcher(da=data, patches=patches, strides=strides)
cached = XRDAPatcher(
    da=data,
    patches=patches,
    strides=strides,
    cache=True,
    preload=True,
)

for label, patcher in [("uncached", uncached), ("cached", cached)]:
    start = perf_counter()
    for _ in range(5):
        _ = patcher[0]
    print(label, perf_counter() - start)
```

---
### Extended Examples

**Caching Repeated Patch Access** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/caching.py)

> We demonstrate how in-memory caching speeds up repeated access to the same patches, with a benchmark comparing cached and uncached access.

**Patching Crash Course** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/patching_cc.py)

> We have an extended example where we demonstrate some of the ways to do the reconstruction!

**PyTorch Integration** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/pytorch_integration.py)

> We have an extended example where we demonstrate some nifty PyTorch Integration.

**Example 3: PyTorch Integration Concatenate Multiple domain** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/xrpatcher_concat_torch_dataloading.py)

> We demonstrate in this example how this tool can be used to create more complex dataloading like jointly training on separate regions

**TorchGeo DataModule Integration** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/torchgeo_datamodule_integration.py)

> We demonstrate how to turn a TorchGeo batch into georeferenced `xarray` objects, patch them with `xrpatcher`, and reconstruct patch-wise outputs.

---

## 🛠️ Installation<a id="installation"></a>

### pip

```bash
pip install xrpatcher
```

### uv

```bash
uv add xrpatcher
```

### Development installation

Clone the repository and install with all dev dependencies using uv:

```bash
git clone https://github.com/jejjohnson/xrpatcher.git
cd xrpatcher
uv sync --all-groups
```

### Releasing to PyPI

Releases are managed by [release-please](https://github.com/googleapis/release-please).
When a release PR is merged into `main`, release-please creates a GitHub Release, which
triggers the `publish-pypi.yml` workflow that builds and publishes the package to PyPI
using OIDC trusted publishing (no API tokens required).

For publishing to work, a maintainer with owner access on PyPI must add this repository
as a trusted publisher with the following settings:

| Field | Value |
|---|---|
| Owner | `jejjohnson` |
| Repository | `xrpatcher` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

---
## Inspiration

There are a few other packages that gave us inspiration for this.

* [xbatcher](https://xbatcher.readthedocs.io/en/latest/index.html)
* [PatchExtractor (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html)
