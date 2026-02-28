# XRPatcher - A ML-Oriented Generic Patcher for `xarray` Data Structures

[**Installation**](#installation)
| [**Examples**](#examples)

![pyver](https://img.shields.io/badge/python-3.12%203.13-red)
[![PyPI version](https://badge.fury.io/py/xrpatcher.svg)](https://badge.fury.io/py/xrpatcher)
![codestyle](https://img.shields.io/badge/codestyle-ruff-black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/pytorch_integration.py)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/jejjohnson/xrpatcher)

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
### Extended Examples

**Patching Crash Course** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/patching_cc.py)

> We have an extended example where we demonstrate some of the ways to do the reconstruction!

**PyTorch Integration** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/pytorch_integration.py)

> We have an extended example where we demonstrate some nifty PyTorch Integration.

**Example 3: PyTorch Integration Concatenate Multiple domain** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/xrpatcher_concat_torch_dataloading.py)

> We demonstrate in this example how this tool can be used to create more complex dataloading like jointly training on separate regions

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

---
## Inspiration

There are a few other packages that gave us inspiration for this.

* [xbatcher](https://xbatcher.readthedocs.io/en/latest/index.html)
* [PatchExtractor (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html)
