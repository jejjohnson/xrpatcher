# XRPatcher - A ML-Oriented Generic Patcher for `xarray` Data Structures

[**Installation**](#installation)
| [**Examples**](#examples)

![pyver](https://img.shields.io/badge/python-3.9%203.10%203.11_-red)
[![PyPI version](https://badge.fury.io/py/xrpatcher.svg)](https://badge.fury.io/py/xrpatcher)
![codestyle](https://img.shields.io/badge/codestyle-black-black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/notebooks/pytorch_integration.ipynb)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/jejjohnson/xrpatcher)

* J. Emmanuel Johnson
* Quentin Febvre

---
## About

This is a lightweight package to create patch *items* from xarray data structures.
This makes it more compatible with machine learning datasets and dataloaders like PyTorch or TensorFlow.
The user simply needs to define the patch dimensions and the stride dimensions and you are good to go!
It also reconstructs (or unpatchifies) from arbitrary patches which allows for more robust inference procedures, e.g. to account for border effects from CNN models.

---
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

### Extended Example

**Example 1: Patching Crash Course** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/notebooks/patching_cc.ipynb)

> We have an extended example where we demonstrate some of the ways to do the reconstruction!

**Example 2: PyTorch Integration** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/notebooks/pytorch_integration.ipynb)

> We have an extended example where we demonstrate some nifty PyTorch Integration.

---

## 🛠️ Installation<a id="installation"></a>


### pip

We can directly install it via pip from the

```bash
pip install xrpatcher
```

### Cloning

We can also clone the git repository

```bash
git clone https://github.com/jejjohnson/xrpatcher.git
cd xrpatcher
```

#### Conda Environment (RECOMMENDED)

We use conda/mamba as our package manager. To install from the provided environment files
run the following command.

```bash
mamba env create -n environment.yaml
```

#### poetry

The easiest way to get started is to simply use the poetry package which installs all necessary dev packages as well

```bash
poetry install
```

#### pip

We can also install via `pip` as well

```bash
pip install .
```

---
## Inspiration

There are a few other packages that gave us inspiration for this.

* [xbatcher](https://xbatcher.readthedocs.io/en/latest/index.html)
* [PatchExtractor (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html)
