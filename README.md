# XRPatcher - A ML-Oriented Generic Patcher for `xarray` Data Structures

* J. Emmanuel Johnson
* Quentin Febvre

---
## About

This is a lightweight package to create patch *items* from xarray data structures. 
This makes it more compatible with machine learning datasets and dataloaders like PyTorch or TensorFlow.
The user simply needs to define the patch dimensions and the stride dimensions and you are good to go!
It also reconstructs (or unpatchifies) from arbitrary patches which allows for more robust inference procedures, e.g. to account for border effects from CNN models.

---
## Example

```python
import xarray as xr
import torch
import itertools
from oceanbench import XRPatcher


# Easy Integration with PyTorch Datasets (and DataLoaders)
class XRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, batcher: XRPatcher, item_postpro=None):
        self.batcher = batcher
        self.postpro = item_postpro
    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.batcher[idx].load().values
        if self.postpro:
            item = self.postpro(item)
        return item
    def reconstruct_from_batches(
            self, batches: list(torch.Tensor), **rec_kws
        ) -> xr.Dataset:
        return self.batcher.reconstruct(
            [*itertools.chain(*batches)], **rec_kws
        )
    def __len__(self) -> int:
        return len(self.batcher)
    
# load demo dataset
data = xr.tutorial.load_dataset("eraint_uvz")

# Instantiate the patching logic for training
patches = dict(longitude=30, latitude=30)
train_patcher = XRPatcher(
    da=data,
    patches=patches,
    strides=patches,        # No Overlap
    check_full_scan=True    # check no extra dimensions
)

# Instantiate the patching logic for testing
patches = dict(longitude=30, latitude=30)
strides = dict(longitude=5, latitude=5)

test_patcher = XRPatcher(
    da=data,
    patches=patches,
    strides=strides,        # Overlap
    check_full_scan=True    # check no extra dimensions
)

# instantiate PyTorch DataSet
train_ds = XRTorchDataset(train_patcher, item_postpro=TrainingItem._make)
test_ds = XRTorchDataset(test_patcher, item_postpro=TrainingItem._make)

# instantiate PyTorch DataLoader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=False)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=10, shuffle=False)
```

---

## Installation Guide


### pip

We can directly install it via pip from the

```bash
pip install "git+https://github.com/jejjohnson/xrpatcher.git"
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