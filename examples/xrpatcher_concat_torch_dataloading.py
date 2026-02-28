# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/xrpatcher_concat_torch_dataloading.py)

# %% [markdown]
# # XRPatcher Concat Torch Dataloading Demo
#
# This example demonstrates how `xrpatcher` can be used to create more complex
# dataloading, like jointly training on separate geographic regions.

# %%
# Install dependencies (Colab)
try:
    import xrpatcher
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "xrpatcher"], check=True)

# %%
import itertools

import matplotlib.pyplot as plt
import torch
import xarray as xr

import xrpatcher

# %% [markdown]
# ## Dataset and DataLoader wrappers

# %%
class XrTorchDataset(torch.utils.data.Dataset):
    def __init__(self, patcher: xrpatcher.XRDAPatcher, item_postpro=None):
        self.patcher = patcher
        self.postpro = item_postpro

    def __getitem__(self, idx):
        item = self.patcher[idx].load().values
        if self.postpro:
            item = self.postpro(item)
        return item

    def reconstruct_from_batches(self, batches, **rec_kws):
        return self.patcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __len__(self):
        return len(self.patcher)


class XrConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *dses: XrTorchDataset):
        super().__init__(dses)

    def reconstruct_from_batches(self, batches, weight=None):
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.patcher.reconstruct(ds_items, weight=weight))

        return rec_das

# %% [markdown]
# ## Loading data and creating domain-specific patchers

# %%
ds = xr.tutorial.load_dataset("air_temperature")
ds = ds.sortby("lat")

domains = [
    dict(lat=slice(15, 35), lon=slice(200, 220)),
    dict(lat=slice(55, 75), lon=slice(300, 320)),
]

patching_kw = dict(patches=dict(time=720), strides=dict(time=200))

patcher1 = xrpatcher.XRDAPatcher(ds.air, domain_limits=domains[0], **patching_kw)
patcher2 = xrpatcher.XRDAPatcher(ds.air, domain_limits=domains[1], **patching_kw)

torch_ds1 = XrTorchDataset(patcher1)
torch_ds2 = XrTorchDataset(patcher2)

torch_ds = XrConcatDataset(torch_ds1, torch_ds2)

print(f"{patcher1=}")
print()

print(f"{patcher2=}")
print()

print(f"{len(torch_ds1)=}")
print(f"{len(torch_ds2)=}")
print()

print(f"{len(torch_ds)=}")

# %% [markdown]
# ## Reconstruct from batches

# %%
dl = torch.utils.data.DataLoader(torch_ds, batch_size=4)
rec_da1, rec_da2 = torch_ds.reconstruct_from_batches(list(dl))

# %% [markdown]
# ## Visualize reconstructions

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
rec_da1.isel(time=10).plot(ax=ax1)
ax1.set_title("Reconstruction domain 1")
patcher1.da.isel(time=10).plot(ax=ax2)
ax2.set_title("Source domain 1")

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
rec_da2.isel(time=10).plot(ax=ax1)
ax1.set_title("Reconstruction domain 2")
patcher2.da.isel(time=10).plot(ax=ax2)
ax2.set_title("Source domain 2")

plt.show()

# %%
rec_da = xr.full_like(ds.air, float("nan"))
rec_da.loc[rec_da1.coords] = rec_da1
rec_da.loc[rec_da2.coords] = rec_da2

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
rec_da.isel(time=10).plot(ax=ax1)
ax1.set_title("Reconstruction")
ds.air.isel(time=10).plot(ax=ax2)
ax2.set_title("Source data")
plt.show()
