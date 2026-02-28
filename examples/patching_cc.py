# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/patching_cc.py)

# %% [markdown]
# # XArrayDataset
#
# This tutorial walks through some of the nice features of the custom `XRDAPatcher` class.
# This is a custom class that slices and dices through an `xr.DataArray` where a user can
# specify explicitly the patch dimensions and the strides.
# We preallocated the *slices* and then we can arbitrarily call the slices at will.
# This is very similar to the *torch.utils.data* object except we are only working with
# `xr.DataArray`'s directly.
#
# Below, we have outlined a few use-cases that users may be interested in:
#
# * Chunking a 1-Dimensional Time Series
# * Patch-ify a 2D Grid
# * Cube-ify a 3D Volume
# * Cube-ify a 2D+T Spatio-Temporal Field
# * Reconstructing Multiple Variables
# * Choosing Specific Dimensions for Reconstructions

# %%
# Install dependencies (Colab)
try:
    import xrpatcher
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "xrpatcher", "xarray_dataclasses", "einops"], check=True)

# %%
import typing as tp
from dataclasses import dataclass

import einops
import numpy as np
import xarray_dataclasses as xrdataclass
from xrpatcher import XRDAPatcher

# %% [markdown]
# ## Case I: Chunking a 1D TS

# %%
TIME = tp.Literal["time"]


@dataclass
class TimeAxis:
    data: xrdataclass.Data[TIME, tp.Literal["datetime64[ns]"]]
    name: xrdataclass.Name[str] = "time"
    long_name: xrdataclass.Attr[str] = "Date"


@dataclass
class Variable1D:
    data: xrdataclass.Data[tuple[TIME], np.float32]
    time: xrdataclass.Coordof[TimeAxis] = 0
    name: xrdataclass.Attr[str] = "var"

# %%
t = np.arange(1, 360 + 1, 1)
rng = np.random.RandomState(seed=123)
ts = np.sin(t)

ts = Variable1D(data=ts, time=t, name="var")

da = xrdataclass.asdataarray(ts)

da

# %% [markdown]
# In this first example, we are going to do a non-overlapping style.
# We will take a 30 day window with a 30 day stride.
# This will give us exactly 12 patches (like 12 months).

# %%
patches = {"time": 30}
strides = {"time": 30}
domain_limits = None
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# In this example, we will incorporate overlapping windows.
# We will do a 30 day window but we will have a 15 day stride.
# So, we have a 15 day overlap when creating the patches.

# %%
patches = {"time": 30}
strides = {"time": 15}
domain_limits = None
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# ## Case II: Patchify a 2D Grid

# %%
X = tp.Literal["x"]
Y = tp.Literal["y"]


@dataclass
class XAxis:
    data: xrdataclass.Data[X, np.float32]
    name: xrdataclass.Name[str] = "x"


@dataclass
class YAxis:
    data: xrdataclass.Data[Y, np.float32]
    name: xrdataclass.Name[str] = "y"


@dataclass
class Variable2D:
    data: xrdataclass.Data[tuple[X, Y], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    name: xrdataclass.Attr[str] = "var"

# %%
x = np.linspace(-1, 1, 128)
y = np.linspace(-2, 2, 128)
rng = np.random.RandomState(seed=123)

data = rng.randn(x.shape[0], y.shape[0])

grid = Variable2D(data=data, x=x, y=y, name="var")

da = xrdataclass.asdataarray(grid)

da

# %% [markdown]
# We will have a `[8,8]` patch with no overlap, `[8,8]`

# %%
patches = {"x": 8, "y": 8}
strides = {"x": 8, "y": 8}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# We will have a `[8,8]` patch with some overlap, like the boundaries of 2, `[2,2]`

# %%
patches = {"x": 8, "y": 8}
strides = {"x": 2, "y": 2}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# ## Case III: Cube-ify a 3D Volume

# %%
Z = tp.Literal["z"]


@dataclass
class ZAxis:
    data: xrdataclass.Data[Z, np.float32]
    name: xrdataclass.Name[str] = "z"


@dataclass
class Variable3D:
    data: xrdataclass.Data[tuple[X, Y, Z], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    z: xrdataclass.Coordof[ZAxis] = 0
    name: xrdataclass.Attr[str] = "var"

# %%
x = np.linspace(-1, 1, 128)
y = np.linspace(-2, 2, 128)
z = np.linspace(-5, 5, 128)
rng = np.random.RandomState(seed=123)

data = rng.randn(x.shape[0], y.shape[0], z.shape[0])

grid = Variable3D(data=data, x=x, y=y, z=z, name="var")

da = xrdataclass.asdataarray(grid)

da

# %% [markdown]
# We will have a `[8,8,8]` patch with no overlap

# %%
patches = {"x": 8, "y": 8, "z": 8}
strides = {"x": 8, "y": 8, "z": 8}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# We will have a `[8,8,8]` patch with some overlap of 2

# %%
patches = {"x": 8, "y": 8, "z": 8}
strides = {"x": 2, "y": 2, "z": 2}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# ## Case IV: Cube-ify a 2D+T Spatio-Temporal Field

# %%
@dataclass
class Variable2DT:
    data: xrdataclass.Data[tuple[TIME, X, Y], np.float32]
    x: xrdataclass.Coordof[XAxis] = 0
    y: xrdataclass.Coordof[YAxis] = 0
    time: xrdataclass.Coordof[TimeAxis] = 0
    name: xrdataclass.Attr[str] = "var"

# %%
x = np.linspace(-1, 1, 200)
y = np.linspace(-2, 2, 200)
t = np.arange(1, 360 + 1, 1)
rng = np.random.RandomState(seed=123)

data = rng.randn(t.shape[0], x.shape[0], y.shape[0])

grid = Variable2DT(data=data, x=x, y=y, time=t, name="var")

da = xrdataclass.asdataarray(grid)

da

# %% [markdown]
# Now, this is a rather big field. We will use `[64,64]` patches with a temporal window of 15 days,
# so the patch will be `[15,64,64]`, with strides `[5,4,4]`.

# %%
patches = {"x": 64, "y": 64, "time": 15}
strides = {"x": 4, "y": 4, "time": 5}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# ## Case V: Reconstructing with multiple variables
#
# In this example, we look at how we can do reconstructions with multiple variables.

# %%
t = np.arange(1, 360 + 1, 1)
rng = np.random.RandomState(seed=123)
ts = np.sin(t)

ts = Variable1D(data=ts, time=t, name="var")

da = xrdataclass.asdataarray(ts)

# %%
patches = {"time": 30}
strides = {"time": 30}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %%
all_batches = list(map(lambda x: x.data, xrda_batches))
all_batches_latent = list(map(lambda x: einops.repeat(x, "... -> ... N", N=5), all_batches))

# %%
dims_labels = ["time", "z"]
weight = np.ones((patches["time"]))
rec_da = xrda_batches.reconstruct(all_batches_latent, dims_labels=dims_labels, weight=weight)
rec_da

# %% [markdown]
# ## Case VI: Choosing a Specific Dimension for Reconstruction

# %%
x = np.linspace(-1, 1, 50)
y = np.linspace(-2, 2, 50)
t = np.arange(1, 30 + 1, 1)
rng = np.random.RandomState(seed=123)

data = rng.randn(t.shape[0], x.shape[0], y.shape[0])

grid = Variable2DT(data=data, x=x, y=y, time=t, name="var")

da = xrdataclass.asdataarray(grid)

da

# %%
patches = {"x": 10, "y": 10, "time": 5}
strides = {"x": 8, "y": 8, "time": 1}
check_full_scan = True

xrda_batches = XRDAPatcher(
    da=da,
    patches=patches,
    strides=strides,
    check_full_scan=check_full_scan,
)

print(xrda_batches)
print(f"Dataset(size): {len(xrda_batches)}")

# %% [markdown]
# Here, we can reconstruct just the time series by taking the mean over x and y.

# %%
all_batches = list(map(lambda x: x.mean(dim=["x", "y"]).data, xrda_batches))

# %%
dims_labels = ["time"]
weight = np.ones((patches["time"]))
rec_da = xrda_batches.reconstruct(all_batches, dims_labels=dims_labels, weight=weight)
rec_da

# %% [markdown]
# Here, we can reconstruct just the x,y patches by taking the mean over the time dimension.

# %%
all_batches = list(map(lambda x: x.mean(dim=["time"]).data, xrda_batches))

# %%
dims_labels = ["x", "y"]
weight = np.ones((patches["x"], patches["y"]))
rec_da = xrda_batches.reconstruct(all_batches, dims_labels=dims_labels, weight=weight)
rec_da
