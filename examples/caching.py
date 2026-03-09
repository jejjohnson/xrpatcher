# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/xrpatcher/blob/main/examples/caching.py)

# %% [markdown]
# # Caching Repeated Patch Access
#
# This tutorial demonstrates how to use the built-in in-memory cache to speed up
# repeated access to the same patches.
#
# When `cache=True`, the first access to a patch slices the underlying `DataArray`
# and stores the result.  Subsequent accesses to the same index return the cached
# value without re-slicing.
#
# `preload=True` (requires `cache=True`) additionally calls `.load()` on each patch
# when it is first cached, pulling lazy array data into RAM so that later reads are
# pure in-memory lookups.

# %%
# Install dependencies (Colab)
try:
    import xrpatcher
except ImportError:
    import subprocess

    subprocess.run(["pip", "install", "xrpatcher"], check=True)

# %%
from time import perf_counter

import xarray as xr

from xrpatcher import XRDAPatcher

# %% [markdown]
# ## Setup

# %%
# Load the demo dataset and extract a 2-D field
data = xr.tutorial.load_dataset("eraint_uvz")
data = data.u[..., :240, :360]

patches = dict(longitude=30, latitude=30)
strides = dict(longitude=5, latitude=5)

# %% [markdown]
# ## Basic Cache Usage

# %%
patcher = XRDAPatcher(
    da=data,
    patches=patches,
    strides=strides,
    cache=True,
    preload=True,  # load array data into RAM on first cache hit
)

patch = patcher[0]      # sliced and cached on first access
patch = patcher[0]      # returned from cache on later access
patcher.clear_cache()   # free cached patches

# %% [markdown]
# ## Benchmark: Cached vs Uncached Repeated Access
#
# Repeat the same index access multiple times and compare elapsed time.

# %%
uncached = XRDAPatcher(da=data, patches=patches, strides=strides)
cached = XRDAPatcher(
    da=data,
    patches=patches,
    strides=strides,
    cache=True,
    preload=True,
)

for label, p in [("uncached", uncached), ("cached", cached)]:
    start = perf_counter()
    for _ in range(10):
        _ = p[0]
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
