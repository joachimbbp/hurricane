import neurovolume as nv
import numpy as np
import xarray as xr
import numpy as np

ds = xr.open_dataset('harvey_landfall.nc')

# Inspect what you have
print(ds)
# Coordinates: level (pressure), latitude, longitude
# Variables: cc (cloud cover), clwc, ciwc, crwc, cswc, r

# --- Option A: Cloud fraction volume (0–1 scalar field) ---
# Shape will be (n_levels, n_lat, n_lon) e.g. (28, 81, 101)
cloud_fraction = ds['cc'].values.squeeze()   # squeeze drops the time dim
print(cloud_fraction.shape)   # (28, 81, 101) → levels × lat × lon

# --- Option B: Total condensed water (liquid + ice + rain + snow) ---
# This is the physically richest density field for VDB rendering
total_water = (
    ds['clwc'].values +   # cloud liquid water
    ds['ciwc'].values +   # cloud ice
    ds['crwc'].values +   # rain water
    ds['cswc'].values     # snow water
).squeeze()

# --- Clean up ---
water_vol = np.nan_to_num(total_water, nan=0.0).astype(np.float32)
cloud_vol = np.flip(np.nan_to_num(ds['cc'].values.squeeze(), nan=0.0), axis=0).astype(np.float32)

# Axis meaning:
# axis 0 = pressure level (28 levels, ~surface to stratosphere)
# axis 1 = latitude  (15°N → 35°N, 0.25° steps)
# axis 2 = longitude (105°W → 80°W, 0.25° steps)

np.save("water", water_vol)
np.save("cloud", cloud_vol)


w = np.load("water.npy")
c = np.load("cloud.npy")

water_grid = nv.Grid(
    "water",
    nv.prep_ndarray(w),
    # prune=np.float32(0.1),
)
cloud_grid = nv.Grid(
    "clouds",
    nv.prep_ndarray(c),
    # prune=np.float32(0.1),
)
# 0.1 is more or less good
save_config = nv.SaveConfig("hurricane_harvey_comb")

vol = nv.Volume([water_grid, cloud_grid], save_config)

vol.write()
print("VDB written")
