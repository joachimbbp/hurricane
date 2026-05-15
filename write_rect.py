import neurovolume as nv
import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
from pathlib import Path

ds = xr.open_dataset('harvey_global.nc')

# Northeast quadrant: lat 0–90N, lon 0–90E
# ERA5 latitude is stored descending (90 → -90), so high value comes first in slice
ds_ne = ds.sel(
    latitude=slice(90, 0),
    longitude=slice(0, 90),
)

total_water = np.nan_to_num(ds_ne['clwc'].values, nan=0.0).astype(np.float32)
for var in ['ciwc', 'crwc', 'cswc']:
    total_water += np.nan_to_num(ds_ne[var].values, nan=0.0).astype(np.float32)
# flip pressure axis so index 0 = 100 hPa (top), -1 = 1000 hPa (surface)
water_4d = np.flip(total_water, axis=1)
del total_water

cloud_4d = np.flip(np.nan_to_num(ds_ne['cc'].values, nan=0.0), axis=1).astype(np.float32)

n_times, n_levels, n_lat, n_lon = water_4d.shape
print(f"Source shape: {water_4d.shape}")  # expect (36, 27, 361, 361)

# Output volume axes: X=lon, Y=lat, Z=level
NX, NY, NZ = 256, 256, 64

lvl_coords = np.linspace(0, n_levels - 1, NZ)
lat_coords = np.linspace(0, n_lat - 1, NY)
lon_coords = np.linspace(0, n_lon - 1, NX)

# meshgrid in (level, lat, lon) order to match source array axes
lv, la, lo = np.meshgrid(lvl_coords, lat_coords, lon_coords, indexing='ij')
indices = [lv.ravel(), la.ravel(), lo.ravel()]

water_rect = np.zeros((n_times, NX, NY, NZ), dtype=np.float32)
cloud_rect = np.zeros((n_times, NX, NY, NZ), dtype=np.float32)

for t in range(n_times):
    print(f"Resampling frame {t+1}/{n_times}...")
    w = map_coordinates(water_4d[t], indices, order=3, mode='nearest', cval=0.0)
    water_rect[t] = w.reshape(NZ, NY, NX).transpose(2, 1, 0)  # → (NX, NY, NZ)

    c = map_coordinates(cloud_4d[t], indices, order=3, mode='nearest', cval=0.0)
    cloud_rect[t] = c.reshape(NZ, NY, NX).transpose(2, 1, 0)

print(f"Output shape: {water_rect.shape}")

water_channel = nv.Channel(
    "water",
    nv.prep_ndarray(water_rect, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)
cloud_channel = nv.Channel(
    "clouds",
    nv.prep_ndarray(cloud_rect, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)

save_config = nv.SaveConfig("harvey_ne_rect_fade", folder=Path("harvey_ne_rect_seq"))
seq = nv.Sequence([water_channel, cloud_channel], save_config)
seq.write()
print("done!")
