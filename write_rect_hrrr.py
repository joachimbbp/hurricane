import neurovolume as nv
import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
from pathlib import Path

DATA_ROOT = Path("/Volumes/Joachim Pfefferkorn Shuttle/hurricane_data")

ds = xr.open_mfdataset(sorted((DATA_ROOT / "hrrr_steps").glob("*.nc")), combine="by_coords")

levels = ds["level"].values
# cfgrib sorts isobaric levels ascending (50 hPa → 1000 hPa); flip so
# index 0 = top (low pressure), index -1 = surface (high pressure)
flip_levels = levels[0] < levels[-1]

n_times  = ds.dims["time"]
n_levels = ds.dims["level"]
n_y      = ds.dims["y"]
n_x      = ds.dims["x"]
print(f"Source shape: (time={n_times}, level={n_levels}, y={n_y}, x={n_x})")

# Output volume axes: X=x (east), Y=y (north), Z=level
NX, NY, NZ = 256, 256, 64

lvl_coords = np.linspace(0, n_levels - 1, NZ)
y_coords   = np.linspace(0, n_y - 1,     NY)
x_coords   = np.linspace(0, n_x - 1,     NX)

# meshgrid in (level, y, x) order to match source array axes
lv, ya, xa = np.meshgrid(lvl_coords, y_coords, x_coords, indexing="ij")
indices = [lv.ravel(), ya.ravel(), xa.ravel()]

water_rect = np.zeros((n_times, NX, NY, NZ), dtype=np.float32)
cloud_rect = np.zeros((n_times, NX, NY, NZ), dtype=np.float32)

for t in range(n_times):
    print(f"Resampling frame {t+1}/{n_times}...")

    water_3d = np.nan_to_num(ds["clwc"].isel(time=t).values, nan=0.0).astype(np.float32)
    for var in ["ciwc", "crwc", "cswc"]:
        water_3d += np.nan_to_num(ds[var].isel(time=t).values, nan=0.0).astype(np.float32)
    cloud_3d = np.nan_to_num(ds["cc"].isel(time=t).values, nan=0.0).astype(np.float32)

    if flip_levels:
        water_3d = np.flip(water_3d, axis=0)
        cloud_3d = np.flip(cloud_3d, axis=0)

    w = map_coordinates(water_3d, indices, order=3, mode="nearest", cval=0.0)
    water_rect[t] = w.reshape(NZ, NY, NX).transpose(2, 1, 0)  # → (NX, NY, NZ)

    c = map_coordinates(cloud_3d, indices, order=3, mode="nearest", cval=0.0)
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

save_config = nv.SaveConfig("harvey_hrrr_rect", folder=DATA_ROOT / "harvey_hrrr_rect_seq")
seq = nv.Sequence([water_channel, cloud_channel], save_config)
seq.write()
print("done!")
