import neurovolume as nv
import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
from pyproj import CRS, Transformer
from pathlib import Path

DATA_ROOT = Path("/Volumes/Joachim Pfefferkorn Shuttle/hurricane_data")

ds = xr.open_mfdataset(sorted((DATA_ROOT / "hrrr_steps").glob("*.nc")), combine="by_coords")

levels = ds["level"].values
flip_levels = levels[0] < levels[-1]

n_times  = ds.dims["time"]
n_levels = ds.dims["level"]
n_y      = ds.dims["y"]
n_x      = ds.dims["x"]
print(f"Source shape: (time={n_times}, level={n_levels}, y={n_y}, x={n_x})")

# HRRR Lambert Conformal projection (NCEP spherical earth)
hrrr_crs = CRS.from_dict({
    "proj": "lcc",
    "lat_1": 38.5, "lat_2": 38.5, "lat_0": 38.5,
    "lon_0": -97.5,
    "a": 6371229, "b": 6371229,
})
wgs84_to_hrrr = Transformer.from_crs("EPSG:4326", hrrr_crs, always_xy=True)

# Project HRRR's 2D lat/lon grid to LCC coordinates to derive origin + spacing
lat2d = ds["latitude"].values   # (y, x)
lon2d = ds["longitude"].values  # (y, x)
lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)  # normalise to [-180, 180]

x_proj, y_proj = wgs84_to_hrrr.transform(lon2d, lat2d)
x0, y0 = x_proj[0, 0], y_proj[0, 0]
dx = (x_proj[0, -1] - x_proj[0,  0]) / (n_x - 1)
dy = (y_proj[-1, 0] - y_proj[0,  0]) / (n_y - 1)

# Build 256^3 Cartesian cube mapped onto a unit sphere
N = 256
coords_1d = np.linspace(-1.0, 1.0, N)
xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

r   = np.sqrt(xx**2 + yy**2 + zz**2)
lat = np.degrees(np.arcsin(zz / np.clip(r, 1e-6, None)))
lon = np.degrees(np.arctan2(yy, xx))  # already [-180, 180]

# Convert sphere lat/lon → HRRR (y, x) grid indices via the LCC projection
x_q, y_q = wgs84_to_hrrr.transform(lon.ravel(), lat.ravel())
y_idx = (y_q - y0) / dy
x_idx = (x_q - x0) / dx

R_inner, R_outer = 0.85, 1.00
level_idx = (r.ravel() - R_inner) / (R_outer - R_inner) * (n_levels - 1)

shell_mask  = (r.ravel() >= R_inner) & (r.ravel() <= R_outer)
domain_mask = (y_idx >= 0) & (y_idx <= n_y - 1) & (x_idx >= 0) & (x_idx <= n_x - 1)
combined_mask = (shell_mask & domain_mask).reshape(N, N, N)

indices = [level_idx, y_idx, x_idx]

# Frame 0 reserved for alignment markers; atmospheric data starts at frame 1
n_output_frames = n_times + 1
water_sphere = np.zeros((n_output_frames, N, N, N), dtype=np.float32)
cloud_sphere = np.zeros((n_output_frames, N, N, N), dtype=np.float32)

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
    water_sphere[t + 1] = w.reshape(N, N, N) * combined_mask

    c = map_coordinates(cloud_3d, indices, order=3, mode="nearest", cval=0.0)
    cloud_sphere[t + 1] = c.reshape(N, N, N) * combined_mask

print(f"Output shape: {water_sphere.shape}")

# Alignment markers (frame 0)
def latlon_to_xyz(lat_deg, lon_deg, r):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)

def xyz_to_voxel(x, y, z, N):
    ix = int(round((x + 1) / 2 * (N - 1)))
    iy = int(round((y + 1) / 2 * (N - 1)))
    iz = int(round((z + 1) / 2 * (N - 1)))
    return np.clip(ix, 0, N-1), np.clip(iy, 0, N-1), np.clip(iz, 0, N-1)

def stamp_marker(arr, ix, iy, iz, value=1.0, size=2):
    arr[ix-size:ix+size, iy-size:iy+size, iz-size:iz+size] = value

R_surface = (R_inner + R_outer) / 2

np_ix, np_iy, np_iz = xyz_to_voxel(*latlon_to_xyz(90, 0, R_surface), N)
ni_ix, ni_iy, ni_iz = xyz_to_voxel(*latlon_to_xyz(0, 0, R_surface), N)
print(f"North pole voxel:  ({np_ix}, {np_iy}, {np_iz})")
print(f"Null island voxel: ({ni_ix}, {ni_iy}, {ni_iz})")

align_sphere = np.zeros((n_output_frames, N, N, N), dtype=np.float32)
stamp_marker(align_sphere[0], np_ix, np_iy, np_iz, value=1.0)
stamp_marker(align_sphere[0], ni_ix, ni_iy, ni_iz, value=0.5)

water_channel = nv.Channel(
    "water",
    nv.prep_ndarray(water_sphere, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)
cloud_channel = nv.Channel(
    "clouds",
    nv.prep_ndarray(cloud_sphere, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)
align_channel = nv.Channel(
    "alignment",
    nv.prep_ndarray(align_sphere, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)

save_config = nv.SaveConfig("harvey_hrrr_sphere", folder=DATA_ROOT / "harvey_hrrr_sphere_seq")
seq = nv.Sequence([water_channel, cloud_channel, align_channel], save_config)
seq.write()
print("done!")
