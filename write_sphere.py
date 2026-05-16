import neurovolume as nv
import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
from pathlib import Path

ds = xr.open_dataset('harvey_global.nc')

# --- Build 4D source arrays (time, level, lat, lon) ---
total_water = np.nan_to_num(ds['clwc'].values, nan=0.0).astype(np.float32)
for var in ['ciwc', 'crwc', 'cswc']:
    total_water += np.nan_to_num(ds[var].values, nan=0.0).astype(np.float32)
water_4d = np.flip(total_water, axis=1)
del total_water

cloud_4d = np.flip(np.nan_to_num(ds['cc'].values, nan=0.0), axis=1).astype(np.float32)

n_times, n_levels, n_lat, n_lon = water_4d.shape
print(f"Source shape: {water_4d.shape}")

# --- Define output Cartesian cube ---
N = 256
coords_1d = np.linspace(-1.0, 1.0, N)
xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')

# --- Convert Cartesian → spherical ---
r   = np.sqrt(xx**2 + yy**2 + zz**2)
lat = np.degrees(np.arcsin(zz / np.clip(r, 1e-6, None)))
lon = np.degrees(np.arctan2(yy, xx))

# --- Map spherical coords to ERA5 array indices ---
lat_min = float(ds.latitude.min())
lat_max = float(ds.latitude.max())
lon_min = float(ds.longitude.min())
lon_max = float(ds.longitude.max())

lat_idx = (lat - lat_min) / 0.25
lon_idx = (lon - lon_min) / 0.25

R_inner = 0.85
R_outer = 1.00
level_idx = (r - R_inner) / (R_outer - R_inner) * (n_levels - 1)

shell_mask = (r >= R_inner) & (r <= R_outer)

indices = [
    level_idx.ravel(),
    lat_idx.ravel(),
    lon_idx.ravel(),
]

# --- Resample each timestep ---
# Frame 0 is reserved for alignment markers — atmospheric data starts at frame 1
n_output_frames = n_times + 1
water_sphere = np.zeros((n_output_frames, N, N, N), dtype=np.float32)
cloud_sphere = np.zeros((n_output_frames, N, N, N), dtype=np.float32)

for t in range(n_times):
    print(f"Resampling frame {t+1}/{n_times}...")
    # claude: allegedly order 3 is cubic spline, which should be smoother
    w = map_coordinates(water_4d[t], indices, order=3, mode='wrap', cval=0.0)
    water_sphere[t + 1] = w.reshape(N, N, N) * shell_mask

    c = map_coordinates(cloud_4d[t], indices, order=3, mode='wrap', cval=0.0)
    cloud_sphere[t + 1] = c.reshape(N, N, N) * shell_mask

print(f"Output shape: {water_sphere.shape}")

# --- Alignment markers (frame 0 only) ---
def latlon_to_xyz(lat_deg, lon_deg, r):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

def xyz_to_voxel(x, y, z, N):
    ix = int(round((x + 1) / 2 * (N - 1)))
    iy = int(round((y + 1) / 2 * (N - 1)))
    iz = int(round((z + 1) / 2 * (N - 1)))
    return np.clip(ix, 0, N-1), np.clip(iy, 0, N-1), np.clip(iz, 0, N-1)

def stamp_marker(arr, ix, iy, iz, value=1.0, size=2):
    arr[ix-size:ix+size, iy-size:iy+size, iz-size:iz+size] = value

R_surface = (R_inner + R_outer) / 2

np_x, np_y, np_z = latlon_to_xyz(90, 0, R_surface)
np_ix, np_iy, np_iz = xyz_to_voxel(np_x, np_y, np_z, N)

ni_x, ni_y, ni_z = latlon_to_xyz(0, 0, R_surface)
ni_ix, ni_iy, ni_iz = xyz_to_voxel(ni_x, ni_y, ni_z, N)

print(f"North pole voxel:  ({np_ix}, {np_iy}, {np_iz})")
print(f"Null island voxel: ({ni_ix}, {ni_iy}, {ni_iz})")

align_sphere = np.zeros((n_output_frames, N, N, N), dtype=np.float32)
stamp_marker(align_sphere[0], np_ix, np_iy, np_iz, value=1.0)   # north pole — bright
stamp_marker(align_sphere[0], ni_ix, ni_iy, ni_iz, value=0.5)   # null island — dimmer

save_config = nv.SaveConfig("harvey_global_sphere_cubic_up", folder=Path("harvey_global_sphere_seq_cubic_up"))

# --- Write as neurovolume sequence ---

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

seq = nv.Sequence([water_channel, cloud_channel, align_channel], save_config)
seq.write()



print("done!")
