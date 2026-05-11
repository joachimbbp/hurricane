import neurovolume as nv
import numpy as np
import xarray as xr


from pathlib import Path


ds = xr.open_dataset('harvey_timeseries.nc')
print(ds)

# Build the full 4D arrays (time, level, lat, lon)
total_water = (
    ds['clwc'].values +
    ds['ciwc'].values +
    ds['crwc'].values +
    ds['cswc'].values
)
water_4d = np.flip(np.nan_to_num(total_water, nan=0.0), axis=1).astype(np.float32)
cloud_4d = np.flip(np.nan_to_num(ds['cc'].values, nan=0.0), axis=1).astype(np.float32)

print(f"water shape: {water_4d.shape}")
print(f"cloud shape: {cloud_4d.shape}")
assert water_4d.shape == cloud_4d.shape, "Frame count mismatch!"


water_channel = nv.Channel(
    "water",
    nv.prep_ndarray(water_4d, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)

cloud_channel = nv.Channel(
    "clouds",
    nv.prep_ndarray(cloud_4d, transpose=(0, 1, 2, 3)),
    interpolation=nv.modes.Interpolation.direct,
)
# water_channel = nv.Channel(
#     "water",
#     nv.prep_ndarray(water_4d, transpose=(3, 2, 1, 0)),
#     interpolation=nv.modes.Interpolation.direct,
# )

# cloud_channel = nv.Channel(
#     "clouds",
#     nv.prep_ndarray(cloud_4d, transpose=(3, 2, 1, 0)),
#     interpolation=nv.modes.Interpolation.direct,
# )

save_config = nv.SaveConfig("hurricane_harvey_zott", folder=Path("harvey_seq"))
seq = nv.Sequence([water_channel, cloud_channel], save_config)
seq.write()
print("done!")
