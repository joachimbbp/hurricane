import xarray as xr
from pathlib import Path

steps_dir = Path("/Volumes/Joachim Pfefferkorn Shuttle/hurricane_data/hrrr_steps")
files = sorted(steps_dir.glob("*.nc"))
print(f"{len(files)} step files found")

if files:
    ds = xr.open_mfdataset(files, combine="by_coords")
    print(ds.dims)
