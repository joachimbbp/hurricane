from herbie import Herbie
import cfgrib
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

dates = pd.date_range("2017-08-24 12:00", "2017-08-26 12:00", freq="1h")

WATER_VARS = {
    "clwmr": "clwc",
    "cice":  "ciwc",
    "rwmr":  "crwc",
    "snmr":  "cswc",
}

cache = Path("/Volumes/Joachim Pfefferkorn Shuttle/hurricane_data/hrrr_cache")
cache.mkdir(exist_ok=True)

steps_dir = Path("/Volumes/Joachim Pfefferkorn Shuttle/hurricane_data/hrrr_steps")
steps_dir.mkdir(exist_ok=True)

for date in dates:
    step_file = steps_dir / f"{date.strftime('%Y%m%d_%H%M')}.nc"
    if step_file.exists():
        print(f"  skipping {date} (cached)")
        continue

    print(f"Fetching {date} ...")
    H = Herbie(date, model="hrrr", product="prs", fxx=0, verbose=False, save_dir=cache)
    local_path = H.download()

    all_ds = cfgrib.open_datasets(
        str(local_path),
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        indexpath="",
    )

    ds_vars = {}
    for ds in all_ds:
        for grib_name, nc_name in WATER_VARS.items():
            if grib_name in ds.data_vars and nc_name not in ds_vars:
                da = ds[grib_name].rename({"isobaricInhPa": "level"})
                da = da.drop_vars([c for c in ["time", "step", "valid_time"] if c in da.coords])
                ds_vars[nc_name] = da

    missing = [nc for nc in WATER_VARS.values() if nc not in ds_vars]
    if missing:
        found = sorted({v for ds in all_ds for v in ds.data_vars})
        raise RuntimeError(f"Missing {missing}. Available isobaric vars: {found}")

    total_w = sum(ds_vars.values())
    cc = (1.0 - np.exp(-total_w / 5e-5)).clip(0.0, 1.0)
    cc.name = "cc"
    ds_vars["cc"] = cc

    xr.Dataset(ds_vars).expand_dims(time=[pd.Timestamp(date)]).to_netcdf(step_file, engine="netcdf4")
    print(f"  wrote {date}")

print(f"\nDone. {len(list(steps_dir.glob('*.nc')))} steps in {steps_dir}")
