import xarray as xr

attr_path = "/Volumes/Untitled/cache/camels_us_attributes.nc"
# ts_path = "/Volumes/Untitled/cache/camels_us_timeseries.nc"

ds_attr = xr.open_dataset(attr_path)
print(ds_attr)
print(list(ds_attr.data_vars))

# ds_ts = xr.open_dataset(ts_path)
# print(ds_ts)
