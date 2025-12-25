import torch
import numpy as np
import xarray as xr
from pathlib import Path
from neuralop.data.datasets.camels_nc_dataset import CAMELSNCDataset, load_camels_us_nc


def test_camels_nc_dataset():
    # Paths to the actual files provided by the user
    attr_path = "/Volumes/Untitled/cache/camels_us_attributes.nc"
    ts_path = "/Volumes/Untitled/cache/camels_us_timeseries.nc"

    if not Path(attr_path).exists() or not Path(ts_path).exists():
        print(f"Skipping test: Files not found at {attr_path} or {ts_path}")
        return

    print("Testing CAMELSNCDataset...")
    dataset = CAMELSNCDataset(
        attr_path=attr_path,
        ts_path=ts_path,
        train_basins=["01013500", "01022500"],
        test_basins=["01030500"],
        t_range_train=("1980-01-01", "1980-12-31"),
        t_range_test=("1981-01-01", "1981-12-31"),
    )

    train_x, train_y = dataset.train_db[0]
    print(f"Train x shape: {train_x.shape}")  # [Channels, Time]
    print(f"Train y shape: {train_y.shape}")  # [Channels, Time]

    # Check if static features are broadcasted
    # Static features are at the end of the channel dimension
    # We can check if values are constant along the time dimension for these channels
    n_dynamic = 7
    static_sample = train_x[n_dynamic:, 0]
    static_sample_later = train_x[n_dynamic:, -1]
    assert torch.allclose(
        static_sample, static_sample_later
    ), "Static features not correctly broadcasted"
    print("Static features broadcast check passed.")

    print("Testing load_camels_us_nc...")
    train_loader, test_loaders, data_processor = load_camels_us_nc(
        attr_path=attr_path,
        ts_path=ts_path,
        batch_size=2,
        t_range_train=("1980-01-01", "1980-12-31"),
    )

    batch = next(iter(train_loader))
    x, y = batch
    print(f"Batch x shape: {x.shape}")  # [Batch, Channels, Time]
    print(f"Batch y shape: {y.shape}")

    # Process batch
    processed_dict = data_processor.preprocess({"x": x, "y": y})
    print(f"Preprocessed x shape: {processed_dict['x'].shape}")
    print("All tests passed!")


if __name__ == "__main__":
    test_camels_nc_dataset()
