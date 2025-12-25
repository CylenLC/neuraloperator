import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer


class CAMELSNCDataset:
    """
    CAMELSNCDataset for reading CAMELS-US data from NetCDF files.

    Parameters
    ----------
    attr_path : str or Path
        Path to camels_us_attributes.nc
    ts_path : str or Path
        Path to camels_us_timeseries.nc
    train_basins : List[str], optional
        List of basin IDs for training. If None, all-but-test basins are used.
    test_basins : List[str], optional
        List of basin IDs for testing.
    t_range_train : Tuple[str, str], optional
        Time range for training, e.g., ("1980-01-01", "2000-12-31")
    t_range_test : Tuple[str, str], optional
        Time range for testing.
    static_features : List[str], optional
        List of static attributes to use.
    dynamic_features : List[str], optional
        List of dynamic forcing variables to use.
    target_feature : str, optional
        Target variable name, default "q_cms_obs"
    batch_size : int, optional
        Batch size for training.
    test_batch_size : int, optional
        Batch size for testing.
    """

    def __init__(
        self,
        attr_path: Union[str, Path],
        ts_path: Union[str, Path],
        train_basins: Optional[List[str]] = None,
        test_basins: Optional[List[str]] = None,
        t_range_train: Tuple[str, str] = ("1980-01-01", "2000-12-31"),
        t_range_test: Tuple[str, str] = ("2001-01-01", "2014-12-31"),
        static_features: Optional[List[str]] = None,
        dynamic_features: Optional[List[str]] = None,
        target_feature: str = "q_cms_obs",
        batch_size: int = 32,
        test_batch_size: int = 32,
    ):
        self.attr_path = Path(attr_path)
        self.ts_path = Path(ts_path)

        # Load NetCDF
        ds_attr = xr.open_dataset(self.attr_path)
        ds_ts = xr.open_dataset(self.ts_path)

        # Default features if not provided
        if static_features is None:
            # Pick some common ones or all numeric ones
            static_features = [
                v
                for v in ds_attr.data_vars
                if ds_attr[v].dtype in [np.float32, np.float64]
            ]
        if dynamic_features is None:
            dynamic_features = [
                "dayl",
                "pcp_mm",
                "solrad_wm2",
                "swe_mm",
                "airtemp_c_max",
                "airtemp_c_min",
                "vp_hpa",
            ]

        all_basins = ds_attr.basin.values.tolist()
        if test_basins is None:
            # Default to a small split if not specified
            test_basins = all_basins[-50:]

        if train_basins is None:
            train_basins = [b for b in all_basins if b not in test_basins]

        # Helper to extract data
        def get_data(basins, t_range):
            # Subset dimensions
            subset_ts = ds_ts.sel(basin=basins, time=slice(t_range[0], t_range[1]))
            subset_attr = ds_attr.sel(basin=basins)

            # Dynamic features: [Basin, Time, Feature]
            x_dynamic = []
            for feat in dynamic_features:
                x_dynamic.append(subset_ts[feat].values)
            x_dynamic = np.stack(x_dynamic, axis=-1)  # [B, T, FD]

            # Static features: [Basin, Feature]
            x_static = []
            for feat in static_features:
                val = subset_attr[feat].values
                # Handle cases where attr might have different shape (though usually (basin,))
                x_static.append(val)
            x_static = np.stack(x_static, axis=-1)  # [B, FS]

            # Broadcast static to dynamic: [B, T, FS]
            T = x_dynamic.shape[1]
            x_static_broadcast = np.repeat(x_static[:, np.newaxis, :], T, axis=1)

            # Concatenate: [B, T, FD + FS]
            x = np.concatenate([x_dynamic, x_static_broadcast], axis=-1)

            # Target: [B, T, 1]
            y = subset_ts[target_feature].values[..., np.newaxis]

            return torch.tensor(x, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )

        x_train, y_train = get_data(train_basins, t_range_train)
        x_test, y_test = get_data(test_basins, t_range_test)

        # In neuraloperator, data is often [Batch, Channels, Dim1, ...]
        # For sequence data, it might be [B, C, T]
        # We'll follow the channel-first convention: [B, C, T]
        x_train = x_train.permute(0, 2, 1)
        y_train = y_train.permute(0, 2, 1)
        x_test = x_test.permute(0, 2, 1)
        y_test = y_test.permute(0, 2, 1)

        # Handle NaNs
        x_train = torch.nan_to_num(x_train, nan=0.0)
        y_train = torch.nan_to_num(y_train, nan=0.0)
        x_test = torch.nan_to_num(x_test, nan=0.0)
        y_test = torch.nan_to_num(y_test, nan=0.0)

        # Encoders (Normalizers)
        # We use UnitGaussianNormalizer across basins and time
        # reduce_dims: dims to reduce over to compute mean/std.
        # For [B, C, T], we reduce over B and T (dims 0 and 2) to get channel-wise normalization.
        self.in_normalizer = UnitGaussianNormalizer(dim=[0, 2])
        self.in_normalizer.fit(x_train)

        self.out_normalizer = UnitGaussianNormalizer(dim=[0, 2])
        self.out_normalizer.fit(y_train)

        self._data_processor = DefaultDataProcessor(
            in_normalizer=self.in_normalizer, out_normalizer=self.out_normalizer
        )

        self._train_db = TensorDataset(x_train, y_train)
        self._test_dbs = {"default": TensorDataset(x_test, y_test)}

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs

    @property
    def data_processor(self):
        return self._data_processor


def load_camels_us_nc(
    attr_path: str,
    ts_path: str,
    train_basins: Optional[List[str]] = None,
    test_basins: Optional[List[str]] = None,
    batch_size: int = 32,
    test_batch_size: int = 32,
    t_range_train: Tuple[str, str] = ("1980-01-01", "2000-12-31"),
    t_range_test: Tuple[str, str] = ("2001-01-01", "2014-12-31"),
    static_features: Optional[List[str]] = None,
    dynamic_features: Optional[List[str]] = None,
):
    """
    Utility function to load CAMELS NC dataset and return DataLoaders.
    """
    dataset = CAMELSNCDataset(
        attr_path=attr_path,
        ts_path=ts_path,
        train_basins=train_basins,
        test_basins=test_basins,
        t_range_train=t_range_train,
        t_range_test=t_range_test,
        static_features=static_features,
        dynamic_features=dynamic_features,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
    )

    train_loader = DataLoader(dataset.train_db, batch_size=batch_size, shuffle=True)
    test_loaders = {
        res: DataLoader(db, batch_size=test_batch_size, shuffle=False)
        for res, db in dataset.test_dbs.items()
    }

    return train_loader, test_loaders, dataset.data_processor
