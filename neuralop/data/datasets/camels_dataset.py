"""
CAMELS Dataset for NeuralOperator (FNO)

This module provides two approaches to integrate CAMELS hydrology data with FNO:
1. CAMELSDataset1D: Treats time as a 1D spatial dimension (per-basin prediction)
2. CAMELSDataset2D: Treats (Time, Basins) as a 2D grid

Also provides CAMELSDataLoader for native data loading using hydrodataset library,
or CAMELSDirectLoader for direct file reading without external dependencies.
"""

import os
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer


class CAMELSDirectLoader:
    """
    Direct CAMELS-US data loader without external dependencies.

    Reads CAMELS-US forcing and streamflow data directly from CSV/text files.
    Works without hydrodataset installation.

    Expected directory structure:
        data_path/
        └── CAMELS_US/ (or camels_us/)
            ├── basin_mean_daymet/
            │   └── {basin_id}_xxx_daymet.txt
            ├── usgs_streamflow/
            │   └── {basin_id}_xxx_streamflow.txt
            └── camels_attributes_v2.0/
                └── camels_*.txt

    Example
    -------
    >>> loader = CAMELSDirectLoader("/path/to/camels/data")
    >>> data = loader.load_data(
    ...     basin_ids=["01022500", "01030500"],
    ...     t_range_train=["2000-10-01", "2005-09-30"],
    ...     t_range_test=["2005-10-01", "2010-09-30"],
    ... )
    """

    # Forcing column names in daymet files
    FORCING_COLS = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]

    # Default static attributes to use
    DEFAULT_ATTR_COLS = [
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "frac_forest",
        "lai_max",
        "lai_diff",
        "soil_depth_pelletier",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "geol_porosity",
        "geol_permeability",
        "p_mean",
        "pet_mean",
        "gauge_lat",
        "gauge_lon",
        "aridity",
    ]

    def __init__(self, data_path: str):
        """
        Initialize CAMELS direct loader.

        Parameters
        ----------
        data_path : str
            Path to CAMELS dataset directory (parent of CAMELS_US folder)
        """
        self.data_path = Path(data_path)

        # Find the CAMELS_US directory
        possible_dirs = ["CAMELS_US", "camels_us", "camels", "CAMELS"]
        self.camels_dir = None
        for d in possible_dirs:
            candidate = self.data_path / d
            if candidate.exists():
                self.camels_dir = candidate
                break

        if self.camels_dir is None:
            # Maybe data_path itself is the CAMELS dir
            self.camels_dir = self.data_path

        # Find forcing and streamflow directories (may be nested)
        self.forcing_dir = self._find_nested_dir(
            ["basin_mean_forcing", "basin_mean_daymet", "daymet"]
        )
        self.streamflow_dir = self._find_nested_dir(["usgs_streamflow", "streamflow"])
        self.attr_dir = self._find_nested_dir(
            ["camels_attributes_v2.0", "attributes", "camels_attributes"]
        )

        # Also check for attribute files in root camels_dir
        if self.attr_dir is None:
            # Attribute files might be directly in camels_dir
            attr_files = list(self.camels_dir.glob("camels_*.txt"))
            if attr_files:
                self.attr_dir = self.camels_dir

        # Load basin metadata (camels_name.txt) to get HUC mapping
        self.basin_meta = self._load_basin_metadata()

        print(f"CAMELS directory: {self.camels_dir}")
        print(f"Forcing directory: {self.forcing_dir}")
        print(f"Streamflow directory: {self.streamflow_dir}")
        print(f"Attributes directory: {self.attr_dir}")
        print(f"Loaded metadata for {len(self.basin_meta)} basins")

    def _load_basin_metadata(self) -> Dict[str, Dict]:
        """Load basin metadata from camels_name.txt."""
        meta_file = self.camels_dir / "camels_name.txt"
        if not meta_file.exists():
            return {}

        meta = {}
        try:
            df = pd.read_csv(meta_file, sep=";", dtype={"gauge_id": str, "huc_02": str})
            for _, row in df.iterrows():
                meta[row["gauge_id"]] = {
                    "huc_02": row["huc_02"],
                    "gage_name": row["gauge_name"],
                }
        except Exception as e:
            print(f"Error loading camels_name.txt: {e}")

        return meta

    def _find_dir(self, names: List[str]) -> Optional[Path]:
        """Find a subdirectory by trying multiple possible names."""
        for name in names:
            candidate = self.camels_dir / name
            if candidate.exists():
                return candidate
        return None

    def _find_nested_dir(self, names: List[str], max_depth: int = 4) -> Optional[Path]:
        """Find a subdirectory by name, searching recursively up to max_depth."""
        candidates_found = []

        # First try direct children
        for name in names:
            candidate = self.camels_dir / name
            if candidate.exists():
                candidates_found.append(candidate)

        # Search recursively
        if not candidates_found:
            for depth in range(1, max_depth + 1):
                pattern = "/".join(["*"] * depth)
                for name in names:
                    found = list(self.camels_dir.glob(f"{pattern}/{name}"))
                    candidates_found.extend(found)

        if not candidates_found:
            return None

        # Filter candidates to find the best one
        # 1. Prefer ones with "v1p2" in parent path (newer version)
        # 2. Check if it actually contains files (recursively)

        best_candidate = None

        # First, check for v1p2 candidates that contain files
        for cand in candidates_found:
            if "v1p2" in str(cand):
                if any(cand.glob("**/*.txt")):
                    return (
                        cand  # Found a v1p2 candidate with files, return it immediately
                    )

        # If no v1p2 candidate with files was found, find any candidate with files
        for cand in candidates_found:
            if any(cand.glob("**/*.txt")):
                best_candidate = cand
                break  # Return the first non-v1p2 candidate that has files

        return best_candidate

    def get_all_basin_ids(self) -> List[str]:
        """Get all available basin IDs."""
        # Prefer metadata
        if self.basin_meta:
            return sorted(list(self.basin_meta.keys()))

        if self.forcing_dir is None:
            raise ValueError("Forcing directory not found")

        basin_ids = []

        # Search for forcing files (may be in subdirectories by region)
        for pattern in [
            "**/0*_*_forcing*.txt",
            "**/0*_*_daymet*.txt",
            "**/*_lump_cida_forcing*.txt",
        ]:
            for f in self.forcing_dir.glob(pattern):
                # Extract basin ID from filename
                basin_id = f.stem.split("_")[0]
                if len(basin_id) == 8 and basin_id.isdigit():
                    basin_ids.append(basin_id)

        if not basin_ids and self.streamflow_dir:
            # Try reading from streamflow directory as fallback
            for f in self.streamflow_dir.glob("**/*_streamflow*.txt"):
                basin_id = f.stem.split("_")[0]
                if len(basin_id) == 8 and basin_id.isdigit():
                    basin_ids.append(basin_id)

        return sorted(set(basin_ids))

    def load_data(
        self,
        basin_ids: Optional[List[str]] = None,
        t_range_train: List[str] = None,
        t_range_test: List[str] = None,
        forcing_cols: Optional[List[str]] = None,
        attr_cols: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Load CAMELS data for training and testing.

        Parameters
        ----------
        basin_ids : list of str, optional
            List of basin IDs. If None, uses first 10 basins.
        t_range_train : list of str
            Training time range ["start", "end"]
        t_range_test : list of str
            Test time range
        forcing_cols : list of str, optional
            Forcing column names. Default: ["prcp", "srad", "tmax", "tmin", "vp"]
        attr_cols : list of str, optional
            Attribute column names.
        normalize : bool
            Whether to apply Z-score normalization.

        Returns
        -------
        dict
            Dictionary with x_train, y_train, c_train, x_test, y_test, c_test
        """
        if basin_ids is None:
            all_basins = self.get_all_basin_ids()
            basin_ids = all_basins[:10]

        if t_range_train is None:
            t_range_train = ["2000-10-01", "2005-09-30"]
        if t_range_test is None:
            t_range_test = ["2005-10-01", "2010-09-30"]

        if forcing_cols is None:
            forcing_cols = ["prcp", "srad", "tmax", "tmin", "vp"]
        if attr_cols is None:
            attr_cols = self.DEFAULT_ATTR_COLS

        print(f"Loading CAMELS data for {len(basin_ids)} basins...")
        print(f"  Training period: {t_range_train}")
        print(f"  Test period: {t_range_test}")

        # Load forcing data
        x_train_list, x_test_list = [], []
        y_train_list, y_test_list = [], []

        for basin_id in basin_ids:
            # Load forcing
            forcing_df = self._read_forcing(basin_id)
            streamflow_df = self._read_streamflow(basin_id)

            if forcing_df is None or streamflow_df is None:
                print(f"Warning: Skipping basin {basin_id} - data not found")
                continue

            # Select forcing columns
            available_cols = [c for c in forcing_cols if c in forcing_df.columns]
            if len(available_cols) < len(forcing_cols):
                print(f"Warning: Some forcing cols not found for {basin_id}")

            # Filter by time range
            train_mask = (forcing_df.index >= t_range_train[0]) & (
                forcing_df.index <= t_range_train[1]
            )
            test_mask = (forcing_df.index >= t_range_test[0]) & (
                forcing_df.index <= t_range_test[1]
            )

            x_train_list.append(forcing_df.loc[train_mask, available_cols].values)
            x_test_list.append(forcing_df.loc[test_mask, available_cols].values)

            y_train_list.append(streamflow_df.loc[train_mask].values.reshape(-1, 1))
            y_test_list.append(streamflow_df.loc[test_mask].values.reshape(-1, 1))

        if not x_train_list:
            raise ValueError("No valid basin data found")

        # Stack to numpy arrays [N_basins, T, F]
        x_train = np.stack(x_train_list, axis=0)
        x_test = np.stack(x_test_list, axis=0)
        y_train = np.stack(y_train_list, axis=0)
        y_test = np.stack(y_test_list, axis=0)

        # Load attributes
        c_data = self._read_attributes(basin_ids, attr_cols)

        print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")

        # Handle NaN
        x_train = np.nan_to_num(x_train, nan=0.0)
        x_test = np.nan_to_num(x_test, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)
        c_data = np.nan_to_num(c_data, nan=0.0)

        result = {
            "x_train": x_train.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "x_test": x_test.astype(np.float32),
            "y_test": y_test.astype(np.float32),
            "c_train": c_data.astype(np.float32),
            "c_test": c_data.astype(np.float32),
        }

        if normalize:
            result = self._normalize_data(result)

        return result

    def _read_forcing(self, basin_id: str) -> Optional[pd.DataFrame]:
        """Read forcing data for a basin."""
        if self.forcing_dir is None:
            return None

        files = []

        # Try finding exact file using HUC if available (much faster/reliable)
        if basin_id in self.basin_meta:
            huc = self.basin_meta[basin_id]["huc_02"]
            # Check likely paths
            paths = [
                self.forcing_dir
                / "daymet"
                / huc
                / f"{basin_id}_lump_cida_forcing_leap.txt",
                self.forcing_dir / huc / f"{basin_id}_lump_cida_forcing_leap.txt",
            ]
            for p in paths:
                if p.exists():
                    files = [p]
                    break

        # Fallback to search if metadata failed or file not found
        if not files:
            patterns = [
                f"**/{basin_id}*lump_cida_forcing*.txt",
                f"**/{basin_id}*daymet*.txt",
                f"**/{basin_id}*forcing*.txt",
                f"{basin_id}*.txt",
            ]
            for pattern in patterns:
                files = list(self.forcing_dir.glob(pattern))
                if files:
                    break

        if not files:
            return None

        # Read CSV with space separator, skip header rows
        try:
            # Check how many header rows to skip (usually 3 or 4)
            # Standard CAMELS is 4 (header line + 3 lines of module info)
            # But let's peek at the file to be safe or try 3 then 4
            df = pd.read_csv(files[0], sep=r"\s+", skiprows=3)

            # Sometimes header is on line 4 (0-indexed line 3), so skiprows=4 is correct for data starting line 5
            # If current read fails check, try skiprows=4
            if not all(c in df.columns for c in ["Year", "Mnth", "Day"]):
                df = pd.read_csv(files[0], sep=r"\s+", skiprows=4)

            # Create datetime index from Year, Mnth, Day columns
            if all(c in df.columns for c in ["Year", "Mnth", "Day"]):
                df["date"] = pd.to_datetime(
                    df[["Year", "Mnth", "Day"]].rename(
                        columns={"Year": "year", "Mnth": "month", "Day": "day"}
                    )
                )
                df = df.set_index("date")

            # Rename columns to standard names
            col_mapping = {
                "Dayl(s)": "dayl",
                "dayl(s)": "dayl",
                "Prcp(mm/day)": "prcp",
                "prcp(mm/day)": "prcp",
                "Srad(W/m2)": "srad",
                "srad(W/m2)": "srad",
                "Swe(mm)": "swe",
                "swe(mm)": "swe",
                "Tmax(C)": "tmax",
                "tmax(C)": "tmax",
                "Tmin(C)": "tmin",
                "tmin(C)": "tmin",
                "Vp(Pa)": "vp",
                "vp(Pa)": "vp",
            }
            df = df.rename(columns=col_mapping)

            return df
        except Exception as e:
            print(f"Error reading forcing for {basin_id}: {e}")
            return None

    def _read_streamflow(self, basin_id: str) -> Optional[pd.Series]:
        """Read streamflow data for a basin."""
        if self.streamflow_dir is None:
            return None

        files = []

        # Try finding exact file using HUC if available
        if basin_id in self.basin_meta:
            huc = self.basin_meta[basin_id]["huc_02"]
            paths = [
                self.streamflow_dir / huc / f"{basin_id}_streamflow_qc.txt",
            ]
            for p in paths:
                if p.exists():
                    files = [p]
                    break

        # Fallback to search
        if not files:
            patterns = [
                f"**/{basin_id}*streamflow*.txt",
                f"{basin_id}*.txt",
            ]
            for pattern in patterns:
                files = list(self.streamflow_dir.glob(pattern))
                if files:
                    break

        if not files:
            return None

        try:
            df = pd.read_csv(
                files[0],
                sep=r"\s+",
                header=None,
                names=["basin", "year", "month", "day", "streamflow", "qc_flag"],
            )

            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            df = df.set_index("date")

            # Convert -999 to NaN
            streamflow = df["streamflow"].replace(-999.0, np.nan)

            return streamflow
        except Exception as e:
            print(f"Error reading streamflow for {basin_id}: {e}")
            return None

    def _read_attributes(
        self, basin_ids: List[str], attr_cols: List[str]
    ) -> np.ndarray:
        """Read static attributes for basins."""
        if self.attr_dir is None:
            # Return dummy attributes
            return np.zeros((len(basin_ids), len(attr_cols)), dtype=np.float32)

        # Try to find and read attribute files
        attr_data = {}

        # Look for attribute files in the directory
        for attr_file in self.attr_dir.glob("camels_*.txt"):
            try:
                df = pd.read_csv(attr_file, sep=";")
                if "gauge_id" in df.columns:
                    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
                    df = df.set_index("gauge_id")
                    for col in df.columns:
                        if col not in attr_data:
                            attr_data[col] = df[col]
            except Exception as e:
                continue

        if not attr_data:
            return np.zeros((len(basin_ids), len(attr_cols)), dtype=np.float32)

        # Combine into array
        result = []
        for basin_id in basin_ids:
            basin_attrs = []
            for col in attr_cols:
                if col in attr_data and basin_id in attr_data[col].index:
                    val = attr_data[col].loc[basin_id]
                    basin_attrs.append(float(val) if pd.notna(val) else 0.0)
                else:
                    basin_attrs.append(0.0)
            result.append(basin_attrs)

        return np.array(result, dtype=np.float32)

    def _normalize_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Z-score normalization."""
        # Normalize x
        x_mean = np.nanmean(data["x_train"], axis=(0, 1), keepdims=True)
        x_std = np.nanstd(data["x_train"], axis=(0, 1), keepdims=True)
        x_std = np.where(x_std == 0, 1.0, x_std)

        data["x_train"] = (data["x_train"] - x_mean) / x_std
        data["x_test"] = (data["x_test"] - x_mean) / x_std
        data["scaler_x"] = (x_mean.squeeze(), x_std.squeeze())

        # Normalize y
        y_mean = np.nanmean(data["y_train"], axis=(0, 1), keepdims=True)
        y_std = np.nanstd(data["y_train"], axis=(0, 1), keepdims=True)
        y_std = np.where(y_std == 0, 1.0, y_std)

        data["y_train"] = (data["y_train"] - y_mean) / y_std
        data["y_test"] = (data["y_test"] - y_mean) / y_std
        data["scaler_y"] = (y_mean.squeeze(), y_std.squeeze())

        # Normalize c
        c_mean = np.nanmean(data["c_train"], axis=0, keepdims=True)
        c_std = np.nanstd(data["c_train"], axis=0, keepdims=True)
        c_std = np.where(c_std == 0, 1.0, c_std)

        data["c_train"] = (data["c_train"] - c_mean) / c_std
        data["c_test"] = (data["c_test"] - c_mean) / c_std
        data["scaler_c"] = (c_mean.squeeze(), c_std.squeeze())

        return data


class CAMELSDataLoader:
    """
    Native CAMELS data loader using hydrodataset library.

    This class provides a way to load CAMELS-US data directly without
    requiring torchhydro. It uses the hydrodataset library which is
    a lighter dependency.

    Example
    -------
    >>> loader = CAMELSDataLoader("/path/to/camels/data")
    >>> data = loader.load_data(
    ...     basin_ids=["01022500", "01030500"],
    ...     t_range_train=["2000-10-01", "2005-09-30"],
    ...     t_range_test=["2005-10-01", "2010-09-30"],
    ... )
    >>> x_train, y_train = data["x_train"], data["y_train"]
    """

    # Default forcing variables from CAMELS daymet
    DEFAULT_FORCING_VARS = [
        "prcp",  # Precipitation
        "srad",  # Solar radiation
        "tmax",  # Max temperature
        "tmin",  # Min temperature
        "vp",  # Vapor pressure
    ]

    # Default static attributes
    DEFAULT_ATTR_VARS = [
        "elev_mean",
        "slope_mean",
        "area",
        "frac_forest",
        "lai_max",
        "lai_diff",
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "geol_porostiy",
        "geol_permeability",
        "p_mean",
        "pet_mean",
        "gauge_lat",
        "gauge_lon",
        "root_depth_50",
    ]

    def __init__(self, data_path: str, region: str = "US"):
        """
        Initialize CAMELS data loader.

        Parameters
        ----------
        data_path : str
            Path to CAMELS dataset directory (parent of CAMELS_US folder)
        region : str
            CAMELS region, default "US"
        """
        if not HYDRODATASET_AVAILABLE:
            raise ImportError(
                "hydrodataset is required for CAMELSDataLoader. "
                "Install with: pip install hydrodataset"
            )

        self.data_path = data_path
        self.region = region
        self._camels = None

    @property
    def camels(self):
        """Lazy initialization of Camels dataset."""
        if self._camels is None:
            self._camels = Camels(self.data_path, region=self.region)
        return self._camels

    def get_all_basin_ids(self) -> List[str]:
        """Get all available basin IDs."""
        return self.camels.read_site_info()["gauge_id"].tolist()

    def load_data(
        self,
        basin_ids: Optional[List[str]] = None,
        t_range_train: List[str] = None,
        t_range_test: List[str] = None,
        forcing_vars: Optional[List[str]] = None,
        target_var: str = "streamflow",
        attr_vars: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Load CAMELS data for training and testing.

        Parameters
        ----------
        basin_ids : list of str, optional
            List of basin IDs. If None, uses first 10 basins.
        t_range_train : list of str
            Training time range ["start", "end"], e.g., ["2000-10-01", "2005-09-30"]
        t_range_test : list of str
            Test time range
        forcing_vars : list of str, optional
            Forcing variable names. Default: ["prcp", "srad", "tmax", "tmin", "vp"]
        target_var : str
            Target variable name. Default: "streamflow"
        attr_vars : list of str, optional
            Static attribute names. Default: common 17 attributes.
        normalize : bool
            Whether to apply Z-score normalization. Default: True

        Returns
        -------
        dict
            Dictionary with keys:
            - x_train: [N_basins, T_train, F_forcing] dynamic features (normalized if requested)
            - y_train: [N_basins, T_train, 1] target streamflow
            - c_train: [N_basins, F_static] static attributes
            - x_test, y_test, c_test: corresponding test data
            - scaler_x: (mean, std) tuple for x if normalized
            - scaler_y: (mean, std) tuple for y if normalized
            - scaler_c: (mean, std) tuple for c if normalized
        """
        # Default values
        if basin_ids is None:
            all_basins = self.get_all_basin_ids()
            basin_ids = all_basins[:10]  # First 10 for testing

        if t_range_train is None:
            t_range_train = ["2000-10-01", "2005-09-30"]
        if t_range_test is None:
            t_range_test = ["2005-10-01", "2010-09-30"]

        if forcing_vars is None:
            forcing_vars = self.DEFAULT_FORCING_VARS
        if attr_vars is None:
            attr_vars = self.DEFAULT_ATTR_VARS

        print(f"Loading CAMELS data for {len(basin_ids)} basins...")
        print(f"  Training period: {t_range_train}")
        print(f"  Test period: {t_range_test}")
        print(f"  Forcing variables: {forcing_vars}")
        print(f"  Static attributes: {attr_vars}")

        # Read forcing data (x)
        x_train_xr = self.camels.read_ts_xrdataset(
            gage_id_lst=basin_ids,
            t_range=t_range_train,
            var_lst=forcing_vars,
        )
        x_test_xr = self.camels.read_ts_xrdataset(
            gage_id_lst=basin_ids,
            t_range=t_range_test,
            var_lst=forcing_vars,
        )

        # Read target data (y)
        y_train_xr = self.camels.read_ts_xrdataset(
            gage_id_lst=basin_ids,
            t_range=t_range_train,
            var_lst=[target_var],
        )
        y_test_xr = self.camels.read_ts_xrdataset(
            gage_id_lst=basin_ids,
            t_range=t_range_test,
            var_lst=[target_var],
        )

        # Read static attributes (c)
        c_xr = self.camels.read_attr_xrdataset(
            gage_id_lst=basin_ids,
            var_lst=attr_vars,
            all_number=True,
        )

        # Convert xarray to numpy arrays
        # x: [basin, time, variable] -> ensure correct dimension order
        x_train = self._xr_to_numpy(x_train_xr, dims=["basin", "time"])
        x_test = self._xr_to_numpy(x_test_xr, dims=["basin", "time"])

        y_train = self._xr_to_numpy(y_train_xr, dims=["basin", "time"])
        y_test = self._xr_to_numpy(y_test_xr, dims=["basin", "time"])

        # c: [basin, variable]
        c_data = self._xr_to_numpy_attrs(c_xr)

        print(
            f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}, c={c_data.shape}"
        )

        # Handle NaN values
        x_train = np.nan_to_num(x_train, nan=0.0)
        x_test = np.nan_to_num(x_test, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)
        c_data = np.nan_to_num(c_data, nan=0.0)

        result = {
            "x_train": x_train.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "x_test": x_test.astype(np.float32),
            "y_test": y_test.astype(np.float32),
            "c_train": c_data.astype(np.float32),
            "c_test": c_data.astype(np.float32),  # Same basins for train/test
        }

        # Normalize if requested
        if normalize:
            result = self._normalize_data(result)

        return result

    def _xr_to_numpy(self, xr_data, dims: List[str]) -> np.ndarray:
        """
        Convert xarray Dataset to numpy array.

        Parameters
        ----------
        xr_data : xarray.Dataset
            Dataset with variables as data_vars
        dims : list of str
            Expected dimension names to verify order

        Returns
        -------
        np.ndarray
            Array with shape [basin, time, n_variables]
        """
        # Stack all variables along a new dimension
        var_names = list(xr_data.data_vars)
        arrays = []
        for var in var_names:
            arr = xr_data[var].values  # [basin, time]
            arrays.append(arr)

        # Stack to [basin, time, n_vars]
        stacked = np.stack(arrays, axis=-1)
        return stacked

    def _xr_to_numpy_attrs(self, xr_data) -> np.ndarray:
        """
        Convert xarray Dataset of attributes to numpy array.

        Returns
        -------
        np.ndarray
            Array with shape [basin, n_attributes]
        """
        var_names = list(xr_data.data_vars)
        arrays = []
        for var in var_names:
            arr = xr_data[var].values  # [basin]
            arrays.append(arr)

        # Stack to [basin, n_vars]
        stacked = np.stack(arrays, axis=-1)
        return stacked

    def _normalize_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply Z-score normalization to data.

        Normalizes x, y, and c separately using training statistics.
        """
        # Normalize x (forcing) - use training data stats
        x_mean = np.nanmean(data["x_train"], axis=(0, 1), keepdims=True)
        x_std = np.nanstd(data["x_train"], axis=(0, 1), keepdims=True)
        x_std = np.where(x_std == 0, 1.0, x_std)  # Avoid division by zero

        data["x_train"] = (data["x_train"] - x_mean) / x_std
        data["x_test"] = (data["x_test"] - x_mean) / x_std
        data["scaler_x"] = (x_mean.squeeze(), x_std.squeeze())

        # Normalize y (streamflow)
        y_mean = np.nanmean(data["y_train"], axis=(0, 1), keepdims=True)
        y_std = np.nanstd(data["y_train"], axis=(0, 1), keepdims=True)
        y_std = np.where(y_std == 0, 1.0, y_std)

        data["y_train"] = (data["y_train"] - y_mean) / y_std
        data["y_test"] = (data["y_test"] - y_mean) / y_std
        data["scaler_y"] = (y_mean.squeeze(), y_std.squeeze())

        # Normalize c (static attributes)
        c_mean = np.nanmean(data["c_train"], axis=0, keepdims=True)
        c_std = np.nanstd(data["c_train"], axis=0, keepdims=True)
        c_std = np.where(c_std == 0, 1.0, c_std)

        data["c_train"] = (data["c_train"] - c_mean) / c_std
        data["c_test"] = (data["c_test"] - c_mean) / c_std
        data["scaler_c"] = (c_mean.squeeze(), c_std.squeeze())

        return data


class TensorDatasetDict(Dataset):
    """Simple Dataset that returns dict with 'x' and 'y' keys,
    compatible with neuralop Trainer."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, C_in, ...]
        y : torch.Tensor
            Output tensor of shape [N, C_out, ...]
        """
        assert x.shape[0] == y.shape[0], "x and y must have same batch size"
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class CAMELSDataset1D:
    """
    CAMELS Dataset for 1D FNO - treats time as spatial dimension.

    Each basin is treated as an independent sample. The model learns
    to map input features over time to streamflow over time.

    Input shape:  [Batch, Channels, Time]
    Output shape: [Batch, 1, Time]

    Where:
    - Batch = number of basins × number of time windows
    - Channels = dynamic features + static features (broadcasted over time)
    - Time = sequence length (e.g., 365 days)

    Parameters
    ----------
    data_config : dict
        Configuration dict compatible with torchhydro, containing:
        - source_cfgs: data source configuration
        - relevant_cols: dynamic input features
        - constant_cols: static basin attributes
        - target_cols: output variables (streamflow)
        - train_period, valid_period, test_period: date ranges
    n_train : int
        Number of training samples (basins or windows)
    n_test : int
        Number of test samples
    batch_size : int
        Training batch size
    test_batch_size : int
        Test batch size
    seq_length : int
        Time sequence length per sample
    encode_input : bool
        Whether to normalize inputs
    encode_output : bool
        Whether to normalize outputs
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        c_train: Optional[np.ndarray] = None,
        c_test: Optional[np.ndarray] = None,
        batch_size: int = 32,
        test_batch_size: int = 32,
        encode_input: bool = True,
        encode_output: bool = True,
        channel_dim: int = 1,
    ):
        """
        Initialize from pre-loaded numpy arrays.

        Parameters
        ----------
        x_train : np.ndarray
            Training dynamic features, shape [N, T, F_dyn]
        y_train : np.ndarray
            Training targets, shape [N, T, 1]
        x_test : np.ndarray
            Test dynamic features
        y_test : np.ndarray
            Test targets
        c_train : np.ndarray, optional
            Training static features, shape [N, F_static]
        c_test : np.ndarray, optional
            Test static features
        batch_size : int
            Training batch size
        test_batch_size : int
            Test batch size
        encode_input : bool
            Whether to normalize inputs
        encode_output : bool
            Whether to normalize outputs
        channel_dim : int
            Channel dimension index (default 1 for [B, C, T])
        """
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        # Convert to tensors and reshape: [N, T, F] -> [N, F, T] (channels first)
        x_train_t = self._prepare_input(x_train, c_train)
        x_test_t = self._prepare_input(x_test, c_test)

        # Output: [N, T, 1] -> [N, 1, T]
        y_train_t = torch.from_numpy(y_train).float().permute(0, 2, 1)
        y_test_t = torch.from_numpy(y_test).float().permute(0, 2, 1)

        # Create normalizers
        if encode_input:
            reduce_dims = list(range(x_train_t.ndim))
            reduce_dims.pop(channel_dim)  # Keep channel dim
            self.in_normalizer = UnitGaussianNormalizer(dim=reduce_dims)
            self.in_normalizer.fit(x_train_t)
            x_train_t = self.in_normalizer.transform(x_train_t)
            x_test_t = self.in_normalizer.transform(x_test_t)
        else:
            self.in_normalizer = None

        if encode_output:
            reduce_dims = list(range(y_train_t.ndim))
            reduce_dims.pop(channel_dim)
            self.out_normalizer = UnitGaussianNormalizer(dim=reduce_dims)
            self.out_normalizer.fit(y_train_t)
            y_train_t = self.out_normalizer.transform(y_train_t)
            y_test_t = self.out_normalizer.transform(y_test_t)
        else:
            self.out_normalizer = None

        # Create datasets
        self._train_db = TensorDatasetDict(x_train_t, y_train_t)
        self._test_db = TensorDatasetDict(x_test_t, y_test_t)

        # Create data processor
        self._data_processor = DefaultDataProcessor(
            in_normalizer=None,  # Already normalized
            out_normalizer=self.out_normalizer,  # For denormalization during eval
        )

    def _prepare_input(
        self, x_dynamic: np.ndarray, c_static: Optional[np.ndarray]
    ) -> torch.Tensor:
        """
        Prepare input by combining dynamic and static features.

        Parameters
        ----------
        x_dynamic : np.ndarray
            Dynamic features [N, T, F_dyn]
        c_static : np.ndarray, optional
            Static features [N, F_static]

        Returns
        -------
        torch.Tensor
            Combined features [N, F_total, T]
        """
        N, T, F_dyn = x_dynamic.shape

        if c_static is not None:
            # Broadcast static features over time: [N, F_static] -> [N, T, F_static]
            c_expanded = np.repeat(c_static[:, np.newaxis, :], T, axis=1)
            # Concatenate: [N, T, F_dyn + F_static]
            x_combined = np.concatenate([x_dynamic, c_expanded], axis=-1)
        else:
            x_combined = x_dynamic

        # [N, T, F] -> [N, F, T]
        return torch.from_numpy(x_combined).float().permute(0, 2, 1)

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        # Return dict keyed by resolution for compatibility with neuralop
        return {"default": self._test_db}

    @property
    def data_processor(self):
        return self._data_processor


class CAMELSDataset2D:
    """
    CAMELS Dataset for 2D FNO - treats (Time, Basins) as a 2D grid.

    This approach allows the model to learn spatial correlations between
    basins while also modeling temporal dynamics.

    Input shape:  [Batch, Channels, Time, Basins]
    Output shape: [Batch, 1, Time, Basins]

    Where:
    - Batch = number of time windows (across all basins simultaneously)
    - Channels = dynamic features + static features
    - Time = sequence length
    - Basins = number of basins (treated as spatial dimension)

    Note: This requires all basins to have the same time range.

    Parameters
    ----------
    x_train : np.ndarray
        Training dynamic features, shape [N_basins, T, F_dyn]
    y_train : np.ndarray
        Training targets, shape [N_basins, T, 1]
    x_test : np.ndarray
        Test dynamic features
    y_test : np.ndarray
        Test targets
    c_train : np.ndarray, optional
        Static features, shape [N_basins, F_static]
    batch_size : int
        Training batch size (number of time windows)
    test_batch_size : int
        Test batch size
    window_size : int
        Size of sliding window for creating samples
    stride : int
        Stride for sliding window
    encode_input : bool
        Whether to normalize inputs
    encode_output : bool
        Whether to normalize outputs
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        c_train: Optional[np.ndarray] = None,
        c_test: Optional[np.ndarray] = None,
        batch_size: int = 8,
        test_batch_size: int = 8,
        window_size: int = 365,
        stride: int = 365,
        encode_input: bool = True,
        encode_output: bool = True,
        channel_dim: int = 1,
    ):
        """
        Initialize 2D CAMELS dataset.
        """
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.window_size = window_size
        self.stride = stride

        # Create windowed samples
        # Input: [N_basins, T_total, F] -> [N_windows, F, window_size, N_basins]
        x_train_t = self._create_2d_samples(x_train, c_train, window_size, stride)
        x_test_t = self._create_2d_samples(x_test, c_test, window_size, stride)

        # Output: [N_basins, T_total, 1] -> [N_windows, 1, window_size, N_basins]
        y_train_t = self._create_2d_targets(y_train, window_size, stride)
        y_test_t = self._create_2d_targets(y_test, window_size, stride)

        # Create normalizers
        if encode_input:
            reduce_dims = [0, 2, 3]  # Normalize over batch, time, basins; keep channels
            self.in_normalizer = UnitGaussianNormalizer(dim=reduce_dims)
            self.in_normalizer.fit(x_train_t)
            x_train_t = self.in_normalizer.transform(x_train_t)
            x_test_t = self.in_normalizer.transform(x_test_t)
        else:
            self.in_normalizer = None

        if encode_output:
            reduce_dims = [0, 2, 3]
            self.out_normalizer = UnitGaussianNormalizer(dim=reduce_dims)
            self.out_normalizer.fit(y_train_t)
            y_train_t = self.out_normalizer.transform(y_train_t)
            y_test_t = self.out_normalizer.transform(y_test_t)
        else:
            self.out_normalizer = None

        # Create datasets
        self._train_db = TensorDatasetDict(x_train_t, y_train_t)
        self._test_db = TensorDatasetDict(x_test_t, y_test_t)

        # Create data processor
        self._data_processor = DefaultDataProcessor(
            in_normalizer=None, out_normalizer=self.out_normalizer
        )

    def _create_2d_samples(
        self,
        x_dynamic: np.ndarray,
        c_static: Optional[np.ndarray],
        window_size: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Create 2D samples from time series data.

        Parameters
        ----------
        x_dynamic : np.ndarray
            Shape [N_basins, T, F_dyn]
        c_static : np.ndarray, optional
            Shape [N_basins, F_static]
        window_size : int
            Window size
        stride : int
            Stride between windows

        Returns
        -------
        torch.Tensor
            Shape [N_windows, F_total, window_size, N_basins]
        """
        N_basins, T, F_dyn = x_dynamic.shape

        # Calculate number of windows
        n_windows = (T - window_size) // stride + 1

        # Prepare static features
        if c_static is not None:
            F_static = c_static.shape[1]
            # Expand: [N_basins, F_static] -> [N_basins, T, F_static]
            c_expanded = np.repeat(c_static[:, np.newaxis, :], T, axis=1)
            x_combined = np.concatenate([x_dynamic, c_expanded], axis=-1)
            F_total = F_dyn + F_static
        else:
            x_combined = x_dynamic
            F_total = F_dyn

        # Create windows: [n_windows, N_basins, window_size, F_total]
        windows = []
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            window = x_combined[:, start:end, :]  # [N_basins, window_size, F]
            windows.append(window)

        # Stack: [n_windows, N_basins, window_size, F_total]
        samples = np.stack(windows, axis=0)

        # Permute to [n_windows, F_total, window_size, N_basins]
        samples = np.transpose(samples, (0, 3, 2, 1))

        return torch.from_numpy(samples).float()

    def _create_2d_targets(
        self,
        y: np.ndarray,
        window_size: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Create 2D target windows.

        Parameters
        ----------
        y : np.ndarray
            Shape [N_basins, T, 1]

        Returns
        -------
        torch.Tensor
            Shape [N_windows, 1, window_size, N_basins]
        """
        N_basins, T, _ = y.shape
        n_windows = (T - window_size) // stride + 1

        windows = []
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            window = y[:, start:end, :]  # [N_basins, window_size, 1]
            windows.append(window)

        # Stack: [n_windows, N_basins, window_size, 1]
        targets = np.stack(windows, axis=0)

        # Permute to [n_windows, 1, window_size, N_basins]
        targets = np.transpose(targets, (0, 3, 2, 1))

        return torch.from_numpy(targets).float()

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return {"default": self._test_db}

    @property
    def data_processor(self):
        return self._data_processor


def load_camels_fno_1d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_train: Optional[np.ndarray] = None,
    c_test: Optional[np.ndarray] = None,
    batch_size: int = 32,
    test_batch_size: int = 32,
    encode_input: bool = True,
    encode_output: bool = True,
    num_workers: int = 0,
) -> tuple:
    """
    Convenience function to load CAMELS data for 1D FNO.

    Parameters
    ----------
    x_train, y_train, x_test, y_test : np.ndarray
        Training and test data arrays
    c_train, c_test : np.ndarray, optional
        Static features
    batch_size : int
        Training batch size
    test_batch_size : int
        Test batch size
    encode_input, encode_output : bool
        Whether to normalize
    num_workers : int
        DataLoader num_workers

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    test_loaders : dict
        Dict of test data loaders
    data_processor : DataProcessor
        Data processor for training
    """
    dataset = CAMELSDataset1D(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        c_train=c_train,
        c_test=c_test,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        encode_input=encode_input,
        encode_output=encode_output,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loaders = {}
    for key, test_db in dataset.test_dbs.items():
        test_loaders[key] = DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loaders, dataset.data_processor


def load_camels_fno_2d(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_train: Optional[np.ndarray] = None,
    c_test: Optional[np.ndarray] = None,
    batch_size: int = 8,
    test_batch_size: int = 8,
    window_size: int = 365,
    stride: int = 365,
    encode_input: bool = True,
    encode_output: bool = True,
    num_workers: int = 0,
) -> tuple:
    """
    Convenience function to load CAMELS data for 2D FNO.

    Parameters
    ----------
    x_train, y_train, x_test, y_test : np.ndarray
        Training and test data arrays
        Shape: [N_basins, T, F] for x, [N_basins, T, 1] for y
    c_train, c_test : np.ndarray, optional
        Static features [N_basins, F_static]
    batch_size : int
        Training batch size (number of time windows)
    test_batch_size : int
        Test batch size
    window_size : int
        Size of time window
    stride : int
        Stride between windows
    encode_input, encode_output : bool
        Whether to normalize
    num_workers : int
        DataLoader num_workers

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    test_loaders : dict
        Dict of test data loaders
    data_processor : DataProcessor
        Data processor for training
    """
    dataset = CAMELSDataset2D(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        c_train=c_train,
        c_test=c_test,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        window_size=window_size,
        stride=stride,
        encode_input=encode_input,
        encode_output=encode_output,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loaders = {}
    for key, test_db in dataset.test_dbs.items():
        test_loaders[key] = DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loaders, dataset.data_processor
