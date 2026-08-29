"""
data_loader.py - Bulletproof HF Data Ingestion & Reception Aggregation
Handles pandas pickle version mismatches, extracts transmission/reception metrics,
and prepares unified DataFrame for feature engineering and modeling.
"""
import logging
import pandas as pd
import numpy as np
import pickle
import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from config import CONFIG, SUCCESS_CRITERIA
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# PANDAS COMPATIBILITY: SafeUnpickler for legacy pickle files
# =============================================================================
class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that maps legacy pandas types to Python primitives."""
    def __init__(self, file):
        super().__init__(file)
        self._cache = {}
    
    def find_class(self, module: str, name: str) -> type:
        if module.startswith("pandas."):
            if "DataFrame" in name:
                return dict
            elif "Series" in name:
                return list
            elif "Index" in name:
                return list
            elif "StringDtype" in name:
                return str
            elif "Categorical" in name:
                return list
            elif "Array" in name or "NDArrayBacked" in name:
                return list
            else:
                return type("PandasStub", (), {})
        return super().find_class(module, name)

def safe_load_pickle(filepath: Path) -> Any:
    """Load .pkl.gz file with fallback for version mismatches."""
    with gzip.open(filepath, "rb") as f:
        try:
            return SafeUnpickler(f).load()
        except Exception:
            f.seek(0)
            try:
                return pickle.load(f)
            except:
                return None

def convert_to_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    """Convert loaded pickle object to clean DataFrame."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return sanitize_dataframe(obj)
    if isinstance(obj, dict):
        for key in ["receive_reports", "transmission_reports", "data"]:
            if key in obj and isinstance(obj[key], pd.DataFrame):
                df = sanitize_dataframe(obj[key])
                if "swr" in obj and "swr" not in df.columns:
                    df["swr"] = obj["swr"]
                return df
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                return sanitize_dataframe(value)
        try:
            return sanitize_dataframe(pd.DataFrame([obj]))
        except:
            pass
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        try:
            return sanitize_dataframe(pd.DataFrame(obj))
        except:
            pass
    return None

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame: convert problematic types, coerce numerics."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = df[col].astype(str).replace("None", "").replace("nan", "")
        except:
            try:
                df[col] = df[col].astype(object)
            except:
                pass
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except:
            pass
    return df

def find_column(df: pd.DataFrame, column_map: Dict[str, List[str]]) -> Optional[str]:
    """Find column by preferred name or fallback alternatives."""
    for preferred, alternatives in column_map.items():
        if isinstance(preferred, str) and preferred in df.columns:
            return preferred
        if isinstance(alternatives, list):
            for alt in alternatives:
                if isinstance(alt, str) and alt in df.columns:
                    return alt
        elif isinstance(alternatives, str) and alternatives in df.columns:
            return alternatives
    return None

def maidenhead_to_latlon(grid: str) -> tuple:
    """Convert 4-char Maidenhead locator to lat/lon."""
    if not grid or len(grid) < 4:
        return (None, None)
    try:
        grid = grid.upper()
        lon = (ord(grid[0]) - ord("A")) * 20 - 180
        lat = (ord(grid[1]) - ord("A")) * 10 - 90
        lon += (ord(grid[2]) - ord("0")) * 2
        lat += (ord(grid[3]) - ord("0"))
        return (lat + 0.5, lon + 1)
    except:
        return (None, None)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    if None in [lat1, lon1, lat2, lon2]:
        return 0.0
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_grid_prefix(grid: str, precision: int = 4) -> str:
    """Extract grid prefix at specified precision."""
    if not grid:
        return None
    return str(grid).upper()[:precision]

# =============================================================================
# DATA LOADER CLASS
# =============================================================================
class DataLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else CONFIG.DATA_DIR
        self.files = sorted(list(self.data_dir.glob("*.pkl.gz")))
    
    def extract_all_columns(self, df: pd.DataFrame, fp_name: str) -> Dict:
        """Extract transmission + reception metrics from DataFrame."""
        record = {"filename": fp_name}
        
        # Handle empty DataFrame (zero receptions)
        if df is None or df.empty:
            record.update({
                "reception_count": 0,
                "has_reception": 0,
                "reception_calls": "",
                "reception_grids": "",
                "reception_snr_avg": None,
                "reception_snr_max": None,
                "reception_snr_min": None,
                "swr": 1.5,
                "s_dial_frequency": 14074000.0,
                "timestamp": pd.Timestamp.now()
            })
            return record
        
        # Frequency
        freq_col = find_column(df, {"s_dial_frequency": ["s_dial_frequency", "frequency_hz", "frequency"]})
        if freq_col:
            try:
                record["s_dial_frequency"] = float(pd.to_numeric(df[freq_col], errors="coerce").iloc[0])
            except:
                record["s_dial_frequency"] = 14074000.0
        else:
            record["s_dial_frequency"] = 14074000.0
        
        # Signal params
        for col in ["swr", "snr", "transmit_power_watts"]:
            if col in df.columns:
                try:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if col == "swr":
                        record[col] = float(vals.mean()) if len(vals) > 0 else 1.5
                    else:
                        record[col] = float(vals.iloc[0]) if len(vals) > 0 else None
                except:
                    record[col] = None
        
        # Timestamp
        ts_col = find_column(df, {"timestamp": ["timestamp", "time", "datetime"]})
        if ts_col:
            try:
                record["timestamp"] = pd.to_datetime(df[ts_col].iloc[0], errors="coerce")
            except:
                record["timestamp"] = None
        else:
            record["timestamp"] = None
        
        # Grids
        if "s_de_grid" in df.columns:
            try:
                record["s_de_grid"] = str(df["s_de_grid"].iloc[0]) if len(df["s_de_grid"]) > 0 else None
            except:
                record["s_de_grid"] = None
        if "s_dx_grid" in df.columns:
            try:
                record["s_dx_grid"] = str(df["s_dx_grid"].iloc[0]) if len(df["s_dx_grid"]) > 0 else None
            except:
                record["s_dx_grid"] = None
        
        # Reception reports: parse callsigns, grids, SNR
        reception_count = 0
        reception_snrs = []
        reception_grids = []
        reception_calls = []
        sender_call = None
        
        if "s_de_call" in df.columns and "s_dx_call" in df.columns:
            try:
                sender_call = df["s_de_call"].iloc[0] if len(df) > 0 else None
                if sender_call:
                    all_calls = df["s_dx_call"].dropna().unique()
                    other_calls = [c for c in all_calls if c != sender_call]
                    reception_count = len(other_calls)
                    reception_calls = other_calls
                    
                    if "snr" in df.columns:
                        for call in other_calls:
                            snr_vals = df[df["s_dx_call"] == call]["snr"].values
                            if len(snr_vals) > 0:
                                try:
                                    best_snr = float(np.max(snr_vals))
                                    reception_snrs.append(best_snr)
                                except:
                                    pass
                    
                    if "s_dx_grid" in df.columns:
                        for call in other_calls:
                            grid_vals = df[df["s_dx_call"] == call]["s_dx_grid"].values
                            if len(grid_vals) > 0:
                                try:
                                    reception_grids.append(str(grid_vals[0]))
                                except:
                                    pass
            except Exception as e:
                logger.debug(f"Could not parse callsign data: {e}")
        
        # Store reception metrics
        record.update({
            "reception_count": reception_count,
            "reception_calls": ",".join(reception_calls) if reception_calls else "",
            "reception_grids": ",".join(reception_grids) if reception_grids else "",
            "reception_snr_avg": float(np.mean(reception_snrs)) if reception_snrs else None,
            "reception_snr_max": float(np.max(reception_snrs)) if reception_snrs else None,
            "reception_snr_min": float(np.min(reception_snrs)) if reception_snrs else None,
            "has_reception": 1 if reception_count > 0 else 0,
            "sender_call": sender_call or (str(df["s_de_call"].iloc[0]) if "s_de_call" in df.columns and len(df) > 0 else None)
        })
        
        return record
    
    def load_and_prepare(self) -> pd.DataFrame:
        """Load all files, extract metrics, and return unified DataFrame."""
        if not self.files:
            raise FileNotFoundError(f"No .pkl.gz files in {self.data_dir}")
        
        records = []
        valid, invalid = 0, 0
        debug_count = 0
        
        logger.info(f" Loading {len(self.files)} files...")
        
        for idx, fp in enumerate(self.files):
            if idx % 1000 == 0:
                logger.info(f"  Processed {idx}/{len(self.files)}...")
            
            try:
                obj = safe_load_pickle(fp)
                if obj is None:
                    invalid += 1
                    continue
                
                df = convert_to_dataframe(obj)
                record = self.extract_all_columns(df, fp.name)
                
                # Fallback: parse frequency from filename if missing
                if df is None or df.empty:
                    try:
                        freq_match = re.search(r"(\d+)Hz", fp.name)
                        if freq_match:
                            record["s_dial_frequency"] = float(freq_match.group(1))
                    except:
                        pass
                    if record.get("timestamp") is None:
                        record["timestamp"] = pd.Timestamp.now()
                
                # Skip invalid timestamps
                if record.get("timestamp") is None:
                    invalid += 1
                    continue
                
                # Validate frequency (3-30 MHz HF band)
                freq = record.get("s_dial_frequency", 0)
                if not (3_000_000 <= freq <= 30_000_000):
                    invalid += 1
                    continue
                
                # Validate SWR (0.5-20.0 operational range)
                swr = record.get("swr", 1.5)
                if swr is not None and (swr < 0.5 or swr > 20.0):
                    invalid += 1
                    continue
                
                records.append(record)
                valid += 1
                
            except Exception as e:
                invalid += 1
                if debug_count < 3:
                    logger.warning(f"  File {idx}: {fp.name} - {str(e)[:80]}")
                    debug_count += 1
        
        if not records:
            raise ValueError(f"No valid records after processing {len(self.files)} files.")
        
        # Create DataFrame
        combined = pd.DataFrame(records)
        
        # Process timestamps
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        # Add time features
        combined["hour_utc"] = combined["timestamp"].dt.hour
        combined["day_of_year"] = combined["timestamp"].dt.dayofyear
        combined["month"] = combined["timestamp"].dt.month
        
        # Add frequency in MHz
        combined["frequency_mhz"] = combined["s_dial_frequency"] / 1e6
        
        # Set reception metrics
        if "has_reception" in combined.columns:
            combined["reception"] = combined["has_reception"]
            combined["receiver_count"] = combined["reception_count"]
            combined["avg_snr"] = combined["reception_snr_avg"]
            combined["max_snr"] = combined["reception_snr_max"]
        else:
            combined["reception"] = 1
            combined["receiver_count"] = 1
            combined["avg_snr"] = combined.get("snr", 0)
            combined["max_snr"] = combined.get("snr", 0)
        
        # Placeholder for distance (real implementation would use grid-to-distance)
        combined["max_distance_km"] = 1000
        
        # Apply multi-criteria success targets from config.py
        for name, func in SUCCESS_CRITERIA.items():
            try:
                temp_df = combined.copy()
                for col, default in [("max_distance_km", 1000), ("avg_snr", -5), ("receiver_count", 1)]:
                    if col not in temp_df.columns:
                        temp_df[col] = default
                combined[f"target_{name}"] = func(temp_df)
            except Exception as e:
                logger.warning(f"  Could not create target_{name}: {e}")
                combined[f"target_{name}"] = 0
        
        # Log statistics
        logger.info(f"✅ Loaded {valid} valid, skipped {invalid}. Total: {len(combined):,}")
        if "reception" in combined.columns:
            success_rate = combined["reception"].mean()
            logger.info(f" Success rate (any reception): {success_rate:.1%}")
            logger.info(f" Total receptions: {combined['receiver_count'].sum():,}")
            logger.info(f" Average receptions per transmission: {combined['receiver_count'].mean():.2f}")
            logger.info(f" Max receptions in one transmission: {combined['receiver_count'].max()}")
            zero_rec = (combined["reception"] == 0).sum()
            logger.info(f" Files with zero receptions: {zero_rec:,}")
        
        # Save sample for EDA pipeline
        sample_path = CONFIG.EDA_DIR / "sample_processed_data.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(sample_path, index=False)
        logger.info(f"💾 Saved sample data to {sample_path}")
        
        return combined

def load_all_samples(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Wrapper for eda_pipeline.py compatibility."""
    loader = DataLoader(data_dir)
    return loader.load_and_prepare()

if __name__ == "__main__":
    # Test single file
    test_dir = Path("data")
    files = list(test_dir.glob("*.pkl.gz"))
    if files:
        print(f"Testing file: {files[0].name}")
        obj = safe_load_pickle(files[0])
        df = convert_to_dataframe(obj)
        if df is not None:
            loader = DataLoader()
            record = loader.extract_all_columns(df, files[0].name)
            print(f"  Has reception: {record.get('has_reception')}")
            print(f"  Reception count: {record.get('reception_count')}")
    
    # Full load test
    print("\nRunning full data load...")
    try:
        loader = DataLoader()
        df = loader.load_and_prepare()
        print(f"✅ Success! Loaded {len(df)} records")
        if "reception" in df.columns:
            print(f"   Reception distribution:\n{df['reception'].value_counts()}")
            print(f"   Reception count stats:\n{df['reception_count'].describe()}")
    except Exception as e:
        print(f"❌ Failed: {e}")