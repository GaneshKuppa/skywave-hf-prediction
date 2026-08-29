"""
config.py - Centralized Configuration & Physics Definitions
Serves as the single source of truth for paths, HF band allocations, 
success criteria, column mappings, and validation parameters.
Aligns with ITU-R recommendations and ionospheric propagation literature.
"""
from pathlib import Path
import pandas as pd
import os

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
DATA_DIR = Path("data")
EDA_DIR = Path("eda_outputs")
MODEL_DIR = Path("model_outputs")
REPORT_DIR = Path("reports")

# =============================================================================
# HF BAND DEFINITIONS (ITU-R Allocations)
# Rationale: Different bands interact with ionospheric layers differently.
# These ranges are used for band classification and physics-informed feature engineering.
# =============================================================================
BAND_DEFINITIONS = {
    "40m": (7.0, 7.3),
    "30m": (10.1, 10.15),
    "20m": (14.0, 14.35),
    "17m": (18.068, 18.168),
    "15m": (21.0, 21.45),
    "12m": (24.89, 24.99),
    "10m": (28.0, 29.7)
}

# =============================================================================
# PATH DISTANCE BINS (Operational Classification)
# Rationale: Propagation characteristics change significantly with distance 
# due to D/E/F layer absorption, skip-zone effects, and curvature losses.
# =============================================================================
PATH_DISTANCE_BINS = {
    "Local": (0, 100),
    "Regional": (100, 500),
    "National": (500, 1500),
    "International": (1500, 3000),
    "Intercontinental": (3000, float('inf'))
}

# =============================================================================
# GRID PRECISION & COLUMN MAPPINGS
# Rationale: WSJT-X uses Maidenhead locators. 4-char precision provides ~100km resolution.
# Column mappings ensure robustness across varying pickle file structures and legacy exports.
# =============================================================================
GRID_PRECISION_CHARS = 4
RECEIVE_REPORTS_COLUMNS = {
    'timestamp': ['timestamp', 'time', 'datetime', 'tx_time'],
    'frequency_hz': ['s_dial_frequency', 'frequency_hz', 'frequency', 'dial_freq'],
    'snr': ['snr', 'SNR', 'signal_snr', 'rx_snr'],
    'sender_grid': ['s_de_grid', 'sender_grid', 'tx_grid'],
    'target_grid': ['s_dx_grid', 'target_grid', 'rx_grid'],
    'swr': ['swr', 'SWR', 'vswr'],
    'power': ['transmit_power_watts', 'power_watts', 'tx_power']
}

# =============================================================================
# SUCCESS CRITERIA (Professor-Requested Multi-Target Definitions)
# Rationale: Binary "any reception" is too easy (~70-90% baseline). 
# Harder criteria force the model to learn marginal propagation physics and 
# validate generalization beyond trivial majority-class predictions.
# =============================================================================
def success_binary_any(df: pd.DataFrame) -> pd.Series:
    """Criterion 1: At least one station received the transmission."""
    return (df.get('receiver_count', pd.Series([1] * len(df))) > 0).astype(int)

def success_distance_2000km(df: pd.DataFrame) -> pd.Series:
    """Criterion 2: Reception occurred at ≥2000 km distance."""
    return (df.get('max_distance_km', pd.Series([0] * len(df))) >= 2000).astype(int)

def success_volume_5plus(df: pd.DataFrame) -> pd.Series:
    """Criterion 3: Five or more distinct stations received the transmission."""
    return (df.get('receiver_count', pd.Series([0] * len(df))) >= 5).astype(int)

def success_quality_snr_minus10(df: pd.DataFrame) -> pd.Series:
    """Criterion 4: Average SNR of received signals ≥ -10 dB."""
    return (df.get('avg_snr', pd.Series([-50] * len(df))) >= -10.0).astype(int)

def success_combined_strong(df: pd.DataFrame) -> pd.Series:
    """Criterion 5: Strong signal (distance ≥1500km, ≥3 stations, SNR ≥ -12dB)."""
    dist = df.get('max_distance_km', pd.Series([0] * len(df)))
    vol = df.get('receiver_count', pd.Series([0] * len(df)))
    snr = df.get('avg_snr', pd.Series([-50] * len(df)))
    return ((dist >= 1500) & (vol >= 3) & (snr >= -12.0)).astype(int)

SUCCESS_CRITERIA = {
    'binary_any': success_binary_any,
    'distance_2000km': success_distance_2000km,
    'volume_5plus': success_volume_5plus,
    'quality_snr_minus10': success_quality_snr_minus10,
    'combined_strong': success_combined_strong
}

# =============================================================================
# MODEL & VALIDATION PARAMETERS
# Rationale: Conservative defaults ensure reproducibility, prevent overfitting,
# and align with imbalanced-class best practices (stratification, calibration, CV).
# =============================================================================
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 8
TEST_SIZE = 0.2
N_FOLDS = 5
CALIBRATION_METHOD = 'isotonic'
CALIBRATION_CV = 3

# =============================================================================
# BACKWARD COMPATIBILITY NAMESPACE
# Ensures `from config import CONFIG` works for legacy imports across modules.
# =============================================================================
class _Config:
    DATA_DIR = DATA_DIR
    EDA_DIR = EDA_DIR
    MODEL_DIR = MODEL_DIR
    REPORT_DIR = REPORT_DIR
    BAND_DEFINITIONS = BAND_DEFINITIONS
    PATH_DISTANCE_BINS = PATH_DISTANCE_BINS
    GRID_PRECISION_CHARS = GRID_PRECISION_CHARS
    RECEIVE_REPORTS_COLUMNS = RECEIVE_REPORTS_COLUMNS
    SUCCESS_CRITERIA = SUCCESS_CRITERIA
    RANDOM_STATE = RANDOM_STATE
    N_ESTIMATORS = N_ESTIMATORS
    MAX_DEPTH = MAX_DEPTH
    MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
    TEST_SIZE = TEST_SIZE
    N_FOLDS = N_FOLDS
    CALIBRATION_METHOD = CALIBRATION_METHOD
    CALIBRATION_CV = CALIBRATION_CV
    FT8_DECODE_DB = -24.0
    HIGH_SNR_DB = -10.0
    GOOD_SWR = 2.0

CONFIG = _Config()

# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================
for d in [DATA_DIR, EDA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)