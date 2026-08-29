"""
feature_engineering.py - Physics-Informed Feature Creation
Creates features justified by HF propagation physics and ionospheric theory.
Includes ablation study support for systematic feature validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from config import CONFIG, BAND_DEFINITIONS, PATH_DISTANCE_BINS, GRID_PRECISION_CHARS
from data_loader import maidenhead_to_latlon, haversine_km, get_grid_prefix, find_column
from config import RECEIVE_REPORTS_COLUMNS
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PHYSICS-BASED FEATURE FUNCTIONS
# =============================================================================

def get_band_category(frequency_mhz: float) -> str:
    """
    Determine ITU band category from frequency.
    Rationale: Different bands interact with ionospheric layers differently
    due to critical frequency and absorption characteristics.
    """
    if pd.isna(frequency_mhz):
        return 'Unknown'
    for band_name, (low, high) in BAND_DEFINITIONS.items():
        if low <= frequency_mhz <= high:
            return band_name
    return 'Other'

def get_path_category(distance_km: float) -> str:
    """
    Classify propagation path by distance.
    Rationale: Skip distance, ground wave vs skywave dominance, and curvature 
    losses change significantly with path length.
    """
    if pd.isna(distance_km):
        return 'Unknown'
    for category, (low, high) in PATH_DISTANCE_BINS.items():
        if low <= distance_km < high:
            return category
    return 'Unknown'

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features for diurnal and seasonal ionospheric cycles.
    
    Physics Rationale:
    - Ionospheric electron density follows solar illumination patterns
    - D-layer absorption is strong during daylight, weak at night
    - F-layer critical frequency varies with solar zenith angle
    - Cyclical encoding preserves temporal continuity (23:00 ≈ 00:00)
    """
    df = df.copy()
    
    # Find timestamp column dynamically
    ts_col = find_column(df, {'timestamp': RECEIVE_REPORTS_COLUMNS['timestamp']})
    if ts_col and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        # Basic time features
        df['hour_utc'] = df[ts_col].dt.hour
        df['day_of_year'] = df[ts_col].dt.dayofyear
        df['month'] = df[ts_col].dt.month
        
        # Cyclical encoding (preserves 23→0 continuity for diurnal patterns)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_utc'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_utc'] / 24)
        
        # Seasonal cyclical encoding
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
        
    return df

def add_band_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add band classification and frequency-derived features.
    
    Physics Rationale:
    - ITU band allocations correspond to distinct propagation regimes
    - Lower bands (40m, 30m) propagate via NVIS and short-skip at night
    - Higher bands (20m, 17m, 15m) require F-layer ionization (daylight) for long-distance skip
    - One-hot encoding allows model to learn band-specific patterns without assuming ordinality
    """
    df = df.copy()
    
    # Find frequency column dynamically
    freq_col = find_column(df, {
        'frequency_mhz': [RECEIVE_REPORTS_COLUMNS['frequency_hz'], 'frequency_mhz']
    })
    
    if not freq_col:
        # Try to calculate from frequency_hz
        freq_hz_col = find_column(df, {'frequency_hz': ['frequency_hz', 'frequency']})
        if freq_hz_col:
            df['frequency_mhz'] = df[freq_hz_col] / 1e6
            freq_col = 'frequency_mhz'
    
    if freq_col:
        # Band category (nominal)
        df['band_category'] = df[freq_col].apply(get_band_category)
        
        # One-hot encoding for band indicators
        for band_name in BAND_DEFINITIONS.keys():
            df[f'is_{band_name}'] = (df['band_category'] == band_name).astype(int)
        
        # Continuous frequency features (for regression-like patterns within bands)
        df['frequency_log'] = np.log10(df[freq_col])  # Log scale matches human perception of frequency
        
    return df

def add_path_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add path-specific physics features using Maidenhead grid locators.
    
    Physics Rationale:
    - Great-circle distance determines skip zone, number of hops, and path loss
    - Local time at receiver affects ionospheric conditions at the endpoint
    - Grid prefix grouping allows regional propagation pattern learning
    """
    df = df.copy()
    
    # Find grid columns dynamically
    sender_col = find_column(df, {'sender_grid': [RECEIVE_REPORTS_COLUMNS['sender_grid'], 's_de_grid']})
    target_col = find_column(df, {'target_grid': [RECEIVE_REPORTS_COLUMNS['target_grid'], 's_dx_grid']})
    
    # Extract sender coordinates (assumed constant per transmission file)
    if sender_col:
        sender_grids = df[sender_col].dropna().astype(str).unique()
        if len(sender_grids) > 0:
            sender_grid = sender_grids[0]
            sender_coords = maidenhead_to_latlon(sender_grid)
            if sender_coords and sender_coords[0] is not None:
                df['source_lat'] = sender_coords[0]
                df['source_lon'] = sender_coords[1]
                df['source_grid'] = sender_grid
                df['source_grid_prefix'] = get_grid_prefix(sender_grid, GRID_PRECISION_CHARS)
    
    # Extract target coordinates and calculate path metrics
    if target_col:
        def get_target_coords(row):
            grid = row.get(target_col)
            if grid and isinstance(grid, str) and len(grid) >= 4:
                return maidenhead_to_latlon(grid)
            return (None, None)
        
        coords = df.apply(get_target_coords, axis=1, result_type='expand')
        df['target_lat'] = coords[0]
        df['target_lon'] = coords[1]
        df['target_grid'] = df[target_col].astype(str).str[:GRID_PRECISION_CHARS].str.upper()
        
        # Calculate great-circle distance
        if 'source_lat' in df.columns and 'source_lon' in df.columns:
            df['distance_km'] = df.apply(
                lambda row: haversine_km(
                    row.get('source_lat'), row.get('source_lon'),
                    row.get('target_lat'), row.get('target_lon')
                ) if row.get('target_lat') is not None else None,
                axis=1
            )
            
            # Path category based on distance bins
            df['path_category'] = df['distance_km'].apply(get_path_category)
            
            # Local time at target (approximate: UTC + longitude/15)
            if 'hour_utc' in df.columns and 'target_lon' in df.columns:
                df['target_hour_local'] = (df['hour_utc'] + df['target_lon'] / 15) % 24
                df['is_daylight_target'] = df['target_hour_local'].between(6, 18).astype(int)
                
    return df

def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add signal quality and antenna efficiency features.
    
    Physics Rationale:
    - SNR categories reflect decode reliability thresholds for digital modes
    - SWR indicates antenna matching efficiency; high SWR reflects power back to transmitter
    - Binary flags simplify learning of threshold effects in propagation
    """
    df = df.copy()
    
    # Find SNR column dynamically
    snr_col = find_column(df, {'snr': [RECEIVE_REPORTS_COLUMNS['snr'], 'snr']})
    if snr_col:
        # SNR categories based on FT8/WSJT-X decode thresholds
        df['snr_category'] = pd.cut(
            df[snr_col],
            bins=[-float('inf'), -24, -20, -15, -10, float('inf')],
            labels=['Below_Decode', 'Marginal', 'Fair', 'Good', 'Excellent']
        )
        # High SNR flag (above -10 dB indicates strong signal)
        df['is_high_snr'] = (df[snr_col] > -10).astype(int)
        # SNR margin above decode threshold (-24 dB for FT8)
        df['snr_margin'] = df[snr_col] - (-24.0)
        
    # SWR features
    if 'swr' in df.columns:
        # Good SWR flag (< 2.0 indicates efficient antenna match)
        df['is_good_swr'] = (df['swr'] < 2.0).astype(int)
        # Effective power proxy (power wasted due to mismatch)
        if 'power_watts' in df.columns:
            df['effective_power'] = df['power_watts'] / df['swr'].clip(lower=1.0)
            
    return df

def add_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction/combined features as requested by professor.
    These capture physics that single features cannot represent alone.
    
    Each feature includes:
    1. Composition formula
    2. Physics rationale (with literature reference where applicable)
    3. Expected impact on model performance
    """
    df = df.copy()
    print("   Adding combined/interaction features...")
    
    # =====================================================================
    # FEATURE 1: Frequency-Daylight Match
    # Composition: frequency_mhz × is_daylight_target
    # Rationale: Higher HF bands (≥14 MHz) require F-layer ionization 
    # (daylight) for long-distance skip via refraction. Lower bands 
    # (≤10 MHz) can propagate at night via reduced D-layer absorption.
    # Reference: McNamara (1995), "The Ionosphere: Communications, 
    # Surveillance, and Direction Finding", Chapter 4.
    # Expected Impact: +3-5% recall for marginal long-distance cases.
    # =====================================================================
    if 'hour_utc' in df.columns and 'frequency_mhz' in df.columns:
        # Approximate target daylight (UTC + longitude offset)
        if 'target_lon' in df.columns:
            target_hour = (df['hour_utc'] + df['target_lon'] / 15) % 24
        else:
            # Fallback: assume Eastern Mediterranean (UTC+2)
            target_hour = (df['hour_utc'] + 2) % 24
            
        df['is_daylight_target'] = ((target_hour >= 6) & (target_hour <= 18)).astype(int)
        df['freq_daylight_match'] = df['frequency_mhz'] * df['is_daylight_target']
        print("      ✓ Added freq_daylight_match (band × sunlight interaction)")
    
    # =====================================================================
    # FEATURE 2: Effective Radiated Power Proxy
    # Composition: power_watts / swr.clip(lower=1.0)
    # Rationale: SWR > 1.0 indicates impedance mismatch; power reflected 
    # back to transmitter reduces effective radiated power. This proxy 
    # estimates actual power launched into the antenna system.
    # Reference: ARRL Antenna Book (2023), Section 2.3: "SWR and Power Transfer".
    # Expected Impact: +2-3% precision for low-power transmissions.
    # =====================================================================
    if 'power_watts' in df.columns and 'swr' in df.columns:
        df['effective_power'] = df['power_watts'] / df['swr'].clip(lower=1.0)
        print("      ✓ Added effective_power (power/SWR ratio)")
    
    # =====================================================================
    # FEATURE 3: SNR Decode Margin
    # Composition: snr - (-24.0)  [FT8 decode threshold]
    # Rationale: FT8/WSJT-X decodes reliably at ≥-24 dB. Margin indicates 
    # robustness against fading, interference, and marginal conditions.
    # Reference: WSJT-X User Manual, "Decoding Thresholds".
    # Expected Impact: +1-2% Brier score improvement (better calibration).
    # =====================================================================
    if 'snr' in df.columns:
        df['snr_margin'] = df['snr'] - (-24.0)
        print("      ✓ Added snr_margin (distance above decode threshold)")
    
    # =====================================================================
    # FEATURE 4: Band-Distance Interaction
    # Composition: is_20m × (distance_km > 1500)
    # Rationale: 20m band excels at intercontinental distances during 
    # daylight due to F2-layer critical frequency. This interaction 
    # captures band-specific skip distance preferences.
    # Expected Impact: +2% recall for intercontinental paths.
    # =====================================================================
    if 'is_20m' in df.columns and 'distance_km' in df.columns:
        df['is_20m_long_distance'] = (df['is_20m'] & (df['distance_km'] > 1500)).astype(int)
        print("      ✓ Added is_20m_long_distance (band × distance interaction)")
    
    return df

# =============================================================================
# ABLATION STUDY SUPPORT
# =============================================================================

def get_feature_groups() -> Dict[str, List[str]]:
    """
    Define feature subsets for systematic ablation study.
    Each group represents a coherent physics-based feature category.
    """
    return {
        'temporal_only': [
            'hour_utc', 'day_of_year', 'month',
            'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos'
        ],
        'signal_only': [
            'frequency_mhz', 'frequency_log', 'swr', 'snr', 'power_watts',
            'is_high_snr', 'is_good_swr', 'snr_margin'
        ],
        'band_only': [
            'band_category', 'is_40m', 'is_30m', 'is_20m', 'is_17m', 'is_15m'
        ],
        'path_only': [
            'distance_km', 'path_category', 'is_daylight_target', 
            'target_hour_local', 'source_grid_prefix'
        ],
        'combined_physics': [
            'freq_daylight_match', 'effective_power', 'snr_margin',
            'is_20m_long_distance'
        ],
        'full_physics_informed': None  # Will be set dynamically to all features
    }

def run_ablation_test(
    X: pd.DataFrame, 
    y: pd.Series, 
    feature_cols: List[str],
    train_fn: Callable,
    eval_fn: Callable
) -> Dict[str, Dict]:
    """
    Run systematic ablation study to justify feature engineering choices.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_cols: List of all available feature names
        train_fn: Function to train model (X_train, y_train) -> model
        eval_fn: Function to evaluate model (model, X_test, y_test) -> metrics dict
    
    Returns:
        Dictionary mapping feature group name to performance metrics
    """
    print("\n🔬 Running Ablation Study...")
    groups = get_feature_groups()
    results = {}
    
    # Test full feature set first (baseline)
    full_cols = [c for c in feature_cols if c in X.columns]
    if full_cols:
        model_full = train_fn(X[full_cols], y)
        metrics_full = eval_fn(model_full, X[full_cols], y)
        metrics_full['features'] = len(full_cols)
        results['full_physics_informed'] = metrics_full
        print(f"   Full set ({len(full_cols)} features): Recall={metrics_full.get('recall', 0):.3f}")
    
    # Test each subset
    for group_name, cols in groups.items():
        if group_name == 'full_physics_informed':
            continue
            
        available = [c for c in cols if c in X.columns]
        if not available:
            continue
            
        model = train_fn(X[available], y)
        metrics = eval_fn(model, X[available], y)
        metrics['features'] = len(available)
        
        # Calculate improvement vs temporal-only baseline
        if 'temporal_only' in results:
            baseline_recall = results['temporal_only'].get('recall', 0)
            metrics['improvement_vs_temporal'] = metrics.get('recall', 0) - baseline_recall
            
        results[group_name] = metrics
        print(f"   {group_name:25s} ({len(available)} features): Recall={metrics.get('recall', 0):.3f}")
    
    return results

# =============================================================================
# MAIN ENGINEERING PIPELINE
# =============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in logical order.
    Order matters: time features needed before path features, etc.
    """
    print("🔄 Engineering features...")
    
    # Step 1: Time features (needed for daylight calculations)
    df = add_time_features(df)
    
    # Step 2: Band features (frequency-based)
    df = add_band_features(df)
    
    # Step 3: Path features (requires time + grids)
    df = add_path_features(df)
    
    # Step 4: Signal features (SNR, SWR, power)
    df = add_signal_features(df)
    
    # Step 5: Combined physics features (interactions)
    df = add_combined_features(df)
    
    print(f"   Total features: {len(df.columns)}")
    return df

# =============================================================================
# EXPORT FOR REPORTING
# =============================================================================

def generate_feature_justification_table() -> pd.DataFrame:
    """
    Generate a table documenting each combined feature for the report.
    This directly addresses professor's request for explicit composition rationale.
    """
    rows = []
    
    # Feature 1
    rows.append({
        'Feature': 'freq_daylight_match',
        'Composition': 'frequency_mhz × is_daylight_target',
        'Physics_Rationale': 'Higher HF bands (≥14 MHz) require F-layer ionization (daylight) for long-distance skip via refraction. Lower bands (≤10 MHz) propagate better at night due to reduced D-layer absorption.',
        'Literature_Reference': 'McNamara (1995), Chapter 4: Ionospheric Propagation',
        'Expected_Impact': '+3-5% recall for marginal long-distance cases',
        'Validation_Method': 'Ablation study: compare model with/without feature'
    })
    
    # Feature 2
    rows.append({
        'Feature': 'effective_power',
        'Composition': 'power_watts / swr.clip(lower=1.0)',
        'Physics_Rationale': 'SWR > 1.0 indicates impedance mismatch; reflected power reduces effective radiated power. This proxy estimates actual power launched into the antenna system.',
        'Literature_Reference': 'ARRL Antenna Book (2023), Section 2.3: SWR and Power Transfer',
        'Expected_Impact': '+2-3% precision for low-power transmissions',
        'Validation_Method': 'Ablation study + correlation with reception success'
    })
    
    # Feature 3
    rows.append({
        'Feature': 'snr_margin',
        'Composition': 'snr - (-24.0)',
        'Physics_Rationale': 'FT8/WSJT-X decodes reliably at ≥-24 dB. Margin indicates robustness against fading, interference, and marginal decode conditions.',
        'Literature_Reference': 'WSJT-X User Manual: Decoding Thresholds',
        'Expected_Impact': '+1-2% Brier score improvement (better probability calibration)',
        'Validation_Method': 'Calibration curve analysis + ablation study'
    })
    
    # Feature 4
    rows.append({
        'Feature': 'is_20m_long_distance',
        'Composition': 'is_20m × (distance_km > 1500)',
        'Physics_Rationale': '20m band excels at intercontinental distances during daylight due to F2-layer critical frequency. Interaction captures band-specific skip distance preferences.',
        'Literature_Reference': 'ITU-R P.533: HF Propagation Prediction Methods',
        'Expected_Impact': '+2% recall for intercontinental paths',
        'Validation_Method': 'Stratified ablation by path category'
    })
    
    return pd.DataFrame(rows)