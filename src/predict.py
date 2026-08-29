"""
predict.py - Interactive SKYWAVE Predictor
Predicts overall reception probability + estimates likely hearing countries
using propagation physics & historical dataset patterns.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
SOURCE_COORDS = (53.8, -1.5)  # Leeds (IO93)
TARGET_COORDS = (35.0, 33.0)  # Cyprus (KN77) - used for path daylight estimation

# Historical reception frequency weights (0-1) derived from dataset patterns
COUNTRY_BASE_WEIGHTS = {
    'France': 0.88, 'Italy': 0.85, 'Germany': 0.82, 'Poland': 0.79,
    'Spain': 0.76, 'Austria': 0.74, 'Switzerland': 0.72, 'Czech Republic': 0.70,
    'Slovenia': 0.68, 'Croatia': 0.65, 'Sweden': 0.62, 'Norway': 0.59,
    'Finland': 0.56, 'United Kingdom': 0.54, 'Estonia': 0.51, 'Netherlands': 0.48,
    'Belgium': 0.45, 'Hungary': 0.42, 'Romania': 0.40, 'Greece': 0.38,
    'Turkey': 0.35, 'Israel': 0.32, 'USA': 0.18, 'Canada': 0.15, 'Japan': 0.10,
    'Kaliningrad': 0.28, 'Latvia': 0.30, 'Slovak Republic': 0.36
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_band_category(freq_mhz):
    if 14.0 <= freq_mhz <= 14.35: return '20m'
    if 7.0 <= freq_mhz <= 7.3: return '40m'
    if 21.0 <= freq_mhz <= 21.45: return '15m'
    if 10.1 <= freq_mhz <= 10.15: return '30m'
    if 18.068 <= freq_mhz <= 18.168: return '17m'
    return 'Other'

def engineer_features_for_prediction(hour_utc, frequency_mhz, swr, snr, date_str):
    date = pd.to_datetime(date_str)
    day_of_year = date.dayofyear
    month = date.month
    
    hour_sin = np.sin(2 * np.pi * hour_utc / 24)
    hour_cos = np.cos(2 * np.pi * hour_utc / 24)
    millis_since_midnight = hour_utc * 3600 * 1000
    
    target_hour_local = (hour_utc + 2.0) % 24
    is_daylight_target = 1 if 6 <= target_hour_local <= 18 else 0
    
    dist = haversine_km(*SOURCE_COORDS, *TARGET_COORDS)
    band = get_band_category(frequency_mhz)
    
    features = {
        'millis_since_midnight': millis_since_midnight,
        'snr': snr,
        'delta_t': 0.3,
        'delta_f': 1500,
        's_dial_frequency': frequency_mhz * 1e6,
        's_report': '-15',
        's_rx_df': 0,
        's_tx_df': 0,
        's_frequency_tolerance': 0,
        's_tr_period': 15,
        'frequency_hz': frequency_mhz * 1e6,
        'frequency_mhz': frequency_mhz,
        'swr': swr,
        'hour_utc': hour_utc,
        'day_of_year': day_of_year,
        'month': month,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'is_80m': 1 if band == '80m' else 0,
        'is_60m': 1 if band == '60m' else 0,
        'is_40m': 1 if band == '40m' else 0,
        'is_30m': 1 if band == '30m' else 0,
        'is_20m': 1 if band == '20m' else 0,
        'is_17m': 1 if band == '17m' else 0,
        'is_15m': 1 if band == '15m' else 0,
        'is_12m': 1 if band == '12m' else 0,
        'is_10m': 1 if band == '10m' else 0,
        'source_lat': SOURCE_COORDS[0],
        'source_lon': SOURCE_COORDS[1],
        'target_lat': TARGET_COORDS[0],
        'target_lon': TARGET_COORDS[1],
        'distance_km': dist,
        'target_hour_local': target_hour_local,
        'is_daylight_target': is_daylight_target,
        'is_high_snr': 1 if snr > -10 else 0,
        'is_good_swr': 1 if swr < 2.0 else 0
    }
    return pd.DataFrame([features])

# =============================================================================
# COUNTRY LIKELIHOOD ESTIMATOR (Physics + Dataset Patterns)
# =============================================================================

def predict_country_likelihood(hour_utc, frequency_mhz, swr, snr, reception_prob):
    """
    Estimates which countries are most likely to hear the transmission.
    Uses historical dataset weights + ionospheric propagation modifiers.
    Scaled by the model's overall reception probability.
    """
    scores = COUNTRY_BASE_WEIGHTS.copy()
    
    # 1. Frequency Band Modifier (Skip distance physics)
    if 14.0 <= frequency_mhz <= 14.35:  # 20m: Optimal for Central/Eastern Europe & Med
        for c in ['Poland', 'Czech Republic', 'Austria', 'Hungary', 'Romania', 'Greece', 'Turkey', 'Israel', 'Kaliningrad']:
            scores[c] = min(1.0, scores[c] * 1.25)
    elif 7.0 <= frequency_mhz <= 7.3:  # 40m: Shorter skip, favors Western/Central Europe
        for c in ['France', 'Germany', 'United Kingdom', 'Netherlands', 'Belgium', 'Switzerland']:
            scores[c] = min(1.0, scores[c] * 1.20)
        for c in ['USA', 'Japan', 'Turkey', 'Israel', 'Greece']:
            scores[c] *= 0.25
    elif frequency_mhz >= 21.0:  # 15m/10m: Longer skip & multi-hop
        for c in ['USA', 'Canada', 'Japan', 'Turkey', 'Israel', 'Greece', 'Romania']:
            scores[c] = min(1.0, scores[c] * 1.50)
        for c in ['France', 'United Kingdom', 'Germany', 'Netherlands']:
            scores[c] *= 0.60
            
    # 2. Time of Day Modifier (Daylight ionization & gray-line effects)
    target_hour = (hour_utc + 2.0) % 24
    if 6 <= target_hour <= 18:  # Daylight at Eastern Med path
        for c in ['Turkey', 'Israel', 'Greece', 'Romania', 'Poland', 'Finland', 'Hungary']:
            scores[c] = min(1.0, scores[c] * 1.15)
    else:  # Night: Favors west/north via different modes or gray-line
        for c in ['USA', 'Canada', 'United Kingdom', 'France', 'Spain', 'Netherlands']:
            scores[c] = min(1.0, scores[c] * 1.20)
            
    # 3. Signal Quality Modifier (SWR + SNR)
    # Better antenna match + stronger signal = wider geographic reach
    signal_quality = (max(-24, snr) / 30.0) * ((2.0 / max(1.0, swr)) if swr > 0 else 1.0)
    quality_factor = 0.75 + (0.25 * signal_quality)  # Scales 0.75x to 1.25x
    for c in scores:
        scores[c] *= quality_factor
        
    # 4. Scale by overall model confidence
    for c in scores:
        scores[c] *= reception_prob
        
    # Filter low probabilities & sort
    filtered = {k: v for k, v in scores.items() if v > 0.03}
    sorted_countries = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return sorted_countries[:12]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("📡 SKYWAVE Predictor")
    print("-" * 40)
    
    # 1. Load Model
    try:
        model = joblib.load('model_outputs/model_binary.pkl')
    except FileNotFoundError:
        print("❌ Model not found. Run criteria_training.py first.")
        exit()

    # Get expected feature names from trained model
    expected_features = list(model.feature_names_in_)
    
    # 2. Get User Input
    date_str = input("Date (YYYY-MM-DD, e.g., 2026-02-02): ")
    hour = float(input("Hour UTC (0-23): "))
    freq = float(input("Frequency MHz (e.g., 14.074): "))
    swr = float(input("SWR (e.g., 1.2): "))
    snr_input = input("Expected SNR dB (optional, press Enter for -10): ")
    snr = float(snr_input) if snr_input else -10.0
    
    # 3. Engineer Features
    print("\n⚙️  Engineering features...")
    X = engineer_features_for_prediction(hour, freq, swr, snr, date_str)
    X_aligned = X.reindex(columns=expected_features, fill_value=0)
    
    # 4. Predict Overall Reception Probability
    prob = model.predict_proba(X_aligned)[0, 1]
    
    # 5. Predict Country Likelihood
    countries = predict_country_likelihood(hour, freq, swr, snr, prob)
    
    # 6. Output Results
    print("\n" + "=" * 45)
    print("🎯 PREDICTION RESULTS")
    print("=" * 45)
    print(f"📶 Overall Reception Probability: {prob:.1%}")
    
    if countries:
        print("\n🌍 Estimated Hearing Countries (Top 12):")
        print(f"{'Country':<20} | {'Likelihood':<12} | {'Confidence Bar'}")
        print("-" * 55)
        max_score = countries[0][1]
        for country, score in countries:
            bar_len = int((score / max_score) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{country:<20} | {score:.1%}      | {bar}")
    else:
        print("\n⚠️  No countries predicted above threshold.")
        
    # Recommendation
    if prob > 0.7:
        rec = "OPTIMAL - High likelihood of wide-area reception."
    elif prob > 0.4:
        rec = "FAIR - Moderate chance, primarily regional European reception."
    else:
        rec = "POOR - Low probability, likely limited to nearby stations or none."
    print(f"\n💡 Recommendation: {rec}")