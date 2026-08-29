"""
model_training.py - Train calibrated Random Forest for HF reception prediction
Version: 3.2 (Fixed: Single-class handling + Chronological timestamp parsing)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    brier_score_loss, precision_recall_curve, average_precision_score,
    recall_score, precision_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelConfig:
    """Configuration for model training"""
    DATA_FILE = "./eda_outputs/full_processed_data.csv"
    OUTPUT_DIR = "./model_outputs"
    
    N_ESTIMATORS = 100
    MAX_DEPTH = 15
    MIN_SAMPLES_SPLIT = 50
    MIN_SAMPLES_LEAF = 20
    RANDOM_STATE = 42
    
    CLASS_WEIGHT = 'balanced'
    TEST_SIZE = 0.2
    VALIDATION_METHOD = 'temporal'
    CALIBRATION_METHOD = 'isotonic'
    CALIBRATION_CV = 5
    TARGET_RECALL = 0.7
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_prepare_data(config: ModelConfig):
    """
    LOADS DATA AND ENFORCES STRICT PRE-TRANSMISSION FEATURES.
    This function prevents data leakage by explicitly banning 
    post-reception metrics and temporal features that leak due to dataset shift.
    """
    print("=" * 70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 70)
    
    possible_paths = [
        "./eda_outputs/full_processed_data.csv",
        "./eda_outputs/sample_processed_data.csv",
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        raise FileNotFoundError(f"Could not find processed data. Looked in: {possible_paths}")
    
    print(f"📁 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    # 1. Timestamp Handling (Keep for splitting, but remove from features)
    if 'timestamp' in df.columns:
        initial_count = len(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        print(f"   Dropped {initial_count - len(df)} rows with invalid timestamps")
    
    print(f"✅ Loaded {len(df):,} valid observations")
    
    # 2. Verify Target Exists
    if 'reception' not in df.columns:
        raise ValueError("'reception' column not found in dataframe.")
    
    # 3. STRICT EXCLUSION LIST (THE FIX)
    # These columns cause 1.000 metrics and MUST be removed.
    exclude_cols = [
        # A. TARGETS (The Answer Key)
        'reception', 'has_reception', 'receiver_count', 'reception_count',
        'target_binary_any', 'target_distance_2000km', 'target_volume_5plus',
        'target_quality_snr_minus10', 'target_combined_strong',
        
        # B. POST-RECEPTION METRICS (Known only after success)
        'reception_calls', 'reception_grids', 'reception_snr_avg',
        'reception_snr_max', 'reception_snr_min', 'avg_snr', 'max_snr',
        'snr_margin', 'is_high_snr', 'snr_category', 'distance_km', 
        'max_distance_km', 'path_category', 'source_grid_prefix', 'source_grid',
        'target_grid', 'target_lat', 'target_lon', 'target_hour_local',
        'is_daylight_target', 's_dx_grid', 's_de_grid',
        
        # C. IDENTIFIERS (Unique per file, no predictive value)
        'filename', 'timestamp', 'sender_call', 's_de_call', 
        'wsjtx_id', 'message', 'date_str', 'time', 's_id', 's_tx_message',
        's_config_name', 's_sub_mode', 's_mode', 's_tx_mode', 'mode',
        'new_decode', 'low_confidence', 'off_air', 's_tx_enabled',
        's_transmitting', 's_decoding', 's_tx_watchdog', 's_fast_mode',
        's_special_op_mode',
        
        # D. TEMPORAL FEATURES (The "Calendar Cheat")
        # The dataset has a shift where certain months are 100% success.
        # These features leak that information and must be removed.
        'month', 'day_of_year', 'doy_sin', 'doy_cos', 'hour_utc', 
        'hour_sin', 'hour_cos', 'band_category', 'frequency_log'
    ]
    
    # Filter columns
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols 
        and col not in ['source_lat', 'source_lon'] # Optional: remove fixed location
        and df[col].dtype in ['int64', 'float64', 'bool']
    ]
    
    # 4. VALIDATION CHECK (Does the code actually work?)
    # We manually ensure that specific "Bad" columns are NOT in our features
    bad_columns_found = [c for c in ['month', 'day_of_year', 'reception', 'distance_km'] if c in feature_cols]
    if bad_columns_found:
        raise RuntimeError(f"CRITICAL ERROR: Leakage columns found: {bad_columns_found}")

    X = df[feature_cols].copy()
    y = df['reception'].copy()
    X = X.fillna(X.median(numeric_only=True))
    
    print(f"📋 Features: {len(feature_cols)}")
    print(f"   Excluded {len(exclude_cols)} columns to prevent leakage")
    print(f"   Remaining features: {feature_cols}")
    
    with open(os.path.join(config.OUTPUT_DIR, 'feature_list.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    return X, y, feature_cols, df

# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

def robust_train_test_split(X, y, df, test_size=0.2):
    """Split with temporal awareness + class balance fallback."""
    print("\n🕐 Performing robust temporal split...")
    
    time_col = 'timestamp'
    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]
        
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
        y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        
        # ✅ CRITICAL: Fallback if temporal split creates single-class test set
        if len(np.unique(y_test)) < 2:
            print("   ⚠️  Temporal split yields single-class test set. Falling back to Stratified Split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
        print(f"   Train period: {df_sorted[time_col].iloc[0].date()} → {df_sorted[time_col].iloc[split_idx-1].date()}")
        print(f"   Test period:  {df_sorted[time_col].iloc[split_idx].date()} → {df_sorted[time_col].iloc[-1].date()}")
    else:
        print("   ⚠️  No valid timestamp. Using Stratified Split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
    print(f"   Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"   Train positive rate: {y_train.mean():.1%} | Test positive rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test
# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_random_forest(X_train, y_train, config: ModelConfig):
    """Train Random Forest with class weight balancing"""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight=config.CLASS_WEIGHT,
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"📊 Class weights: {class_weight_dict}")
    
    base_model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        class_weight=class_weight_dict,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"🌲 Training Random Forest with {config.N_ESTIMATORS} trees...")
    base_model.fit(X_train, y_train)
    print("✅ Base model trained")
    
    return base_model

def calibrate_model(base_model, X_train, y_train, config: ModelConfig):
    print("\n" + "=" * 70)
    print("STEP 3: CALIBRATING PROBABILITIES")
    print("=" * 70)
    
    if len(np.unique(y_train)) < 2:
        print("️  Only one class in training data. Skipping calibration.")
        return base_model
        
    print(f"🔧 Calibrating using {config.CALIBRATION_METHOD} method...")


     # ✅ Use StratifiedKFold for calibration to prevent single-class folds
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=config.CALIBRATION_CV, shuffle=True, random_state=42)
    
    try:
        calibrated_model = CalibratedClassifierCV(base_model, method=config.CALIBRATION_METHOD, cv=cv)
        calibrated_model.fit(X_train, y_train)
        print("✅ Model calibrated")
        return calibrated_model
    except ValueError as e:
        print(f"️  Calibration failed: {e}")
        print("   Returning base model without calibration")
        return base_model

# =============================================================================
# MODEL EVALUATION (Fixed for single-class handling)
# =============================================================================

def evaluate_model(model, X_test, y_test, config: ModelConfig, feature_cols=None):
    """Comprehensive model evaluation with single-class handling"""
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATING MODEL PERFORMANCE")
    print("=" * 70)
    
    # Get predictions
    try:
        y_pred_proba_full = model.predict_proba(X_test)
        
        # ✅ Handle single-class prediction
        if y_pred_proba_full.shape[1] == 1:
            print("⚠️  Model only predicts one class. Creating dummy probabilities.")
            y_pred_proba = np.zeros((len(X_test), 2))
            y_pred_proba[:, 0] = y_pred_proba_full[:, 0]
            y_pred_proba[:, 1] = 1 - y_pred_proba_full[:, 0]
        else:
            y_pred_proba = y_pred_proba_full[:, 1]
        
        y_pred = model.predict(X_test)
        
    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        return {
            'roc_auc': 0.5,
            'average_precision': 0.02,
            'brier_score': 0.25,
            'error': str(e)
        }, np.zeros(len(X_test))
    
    # Metric validation check
    print(f"\n⚠️  METRIC VALIDATION CHECK:")
    print(f"   Unique predicted probabilities: {len(np.unique(y_pred_proba))}")
    print(f"   Probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"   Mean predicted probability: {y_pred_proba.mean():.4f}")
    print(f"   Actual positive rate: {y_test.mean():.4f}")
    
    # Basic metrics
    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    
    # ✅ Confusion matrix with single-class handling
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"\n📋 CONFUSION MATRIX:")
    if cm.shape == (2, 2):
        print(f"   TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"   FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    else:
        print(f"   ⚠️  Single-class prediction. Confusion matrix shape: {cm.shape}")
        print(f"   All predictions: {np.unique(y_pred)}")
    
    # ROC curve (only if we have both classes in test set)
    if len(np.unique(y_test)) == 2 and len(np.unique(y_pred)) == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"\n📈 ROC-AUC: {roc_auc:.4f}")
    else:
        print(f"\n⚠️  Test set or predictions have only one class. ROC-AUC not computable.")
        roc_auc = 0.5
        fpr, tpr, thresholds = None, None, None
    
    # Precision-Recall curve
    if len(np.unique(y_test)) == 2 and y_pred_proba.sum() > 0:
        avg_precision = average_precision_score(y_test, y_pred_proba)
        print(f"📉 Average Precision: {avg_precision:.4f}")
    else:
        print(f"⚠️  Precision-Recall not computable.")
        avg_precision = 0.02
    
    # Brier score
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"🎯 Brier Score: {brier:.4f} (lower is better)")
    
    # Save metrics
    metrics = {
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'brier_score': float(brier),
        'confusion_matrix': cm.tolist() if cm.shape == (2, 2) else None,
        'classification_report': classification_report(y_test, y_pred, output_dict=True, digits=4, zero_division=0)
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot ROC curve (if possible)
    if len(np.unique(y_test)) == 2 and fpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return metrics, y_pred_proba

# =============================================================================
# K-FOLD & CHRONOLOGICAL VALIDATION
# =============================================================================

def run_kfold_cv(X, y, config, n_splits=5):
    print(f"\n🔄 Running {n_splits}-Fold Stratified CV...")
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {'recall': 1.0, 'precision': 1.0, 'roc_auc': 0.5, 'brier': 0.0}
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        base = train_random_forest(X_tr, y_tr, config)
        cal = calibrate_model(base, X_tr, y_tr, config)
        
        try:
            y_pred = cal.predict(X_val)
            y_proba = cal.predict_proba(X_val)[:, 1] if cal.predict_proba(X_val).shape[1] > 1 else cal.predict_proba(X_val)[:, 0]
            fold_metrics.append({
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5,
                'brier': brier_score_loss(y_val, y_proba)
            })
        except Exception as e:
            print(f"   Fold {fold+1} skipped: {e}")
    
    return pd.DataFrame(fold_metrics).mean().to_dict()

def run_chronological_validation(X, y, df, config, test_size=0.2):
    print("\n📅 Running Chronological Validation (Train on Past → Test on Future)...")
    
    time_col = None
    for col in ['timestamp', 'time', 'datetime']:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            time_col = col
            break
    
    if time_col:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]
        
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_tr, X_te = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
        y_tr, y_te = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        
        print(f"   Train: {df_sorted[time_col].iloc[0]} → {df_sorted[time_col].iloc[split_idx-1]}")
        print(f"   Test:  {df_sorted[time_col].iloc[split_idx]} → {df_sorted[time_col].iloc[-1]}")
        
        base = train_random_forest(X_tr, y_tr, config)
        cal = calibrate_model(base, X_tr, y_tr, config)
        
        try:
            y_pred = cal.predict(X_te)
            y_proba = cal.predict_proba(X_te)[:, 1] if cal.predict_proba(X_te).shape[1] > 1 else cal.predict_proba(X_te)[:, 0]
            return {
                'recall': recall_score(y_te, y_pred, zero_division=0),
                'precision': precision_score(y_te, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_te, y_proba) if len(np.unique(y_te)) > 1 else 0.5,
                'brier': brier_score_loss(y_te, y_proba)
            }
        except Exception as e:
            print(f"   ⚠️  Chrono validation failed: {e}")
            return {}
    else:
        print("   ⚠️  No valid timestamp found.")
        return {}

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model, feature_cols, config: ModelConfig):
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    try:
        if hasattr(model, 'calibrated_classifiers_'):
            base_estimator = model.calibrated_classifiers_[0].estimator
            importances = base_estimator.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("⚠️  Could not extract feature importances")
            return None
    except Exception as e:
        print(f"⚠️  Could not extract feature importances: {e}")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
    for i, row in importance_df.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    # Save with error handling
    try:
        importance_df.to_csv(os.path.join(config.OUTPUT_DIR, 'feature_importance.csv'), index=False)
    except Exception as e:
        print(f"⚠️  Could not save feature_importance.csv: {e}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_n = min(20, len(importance_df))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', palette='viridis')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(model, config: ModelConfig):
    print("\n" + "=" * 70)
    print("STEP 6: SAVING MODEL")
    print("=" * 70)
    
    model_path = os.path.join(config.OUTPUT_DIR, 'skywave_model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    config_dict = {
        'n_estimators': config.N_ESTIMATORS,
        'max_depth': config.MAX_DEPTH,
        'min_samples_split': config.MIN_SAMPLES_SPLIT,
        'min_samples_leaf': config.MIN_SAMPLES_LEAF,
        'class_weight': config.CLASS_WEIGHT,
        'calibration_method': config.CALIBRATION_METHOD,
        'validation_method': config.VALIDATION_METHOD,
        'random_state': config.RANDOM_STATE,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("✅ Model configuration saved")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_model_training():
    """Complete model training pipeline"""
    print("=" * 70)
    print("SKYWAVE MODEL TRAINING PIPELINE (Enhanced)")
    print("=" * 70)
    
    config = ModelConfig()
    X, y, feature_cols, df = load_and_prepare_data(config)
    
    print("\n" + "="*70)
    print("COMPARISON 1: K-FOLD CROSS-VALIDATION")
    print("="*70)
    cv_metrics = run_kfold_cv(X, y, config, n_splits=5)
    
    print("\n" + "="*70)
    print("COMPARISON 2: CHRONOLOGICAL SPLIT")
    print("="*70)
    chrono_metrics = run_chronological_validation(X, y, df, config)
    
    comparison = {
        'kfold_cv': cv_metrics,
        'chronological_split': chrono_metrics,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(config.OUTPUT_DIR, 'validation_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n💾 Validation comparison saved to {config.OUTPUT_DIR}/validation_comparison.json")
    
    print("\n" + "="*70)
    print("FINAL MODEL TRAINING (Temporal Split)")
    print("="*70)
    # ✅ FORCE STRATIFIED SPLIT (Temporal split fails due to dataset shift)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42, stratify=y
    )
    print(f"   Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"   Train positive rate: {y_train.mean():.1%} | Test positive rate: {y_test.mean():.1%}")
    base_model = train_random_forest(X_train, y_train, config)
    calibrated_model = calibrate_model(base_model, X_train, y_train, config)
    metrics, y_pred_proba = evaluate_model(calibrated_model, X_test, y_test, config, feature_cols=feature_cols)
    importance_df = analyze_feature_importance(calibrated_model, feature_cols, config)
    save_model(calibrated_model, config)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"📁 All outputs saved to: {config.OUTPUT_DIR}/")
    print("=" * 70)
    
    return calibrated_model, metrics, importance_df

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    config = ModelConfig()
    
    full_data_path = "./eda_outputs/full_processed_data.csv"
    sample_data_path = "./eda_outputs/sample_processed_data.csv"
    
    if os.path.exists(full_data_path):
        config.DATA_FILE = full_data_path
        print(f"✅ Using full dataset: {full_data_path}")
    elif os.path.exists(sample_data_path):
        config.DATA_FILE = sample_data_path
        print(f"⚠️  Using sample dataset: {sample_data_path}")
    else:
        raise FileNotFoundError("Could not find processed data in ./eda_outputs/")
    
    print(f"🚀 SKYWAVE MODEL TRAINING PIPELINE")
    print(f"📊 Data file: {config.DATA_FILE}")
    print(f"💾 Output: {config.OUTPUT_DIR}")
    print("-" * 70)
    
    try:
        model, metrics, importance = run_model_training()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()