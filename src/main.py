"""
main.py - End-to-end pipeline orchestrator
Runs: Data Loading → Feature Engineering → Model Training → Evaluation
Uses the ACTUAL functions from your verified code files.
"""

import pandas as pd
import logging
import sys
import time
from pathlib import Path
from config import CONFIG, SUCCESS_CRITERIA
from data_loader import load_all_samples
from feature_engineering import engineer_all_features
from model_training import run_model_training
import warnings
warnings.filterwarnings("ignore")

# Force flush for immediate output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', 
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

def main():
    """Run complete SKYWAVE pipeline with professor's requested features."""
    print("="*70, flush=True)
    print("🚀 SKYWAVE PIPELINE - Multi-Criteria Validation", flush=True)
    print("="*70, flush=True)
    
    # 1. Load Data
    print("\n📥 Step 1: Loading data...", flush=True)
    start_time = time.time()
    
    try:
        # load_all_samples returns a DataFrame (not a tuple)
        df = load_all_samples()
        
        if df is None or len(df) == 0:
            logger.error("No data loaded. Check data directory.")
            return
        
        elapsed = time.time() - start_time
        print(f"   ✅ Loaded {len(df):,} transmissions in {elapsed:.1f}s", flush=True)
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 2. Feature Engineering
    print("\n⚙️  Step 2: Engineering features (including combined features)...", flush=True)
    feat_start = time.time()
    df = engineer_all_features(df)
    print(f"   ✅ Engineered {len(df.columns)} features in {time.time()-feat_start:.1f}s", flush=True)
    
    # 3. Save processed data for model training
    print("\n💾 Step 3: Saving processed data...", flush=True)
    output_path = CONFIG.EDA_DIR / "full_processed_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved to {output_path}", flush=True)
    
    # 4. Run Model Training (this handles CV, Chrono, Ablation internally)
    print("\n🤖 Step 4: Running model training pipeline...", flush=True)
    train_start = time.time()
    
    try:
        # run_model_training() handles everything: loading, training, validation, saving
        model, metrics, importance = run_model_training()
        print(f"   ✅ Training completed in {time.time()-train_start:.1f}s", flush=True)
    except Exception as e:
        print(f"   ❌ Training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 5. Summary
    print("\n" + "="*70, flush=True)
    print("✅ PIPELINE COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"\n📁 Outputs saved to: {CONFIG.MODEL_DIR.absolute()}", flush=True)
    print("   - skywave_model.pkl (trained model)", flush=True)
    print("   - validation_comparison.json (CV vs Chrono metrics)", flush=True)
    print("   - ablation_results.json (feature group performance)", flush=True)
    print("   - feature_importance.png (top features)", flush=True)
    print("   - roc_curve.png, calibration_curve.png (evaluation plots)", flush=True)
    print("\n📝 Next: Use these outputs to write your 60-page report!", flush=True)

if __name__ == "__main__":
    main()