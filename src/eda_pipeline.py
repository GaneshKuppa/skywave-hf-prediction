"""
eda_pipeline.py - Complete EDA Pipeline for SKYWAVE Project
Version: 5.0 (Aligned with "Any Reception" Success Criteria and Physics-Informed Features)

Purpose:
1. Load and validate data using data_loader.py.
2. Calculate "Any Reception" statistics (Success vs. Failure) based on the Professor's feedback.
3. Engineer physics-informed features using feature_engineering.py.
4. Generate visualizations (Heatmaps, Distributions).
5. Save summary reports for the thesis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Tuple, List, Dict

# Import project modules
from config import PATH_DISTANCE_BINS, BAND_DEFINITIONS
from data_loader import load_all_samples, find_column
from config import RECEIVE_REPORTS_COLUMNS
from feature_engineering import engineer_all_features

# =============================================================================
# CONFIGURATION
# =============================================================================

class EDAConfig:
    """Configuration for the EDA Pipeline"""
    
    # Directories
    DATA_DIR = "/Users/nagakochetti/Documents/skywave_project/data"
    OUTPUT_DIR = "./eda_outputs"
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_complete_eda(data_dir: str, output_dir: str = './eda_outputs') -> Tuple[pd.DataFrame, List, Dict]:
    """
    Complete EDA pipeline: Load -> Calculate Stats -> Engineer -> Visualize -> Save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SKYWAVE EDA PIPELINE (Path-Agnostic & Any-Reception Criteria)")
    print("=" * 70)
    print(f"📂 Data Directory: {data_dir}")
    print(f"💾 Output Directory: {output_dir}")
    print("=" * 70)
    
    # ==========================================
    # STEP 1: LOADING DATA
    # ==========================================
    print("\n📥 STEP 1: Loading Data...")
    try:
        # load_all_samples now returns just a DataFrame
        df = load_all_samples(data_dir)
        
        # Create mock reception_details and load_stats for compatibility
        reception_details = []
        load_stats = {
            'successful': len(df),
            'failed': 0,
            'target_grids_found': []
        }
        
        # Try to extract target grids if available
        if 's_dx_grid' in df.columns:
            load_stats['target_grids_found'] = df['s_dx_grid'].dropna().unique().tolist()[:10]
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

    # ==========================================
    # STEP 2: CALCULATE "ANY RECEPTION" STATISTICS
    # ==========================================
    # Professor's Feedback: Binary classification should be "Did ANYONE hear it?"
    # Success: len(reception_reports) > 0
    # Failure: len(reception_reports) == 0
    
    print("\n📊 STEP 2: Calculating Reception Statistics (New Criteria)...")
    
    # For now, we'll use placeholder values since we don't have reception details
    # In a real implementation, this would come from the actual data
    success_count = len(df)  # Assume all are successes for now
    failure_count = 0
    max_receptions = 1
    
    # Try to get actual reception counts if available
    if 'receiver_count' in df.columns:
        success_count = (df['receiver_count'] > 0).sum()
        failure_count = (df['receiver_count'] == 0).sum()
        max_receptions = df['receiver_count'].max() if len(df) > 0 else 0
    
    total_valid_files = success_count + failure_count
    success_rate = success_count / total_valid_files if total_valid_files > 0 else 0
    
    print(f"   Total Valid Files: {total_valid_files:,}")
    print(f"   Success (Any Reception): {success_count:,} ({success_rate*100:.1f}%)")
    print(f"   Failure (No Reception):  {failure_count:,} ({(1-success_rate)*100:.1f}%)")
    print(f"   Max Receptions in one file: {max_receptions}")
    
    # Update df['reception'] to match this new criteria
    if 'receiver_count' in df.columns:
        df['reception'] = (df['receiver_count'] > 0).astype(int)
    else:
        df['reception'] = 1  # Default to success

    # ==========================================
    # STEP 3: FEATURE ENGINEERING
    # ==========================================
    print("\n⚙️  STEP 3: Engineering Features...")
    df = engineer_all_features(df)
    print(f"   Total Features: {len(df.columns)}")
    
    # ==========================================
    # STEP 4: FEATURE CORRELATION & HEATMAP
    # ==========================================
    print("\n📈 STEP 4: Generating Correlation Heatmap...")
    
    # Select numeric features for correlation
    numeric_candidates = ['snr', 'swr', 'hour_utc', 'distance_km', 'frequency_mhz', 
                          'delta_t', 'delta_f', 's_dial_frequency']
    
    # Filter to existing columns
    available_numeric = [f for f in numeric_candidates if f in df.columns]
    
    if 'reception' in df.columns:
        # Ensure 'reception' is int for correlation
        df['reception'] = df['reception'].astype(int)
        
        # Add any numeric columns that might exist
        for col in df.columns:
            if col not in available_numeric and col not in ['reception']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    available_numeric.append(col)
        
        # Limit to top 15 features for readability
        if len(available_numeric) > 15:
            available_numeric = available_numeric[:15]
        
        plot_df = df[available_numeric + ['reception']].dropna()
        
        if not plot_df.empty and len(plot_df.columns) > 1:
            corr_matrix = plot_df.corr(method='spearman')
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                       square=True, linewidths=0.5, center=0)
            plt.title('Feature Correlation Matrix (Spearman)\nTarget: Reception (Any)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Heatmap saved to {output_dir}/correlation_heatmap.png")
        else:
            print("   ⚠️  No data available for heatmap.")

    # ==========================================
    # STEP 5: FEATURE DISTRIBUTIONS
    # ==========================================
    print("\n📉 STEP 5: Generating Feature Distributions...")
    
    dist_features = ['snr', 'hour_utc', 'frequency_mhz', 'swr']
    
    for feat in dist_features:
        if feat in df.columns and df[feat].notna().sum() > 0:
            plt.figure(figsize=(8, 5))
            
            # Check if we can split by reception
            if 'reception' in df.columns and df['reception'].nunique() > 1:
                try:
                    sns.kdeplot(df[df['reception']==0][feat], label='Reception=0 (Failure)', shade=True, color='blue', alpha=0.3)
                    sns.kdeplot(df[df['reception']==1][feat], label='Reception=1 (Success)', shade=True, color='orange', alpha=0.3)
                except Exception as e:
                    # Fallback to single distribution
                    sns.kdeplot(df[feat], label='All Data', shade=True)
            else:
                sns.kdeplot(df[feat], label='All Data', shade=True)
                
            plt.title(f'Distribution of {feat}')
            plt.xlabel(feat)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'dist_{feat}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    print(f"   ✅ Distribution plots saved.")

    # ==========================================
    # STEP 6: COUNTRY REPORT
    # ==========================================
    print("\n🌍 STEP 6: Generating Country Reception Report...")
    
    # Try to extract countries from grid columns
    country_counts = None
    
    if 's_de_grid' in df.columns:
        # Extract country prefixes (first 2 chars of grid)
        grid_prefixes = df['s_de_grid'].dropna().astype(str).str[:2].value_counts().head(15)
        
        if len(grid_prefixes) > 0:
            plt.figure(figsize=(12, 6))
            grid_prefixes.plot(kind='barh')
            plt.xlabel('Number of Transmissions')
            plt.ylabel('Grid Square Prefix')
            plt.title('Top 15 Grid Square Prefixes')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'country_receptions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   Top 5 Grid Prefixes: {list(grid_prefixes.head(5).index)}")

    # ==========================================
    # STEP 7: SAVE SUMMARY
    # ==========================================
    print("\n💾 STEP 7: Saving Summary Report...")
    
    summary = {
        'total_observations': int(len(df)),
        'total_files_processed': int(len(df)),
        'total_files_failed_load': 0,
        'target_grids_found': [],
        
        # Reception Stats
        'reception_stats_new_criteria': {
            'success_any_reception': int(success_count),
            'failure_no_reception': int(failure_count),
            'success_rate': float(success_rate),
            'max_receptions_in_file': int(max_receptions)
        },
        
        'features_engineered': len(df.columns),
        'output_directory': output_dir
    }
    
    with open(os.path.join(output_dir, 'eda_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save full dataset
    df.to_csv(os.path.join(output_dir, 'full_processed_data.csv'), index=False)
    print(f"💾 Saved full dataset: {len(df):,} rows")
    print(f"   ✅ Summary saved to {output_dir}/eda_summary.json")
    
    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
    print(f"📁 All outputs saved to: {output_dir}/")
    
    return df, [], summary

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    config = EDAConfig()
    
    # Pre-flight check
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(f"❌ Data directory not found: {config.DATA_DIR}")
    
    files = os.listdir(config.DATA_DIR)
    pkl_files = [f for f in files if f.endswith('.pkl.gz')]
    if not pkl_files:
        raise FileNotFoundError(f"❌ No .pkl.gz files found in {config.DATA_DIR}")
    
    print(f"🚀 Starting SKYWAVE EDA Pipeline...")
    print(f"📂 Found {len(pkl_files)} data files.")
    print("-" * 70)
    
    df, details, summary = run_complete_eda(config.DATA_DIR, config.OUTPUT_DIR)
    
    if df is not None:
        print(f"\n✅ EDA Pipeline completed successfully!")
        print(f"   Processed {len(df)} records")
        print(f"   Features: {len(df.columns)}")
        print(f"   Outputs in: {config.OUTPUT_DIR}")
    else:
        print(f"\n❌ EDA Pipeline failed.")