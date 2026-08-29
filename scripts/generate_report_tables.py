#!/usr/bin/env python3
"""
SKYWAVE Report: Final Table Generator (Optimized Layout)
Fixes text overlap by increasing canvas width and cell height.
"""

import matplotlib.pyplot as plt
import os

os.makedirs('tables', exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def render_academic_table(data, columns, title, filename, is_wide=False):
    # --- KEY FIX: Wider canvas for text-heavy tables ---
    fig_width = 16 if is_wide else 10
    fig_height = max(5, len(data) * 0.6 + 2.0)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=data,
                     colLabels=columns,
                     loc='center',
                     cellLoc='left')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # --- KEY FIX: Scale up cell height significantly ---
    table.scale(1.0, 2.8) 
    
    # Style cells
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#E5E7EB')
        cell.set_linewidth(0.5)
        cell.set_text_props(ha='left', va='center', wrap=True)
        
        if i == 0:  # Header
            cell.set_facecolor('#1E3A8A')
            cell.set_text_props(weight='bold', color='white', fontsize=11, ha='center')
        elif i % 2 == 1:
            cell.set_facecolor('#F8FAFC')
        else:
            cell.set_facecolor('#FFFFFF')
            
        # Center align specific columns (Metrics, Scores)
        if is_wide:
            if j in [1, 2]: # Variables, Math columns
                 # Keep left aligned for text, maybe center for math if simple
                 pass
        else:
            if j in [1, 2]:
                cell.set_text_props(ha='center')
            
    plt.title(title, fontsize=15, fontweight='bold', pad=35, family='sans-serif')
    plt.tight_layout(pad=2.0)
    plt.savefig(f'tables/{filename}.png', bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"✅ Saved: tables/{filename}.png")

# ─────────────────────────────────────────────────────────────
# 2. DATA DEFINITIONS (Cleaned)
# ─────────────────────────────────────────────────────────────

# Table 4.1
t41_data = [
    ['Cyclical Temporal', 'hour_sin, hour_cos,\ndoy_sin, doy_cos', 'sin(2πh/24),\ncos(2πh/24)', 'Preserves diurnal continuity;\neliminates discontinuity between 23:00–00:00'],
    ['Frequency & Band', 'frequency_mhz, is_20m,\nis_40m, is_30m, is_17m', 'One-hot encoding of ITU\nband allocations', 'Captures frequency-specific\nionospheric interaction regimes'],
    ['Antenna & Power', 'swr, is_good_swr', 'effective_power = power_watts /\nmax(swr, 1.0)', 'Estimates radiated efficiency;\npenalises impedance mismatch'],
    ['Propagation Interaction', 'freq_daylight_match,\nis_20m_long_distance', 'freq_daylight_match =\nfrequency_mhz × I(daylight)', 'Encodes F-layer ionisation\ndependency for ≥14 MHz skip\npropagation'],
    ['Signal Margin', 'snr_margin', 'snr_margin = snr − (−24.0)', 'Measures robustness above\nFT8 decode threshold']
]
# Pass is_wide=True to fix the text squashing
render_academic_table(t41_data, 
                      ['Feature Category', 'Variables', 'Mathematical Formulation', 'Operational Rationale'],
                      'Table 4.1: Physics-Informed Feature Composition and Operational Rationale',
                      'table_4_1_feature_composition',
                      is_wide=True)

# Table 4.2
t42_data = [
    ['Recall', '0.8621', '0.012', '[0.851, 0.873]'],
    ['Precision', '0.8435', '0.015', '[0.830, 0.857]'],
    ['ROC-AUC', '0.8790', '0.009', '[0.871, 0.887]'],
    ['F1-Score', '0.8525', '0.011', '[0.842, 0.863]'],
    ['Brier Score', '0.0892', '0.004', '[0.085, 0.093]']
]
render_academic_table(t42_data,
                      ['Metric', 'Mean', 'Standard Deviation', '95% Confidence Interval'],
                      'Table 4.2: Stratified 5-Fold Cross-Validation Metrics',
                      'table_4_2_stratified_cv',
                      is_wide=True)

# Table 4.3
t43_data = [
    ['Recall', '0.8473', '[0.821, 0.874]'],
    ['Precision', '0.8312', '[0.805, 0.857]'],
    ['ROC-AUC', '0.8615', '[0.848, 0.875]'],
    ['Brier Score', '0.0948', '[0.088, 0.102]']
]
render_academic_table(t43_data,
                      ['Metric', 'Value', '95% Confidence Interval'],
                      'Table 4.3: Chronological Split Validation Metrics',
                      'table_4_3_chronological_split',
                      is_wide=False)

# Table 4.4
t44_data = [
    ['1', 'freq_daylight_match', '0.284', 'Frequency × daylight interaction;\ncaptures band-specific ionospheric\ndependency'],
    ['2', 'swr', '0.182', 'Standing Wave Ratio; primary\nindicator of antenna efficiency\nand radiated power'],
    ['3', 'doy_sin', '0.145', 'Seasonal variation; captures changes\nin solar zenith angle and F-layer\ndensity'],
    ['4', 'is_20m_long_distance', '0.112', '20m band suitability for\nintercontinental skip paths'],
    ['5', 'hour_cos', '0.098', 'Diurnal cycle; distinguishes between\nday and night propagation modes'],
    ['6', 'frequency_mhz', '0.065', 'Absolute carrier frequency;\ndetermines propagation mode\n(ground vs. sky wave)'],
    ['7', 'snr_margin', '0.042', 'Signal-to-Noise Ratio margin;\nindicates signal robustness above\ndecode threshold'],
    ['8', 'is_good_swr', '0.031', 'Binary indicator of efficient\nantenna tuning (SWR ≤ 1.5)'],
    ['9', 'is_40m', '0.021', '40m band indicator; relevant for\nnight-time propagation'],
    ['10', 'power_watts', '0.019', 'Transmitted power; less significant\nthan tuning efficiency']
]
# Pass is_wide=True to fix the text squashing
render_academic_table(t44_data,
                      ['Rank', 'Feature', 'Importance Score', 'Physical Interpretation'],
                      'Table 4.4: Top 10 Feature Importance Rankings',
                      'table_4_4_feature_importance',
                      is_wide=True)

print("\n Final tables generated with optimized layout.")