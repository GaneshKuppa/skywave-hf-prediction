"""
evaluation.py - Advanced Visualization & Reporting
Generates publication-quality figures and structured metric tables for the thesis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from config import CONFIG, MODEL_DIR, EDA_DIR

# Set style for professional plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_validation_comparison(cv_metrics: dict, chrono_metrics: dict):
    """
    Generates a comparison bar chart between Random CV and Chronological Split.
    Highlights the "Generalization Gap" to prove future-readiness.
    """
    metrics = ['recall', 'precision', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data extraction
    cv_vals = [cv_metrics[m] for m in metrics]
    chrono_vals = [chrono_metrics[m] for m in metrics]

    bars1 = ax.bar(x - width/2, cv_vals, width, label='Stratified CV (Random)', color='#4C72B0')
    bars2 = ax.bar(x + width/2, chrono_vals, width, label='Chronological (Future)', color='#DD8452')

    ax.set_xlabel('Evaluation Metric')
    ax.set_ylabel('Score')
    ax.set_title('Validation Strategy Comparison: Random vs. Chronological Split')
    ax.set_xticks(x)
    ax.set_xticklabels(['Recall', 'Precision', 'ROC-AUC'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    output_path = MODEL_DIR / 'validation_comparison.png'
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved validation comparison to {output_path}")

def plot_ablation_results(ablation_results: dict):
    """
    Visualizes the ablation study to justify feature engineering choices.
    Shows how adding physics-informed features improves recall.
    """
    # Sort results by feature count or recall
    groups = list(ablation_results.keys())
    recalls = [ablation_results[g]['recall'] for g in groups]
    features_count = [ablation_results[g]['features'] for g in groups]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Feature Group')
    ax1.set_ylabel('Recall Score', color=color)
    ax1.bar(groups, recalls, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)

    # Add feature count labels
    for i, v in enumerate(recalls):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        ax1.text(i, v - 0.1, f'Features: {features_count[i]}', ha='center', fontsize=8, color='gray')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Number of Features', color=color)
    ax2.plot(groups, features_count, marker='o', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Ablation Study: Impact of Feature Groups on Model Performance')
    plt.xticks(rotation=45, ha='right')
    
    output_path = MODEL_DIR / 'ablation_study.png'
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved ablation study plot to {output_path}")

def plot_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 15):
    """
    Horizontal bar chart of feature importance.
    Essential for discussing "Physics Alignment" in the report.
    """
    # Sort features
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices[:top_n]]
    sorted_importances = importances[indices][:top_n]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.xlabel('Gini Importance')
    plt.ylabel('Feature Name')
    
    # Add value labels
    for i, v in enumerate(sorted_importances):
        plt.text(v, i, f' {v:.4f}', va='center')

    output_path = MODEL_DIR / 'feature_importance.png'
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved feature importance to {output_path}")

def plot_confusion_matrix(y_true: list, y_pred: list):
    """
    Heatmap visualization of the Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Failure (0)', 'Success (1)'],
                yticklabels=['Failure (0)', 'Success (1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    output_path = MODEL_DIR / 'confusion_matrix.png'
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved confusion matrix to {output_path}")

def plot_roc_curve(y_true: list, y_proba: list):
    """
    ROC Curve with AUC score annotation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    output_path = MODEL_DIR / 'roc_curve.png'
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved ROC curve to {output_path}")

def plot_calibration_curve(y_true: list, y_proba: list):
    """
    Reliability Diagram to assess probability calibration.
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model Calibration")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    
    output_path = MODEL_DIR / 'calibration_curve.png'
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved calibration curve to {output_path}")

# =============================================================================
# REPORTING HELPERS
# =============================================================================

def export_metrics_table(metrics: dict, output_path: str = None):
    """
    Generates a clean, formatted text table of metrics for easy copy-pasting into the report.
    """
    table_str = "=====================================================================\n"
    table_str += "MODEL PERFORMANCE METRICS\n"
    table_str += "=====================================================================\n"
    
    for key, value in metrics.items():
        if isinstance(value, float):
            table_str += f"{key:<25} : {value:.4f}\n"
        else:
            table_str += f"{key:<25} : {value}\n"
            
    table_str += "=====================================================================\n"
    
    print(table_str)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"✅ Saved metrics table to {output_path}")

# =============================================================================
# MAIN EXECUTION (For Testing/Stand-alone Runs)
# =============================================================================

if __name__ == "__main__":
    print("🧪 Running Evaluation Tests...")
    
    # Check if results exist
    val_path = MODEL_DIR / 'validation_comparison.json'
    abl_path = MODEL_DIR / 'ablation_study.json'
    
    if val_path.exists():
        with open(val_path) as f:
            val_data = json.load(f)
        plot_validation_comparison(val_data['random_cv'], val_data['chronological_split'])
    
    if abl_path.exists():
        with open(abl_path) as f:
            abl_data = json.load(f)
        plot_ablation_results(abl_data)
        
    print("✅ Evaluation tests complete.")