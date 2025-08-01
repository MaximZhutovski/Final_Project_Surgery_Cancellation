# 007_Leave_One_Out_Refined.py
"""
Starts with a refined set of 15 features (excluding wait_days_category and others).
Performs a leave-one-out analysis on these 15 features to check for any
remaining noisy features that could be removed for further improvement.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging
import shutil

import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                             accuracy_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy.stats import randint
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup (Paths, Logging, Directories)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
try:
    from config import ENGINEERED_DATA_XLSX, RESULTS_DIR, PLOT_DIR
    print("Successfully imported paths for 007 (Leave-One-Out on Refined 15 Features)")
except ImportError:
    print("CRITICAL (007-LOO-Refined): Config import error.")
    scripts_dir=Path(__file__).resolve().parent; project_root_alt=scripts_dir.parent
    ENGINEERED_DATA_XLSX=project_root_alt/"data"/"surgery_data_engineered_v3.xlsx"
    RESULTS_DIR=project_root_alt/"results"; PLOT_DIR=project_root_alt/"plots"
    for p in [RESULTS_DIR, PLOT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".txt") -> Path:
    base_name = prefix; counter = 0
    while True:
        f_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not f_path.exists():
            return f_path
        counter += 1

log_filename_base = Path(__file__).stem + "_refined"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])

main_plot_dir = PLOT_DIR / log_filename_base
if main_plot_dir.exists():
    shutil.rmtree(main_plot_dir)
main_plot_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"--- Leave-One-Out Feature Tuning (on Refined 15 Features) ---")
logger.info(f"All plots for this run will be saved in subdirectories under: {main_plot_dir.resolve()}")
logger.info(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# 1. Load, Clean, Normalize Data
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Loading and Preparing Data ===")
input_sheet_name = "features_focused_v1"
try:
    df_full = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name=input_sheet_name)
    logger.info(f"Loaded data: {df_full.shape}")
except Exception as e:
    logger.error(f"Error loading Excel: {e}"); sys.exit(1)
target_col = 'was_canceled'
df_full.dropna(subset=[target_col], inplace=True)
df_full[target_col] = df_full[target_col].map({0: 1, 1: 0})
y = df_full[target_col]

# <-- תיקון: רשימת ההסרה המקורית, עכשיו עם wait_days_category -->
initial_cols_to_exclude_from_X = [
    target_col, 'plan_id', 'procedure_code', 'age_bucket', 'city',
    'distance_bucket', 'is_weekend', 'wait_days_category' # <-- הוספה
]
logger.info(f"Defining base feature set by excluding: {initial_cols_to_exclude_from_X}")

X_base = df_full.drop(columns=initial_cols_to_exclude_from_X, errors='ignore')
if X_base.isnull().sum().any():
    valid_indices = X_base.dropna().index
    X = X_base.loc[valid_indices]; y = y.loc[valid_indices]
else:
    X = X_base
features_to_test = X.columns.tolist()
logger.info(f"Base number of features for analysis: {len(features_to_test)}")
logger.info(f"Features list: {features_to_test}")

# ---------------------------------------------------------------------------
# 2. Define Helper Function for Creating Summary Image
# ---------------------------------------------------------------------------
def create_summary_image(metrics_dict, cm_path, roc_path, pr_path, fi_path, output_path, model_name="Model"):
    # (The function remains the same)
    fig, axs = plt.subplots(3, 2, figsize=(16, 20), gridspec_kw={'height_ratios': [1, 3, 3]})
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=20, y=0.99)
    ax_text = axs[0, 0]; ax_text.axis('off')
    metrics_text = "Key Performance Metrics:\n\n"
    for k, v in metrics_dict.items():
        metrics_text += f"{k}: {v:.4f}\n"
    ax_text.text(0.05, 0.95, metrics_text, transform=ax_text.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))
    axs[0, 1].axis('off')
    plot_details = [
        ('Confusion Matrix', axs[1, 0], cm_path), ('ROC Curve', axs[1, 1], roc_path),
        ('Precision-Recall Curve', axs[2, 0], pr_path), ('Feature Importances', axs[2, 1], fi_path)
    ]
    for title, ax, path_str in plot_details:
        path = Path(path_str)
        try:
            if path.exists():
                img = mpimg.imread(path); ax.imshow(img); ax.set_title(title, fontsize=14)
            else:
                ax.text(0.5, 0.5, f'{title}\nNot Found', ha='center', va='center', color='red')
        except Exception as e:
            ax.text(0.5, 0.5, f'Err loading {title}', ha='center', va='center', color='red')
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(output_path); plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Define the Full Evaluation & Reporting Function
# ---------------------------------------------------------------------------
def run_evaluate_and_report(X_data, y_data, scenario_name, base_plot_dir):
    # (The function remains the same)
    logger.info(f"\n--- Starting Scenario: {scenario_name} ---")
    scenario_plot_dir = base_plot_dir / scenario_name.replace(':', '').replace(' ', '_')
    scenario_plot_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
    )
    param_dist_rf = {
        'n_estimators': randint(100, 500), 'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': randint(2, 20), 'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', None], 'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist_rf, n_iter=50, cv=3,
        scoring='f1_weighted', verbose=0, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_proba),
        "Precision (Canceled)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (Canceled)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1-score (Canceled)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Avg Precision (PR)": average_precision_score(y_test, y_pred_proba)
    }
    cm = confusion_matrix(y_test, y_pred)
    cm_plot_path = scenario_plot_dir / "confusion_matrix.png"
    plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Not Canceled','Pred Canceled'], yticklabels=['Actual Not Canceled','Actual Canceled']); plt.title(f'CM - {scenario_name}'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(cm_plot_path); plt.close()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_plot_path = scenario_plot_dir / "roc_curve.png"
    plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {metrics["ROC AUC"]:.2f})'); plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve - {scenario_name}'); plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.savefig(roc_plot_path); plt.close()
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_plot_path = scenario_plot_dir / "precision_recall_curve.png"
    plt.figure(figsize=(8,6)); plt.step(rec_curve, prec_curve, color='b', alpha=0.7, where='post', label=f'PR curve (AP = {metrics["Avg Precision (PR)"]:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.ylim([0,1.05]); plt.xlim([0,1]); plt.title(f'PR Curve - {scenario_name}'); plt.legend(loc="lower left"); plt.grid(alpha=0.3); plt.savefig(pr_plot_path); plt.close()
    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({'feature':X_train.columns, 'importance':importances}).sort_values('importance',ascending=False)
    fi_plot_path = scenario_plot_dir / "feature_importances.png"
    plt.figure(figsize=(12,max(6, len(fi_df.head(20))*0.4))); sns.barplot(x='importance', y='feature', data=fi_df.head(20), palette='viridis'); plt.title(f'Top 20 FI - {scenario_name}'); plt.tight_layout(); plt.savefig(fi_plot_path); plt.close()
    summary_img_path = scenario_plot_dir / f"SUMMARY_{scenario_name.replace(':','').replace(' ','_')}.png"
    create_summary_image(
        metrics_dict={**{'Accuracy': metrics['Accuracy']}, **{k: v for k, v in metrics.items() if k != "Accuracy" and k != "Avg Precision (PR)"}},
        cm_path=cm_plot_path, roc_path=roc_plot_path, pr_path=pr_plot_path, fi_path=fi_plot_path,
        output_path=summary_img_path, model_name=f"Tuned RF ({scenario_name})"
    )
    return {
        'Scenario': scenario_name, 'Accuracy': metrics['Accuracy'], 'Test F1 (Canceled)': metrics['F1-score (Canceled)'],
        'Test Recall (Canceled)': metrics['Recall (Canceled)'], 'Test Precision (Canceled)': metrics['Precision (Canceled)'],
        'Test ROC AUC': metrics['ROC AUC'], 'Best CV Score': search.best_score_
    }

# ---------------------------------------------------------------------------
# 4. Run All Scenarios and Collect Results
# ---------------------------------------------------------------------------
all_results = []
# The list of scenarios is now 1 (baseline) + 15 (leave-one-out) = 16 total
scenarios_to_run = ["Baseline (15 Features)"] + [f"Removed: {f}" for f in features_to_test]

logger.info("\n=== 4. Starting Main Evaluation Loop ===")
progress_bar = tqdm(scenarios_to_run, desc="Overall Progress", unit="scenario")

for scenario_name in progress_bar:
    progress_bar.set_postfix_str(f"Current: {scenario_name}")
    if scenario_name == "Baseline (15 Features)":
        results = run_evaluate_and_report(X, y, scenario_name, main_plot_dir)
        baseline_results = results
    else:
        feature_to_remove = scenario_name.replace("Removed: ", "")
        X_subset = X.drop(columns=[feature_to_remove])
        results = run_evaluate_and_report(X_subset, y, scenario_name, main_plot_dir)
    all_results.append(results)

# ---------------------------------------------------------------------------
# 5. Final Summary and Recommendation
# ---------------------------------------------------------------------------
logger.info("\n\n" + "="*80)
logger.info("=== FINAL OVERALL SUMMARY - LEAVE-ONE-OUT (ON 15 REFINED FEATURES) ===")
logger.info("="*80)
results_df = pd.DataFrame(all_results)
column_order = ['Accuracy', 'Test F1 (Canceled)', 'Test Recall (Canceled)', 'Test Precision (Canceled)', 'Test ROC AUC', 'Best CV Score']
results_df = results_df.set_index('Scenario')[column_order]
results_df = results_df.sort_values(by='Test F1 (Canceled)', ascending=False)
pd.options.display.float_format = '{:.4f}'.format
logger.info("\nPerformance Metrics for Each Scenario (Sorted by Test F1 Score):\n")
logger.info(results_df.to_string())
logger.info("\n" + "="*80)
logger.info("=== FINAL RECOMMENDATION ===")
logger.info("="*80)
baseline_f1 = baseline_results['Test F1 (Canceled)']
best_scenario_row = results_df.iloc[0]
best_scenario_name = best_scenario_row.name
best_f1 = best_scenario_row['Test F1 (Canceled)']
logger.info(f"Baseline F1 Score (with {len(features_to_test)} features): {baseline_f1:.4f}")
logger.info(f"Best F1 Score achieved: {best_f1:.4f} (in scenario: '{best_scenario_name}')")
if best_f1 > baseline_f1:
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
    logger.info(f"\n✅ RECOMMENDATION: Remove the feature '{best_scenario_name.replace('Removed: ', '')}'.")
    logger.info(f"This resulted in an F1 score improvement of {improvement:.2f}%.")
    logger.info(f"Review the detailed visual report for this scenario in the directory: plots\\{log_filename_base}\\{best_scenario_name.replace(': ','_').replace(' ','_')}")
else:
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
    logger.info(f"\nℹ️ RECOMMENDATION: Keep all {len(features_to_test)} features.")
    logger.info("Removing any single feature did not improve the F1 score on the test set.")
    if improvement < 0:
        logger.info(f"The best result after removing a feature was actually a decrease of {abs(improvement):.2f}% in F1 score.")
    logger.info(f"You should proceed with the full set of {len(features_to_test)} features. See the baseline report for details.")
logger.info(f"\n--- Full analysis and reporting complete. Log saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass