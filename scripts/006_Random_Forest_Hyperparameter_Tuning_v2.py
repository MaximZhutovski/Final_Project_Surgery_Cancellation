# 006_Random_Forest_Hyperparameter_Tuning_v3.py
"""
Loads cleaned data, EXCLUDES a specific list of features to remain with 15,
performs hyperparameter tuning for Random Forest using RandomizedSearchCV,
trains a final model with best parameters, evaluates, and saves a full visual report.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import joblib
from scipy.stats import randint

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
try:
    from config import (ENGINEERED_DATA_XLSX, RESULTS_DIR, PLOT_DIR, MODEL_DIR) 
    print("Successfully imported paths for 006 (RF Hyperparameter Tuning v3 - 15 Features)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
except ImportError: # Fallback paths
    print("CRITICAL (006-RF-Tune-v3): Config import error.")
    scripts_dir=Path(__file__).resolve().parent; project_root_alt=scripts_dir.parent
    ENGINEERED_DATA_XLSX=project_root_alt/"data"/"surgery_data_engineered_v3.xlsx"; RESULTS_DIR=project_root_alt/"results"
    PLOT_DIR=project_root_alt/"plots"; MODEL_DIR=project_root_alt/"models"
    for p in [RESULTS_DIR,PLOT_DIR,MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup Output Logging
# ---------------------------------------------------------------------------
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".txt") -> Path:
    base_name = prefix; counter = 0
    while True:
        f_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not f_path.exists(): return f_path
        counter += 1

log_filename_base = Path(__file__).stem.replace('_v2', '_v3') # New log file name
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
logger = logging.getLogger(__name__)
if logger.hasHandlers(): logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])

# --- Start of Script ---
logger.info(f"--- Random Forest Hyperparameter Tuning (Script: {log_filename_base}) ---")
logger.info(f"--- VERSION: Using 15 features (wait_days_category excluded) ---")
logger.info(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Input: {ENGINEERED_DATA_XLSX}"); input_sheet_name = "features_focused_v1"
logger.info(f"Sheet: {input_sheet_name}")
if not ENGINEERED_DATA_XLSX.exists(): logger.error(f"Data file not found: {ENGINEERED_DATA_XLSX.resolve()}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load, Clean, Normalize
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Load, Initial Clean, Normalize ===")
try: df_model_input = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name=input_sheet_name)
except Exception as e: logger.error(f"Error loading Excel: {e}"); sys.exit(1)
logger.info(f"Loaded data: {df_model_input.shape}")
target_col = 'was_canceled'
if target_col not in df_model_input.columns: logger.error(f"Target '{target_col}' not found."); sys.exit(1)
df_model_input.dropna(subset=[target_col], inplace=True)
if df_model_input.empty: logger.error("Empty DF after target NaNs removal."); sys.exit(1)
cols_to_norm = {target_col: {0:1,1:0}, 'is_weekend': {0:1,1:0}, 'near_holiday': {0:1,1:0}}
for col, mapping in cols_to_norm.items():
    if col in df_model_input.columns:
        unique_v = set(df_model_input[col].dropna().unique())
        if unique_v <= {0,1} or unique_v <= {0.0,1.0}: df_model_input[col] = df_model_input[col].map(mapping)
        elif unique_v == {True,False}: df_model_input[col] = df_model_input[col].astype(int)
        df_model_input[col] = pd.to_numeric(df_model_input[col],errors='coerce').fillna(-1).astype(int)
        if -1 in df_model_input[col].unique(): logger.warning(f"NaNs in '{col}' became -1.")
if -1 in df_model_input[target_col].unique(): logger.error(f"Target '{target_col}' invalid. Halting."); sys.exit(1)
logger.info(f"Confirmed target variable normalization: 1 means 'Canceled'.")

# ---------------------------------------------------------------------------
# 2. Define X,y & Handle NaNs
# ---------------------------------------------------------------------------
logger.info("\n=== 2. Define X,y & Handle X NaNs ===")
y = df_model_input[target_col]

# <-- השינוי המרכזי כאן -->
cols_to_drop = [
    target_col, 
    'plan_id', 
    'procedure_code',
    'age_bucket',
    'city',
    'distance_bucket',
    'is_weekend',
    'wait_days_category' # <-- הוספה
]
logger.info(f"Excluding the following columns from features (X): {cols_to_drop}")

X = df_model_input.drop(columns=cols_to_drop, errors='ignore')

if X.isnull().sum().any():
    logger.warning(f"NaNs found in X. Dropping rows with NaNs...")
    logger.warning(f"NaNs count per column:\n{X.isnull().sum()[X.isnull().sum()>0]}")
    X.dropna(inplace=True); y = y.loc[X.index]
    logger.info(f"Removed rows with NaNs in X. New X shape: {X.shape}")
if X.isnull().sum().any(): logger.error(f"CRITICAL: NaNs still in X after dropping. Halting."); sys.exit(1)
if len(X)==0: logger.error("X is empty after processing. Halting."); sys.exit(1)

logger.info(f"Final X shape for modeling: {X.shape}")
logger.info(f"Final y shape for modeling: {y.shape}")
logger.info(f"Features included in the model ({len(X.columns)}):\n{X.columns.tolist()}")

# ---------------------------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------------------------
logger.info("\n=== 3. Train-Test Split ===")
if y.nunique()<2: logger.error(f"Target has <2 unique values. Halting."); sys.exit(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# ---------------------------------------------------------------------------
# 4. Random Forest Hyperparameter Tuning
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Random Forest Hyperparameter Tuning ===")
param_dist_rf = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None] 
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_random_search = RandomizedSearchCV(
    estimator=rf_base, param_distributions=param_dist_rf,
    n_iter=50, cv=3, scoring='f1_weighted',
    verbose=2, random_state=42, n_jobs=-1
)

logger.info(f"Starting RandomizedSearchCV...")
rf_random_search.fit(X_train, y_train)

logger.info("RandomizedSearchCV complete.")
logger.info(f"Best parameters found: {rf_random_search.best_params_}")
logger.info(f"Best cross-validation score ({rf_random_search.scoring}): {rf_random_search.best_score_:.4f}")
best_rf_model = rf_random_search.best_estimator_

# ---------------------------------------------------------------------------
# 5. Prediction, Evaluation & Plots
# ---------------------------------------------------------------------------
logger.info("\n=== 5. Tuned RF Prediction, Evaluation & Plots ===")
y_pred_rf_tuned = best_rf_model.predict(X_test)
y_pred_proba_rf_tuned = best_rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_rf_tuned),
    "Precision (Canceled)": precision_score(y_test, y_pred_rf_tuned, pos_label=1, zero_division=0),
    "Recall (Canceled)": recall_score(y_test, y_pred_rf_tuned, pos_label=1, zero_division=0),
    "F1-score (Canceled)": f1_score(y_test, y_pred_rf_tuned, pos_label=1, zero_division=0),
    "ROC AUC": roc_auc_score(y_test, y_pred_proba_rf_tuned),
    "Avg Precision (PR)": average_precision_score(y_test, y_pred_proba_rf_tuned)
}
for name, value in metrics.items():
    logger.info(f"Tuned RF - {name}: {value:.4f}")

logger.info("\nTuned RF - Classification Report:\n" + classification_report(y_test, y_pred_rf_tuned, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0))
cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)
logger.info(f"\nTuned RF - Confusion Matrix:\n{cm_rf_tuned}")

# --- Generate Plots ---
# CM Plot
cm_plot_path_rf_tuned = PLOT_DIR / f"{log_filename_base}_confusion_matrix.png"
plt.figure(figsize=(8,6)); sns.heatmap(cm_rf_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Not Canceled','Pred Canceled'], yticklabels=['Actual Not Canceled','Actual Canceled']); plt.title('CM - Tuned RF (15 Features)'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(cm_plot_path_rf_tuned); plt.close()
logger.info(f"CM plot saved: {cm_plot_path_rf_tuned}")

# ROC Curve
fpr_rf_tuned, tpr_rf_tuned, _ = roc_curve(y_test, y_pred_proba_rf_tuned)
roc_plot_path_rf_tuned = PLOT_DIR / f"{log_filename_base}_roc_curve.png"
plt.figure(figsize=(8,6)); plt.plot(fpr_rf_tuned, tpr_rf_tuned, color='darkorange', lw=2, label=f'ROC (AUC = {metrics["ROC AUC"]:.2f})'); plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve - Tuned RF (15 Features)'); plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.savefig(roc_plot_path_rf_tuned); plt.close()
logger.info(f"ROC plot saved: {roc_plot_path_rf_tuned}")

# PR Curve
precision_rf_curve_tuned, recall_rf_curve_tuned, _ = precision_recall_curve(y_test, y_pred_proba_rf_tuned)
pr_plot_path_rf_tuned = PLOT_DIR / f"{log_filename_base}_precision_recall_curve.png"
plt.figure(figsize=(8,6)); plt.step(recall_rf_curve_tuned, precision_rf_curve_tuned, color='b', alpha=0.7, where='post', label=f'PR curve (AP = {metrics["Avg Precision (PR)"]:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.ylim([0,1.05]); plt.xlim([0,1]); plt.title('PR Curve - Tuned RF (15 Features)'); plt.legend(loc="lower left"); plt.grid(alpha=0.3); plt.savefig(pr_plot_path_rf_tuned); plt.close()
logger.info(f"PR curve plot saved: {pr_plot_path_rf_tuned}")

# Feature Importance
importances_rf_tuned = best_rf_model.feature_importances_
fi_df_rf_tuned = pd.DataFrame({'feature':X_train.columns, 'importance':importances_rf_tuned}).sort_values('importance',ascending=False)
logger.info("\nTop Feature Importances (Tuned RF):\n" + fi_df_rf_tuned.to_string())
fi_plot_path_rf_tuned = PLOT_DIR / f"{log_filename_base}_feature_importances.png"
plt.figure(figsize=(12,max(6, len(fi_df_rf_tuned)*0.4))); sns.barplot(x='importance', y='feature', data=fi_df_rf_tuned, palette='viridis'); plt.title('Feature Importances - Tuned RF (15 Features)'); plt.tight_layout(); plt.savefig(fi_plot_path_rf_tuned); plt.close()
logger.info(f"FI plot saved: {fi_plot_path_rf_tuned}")

# ---------------------------------------------------------------------------
# 6. Create Summary Image
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Creating Summary Image ===")
def create_summary_image(metrics_dict, cm_path, roc_path, pr_path, fi_path, output_path, model_name="Model"):
    fig, axs = plt.subplots(3, 2, figsize=(16, 20), gridspec_kw={'height_ratios': [1, 3, 3]})
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=20, y=0.99)
    ax_text = axs[0, 0]; ax_text.axis('off') 
    metrics_text = "Key Performance Metrics:\n\n"; 
    for k, v in metrics_dict.items(): metrics_text += f"{k}: {v:.4f}\n"
    ax_text.text(0.05,0.95,metrics_text,transform=ax_text.transAxes,fontsize=12,va='top',bbox=dict(boxstyle='round,pad=0.5',fc='aliceblue',alpha=0.7))
    axs[0, 1].axis('off') 
    plot_details = [
        ('Confusion Matrix', axs[1,0], cm_path), ('ROC Curve', axs[1,1], roc_path),
        ('Precision-Recall Curve', axs[2,0], pr_path), ('Feature Importances', axs[2,1], fi_path)
    ]
    for title, ax, path_str in plot_details:
        path = Path(path_str)
        try:
            if path.exists(): img = mpimg.imread(path); ax.imshow(img); ax.set_title(title, fontsize=14)
            else: logger.warning(f"Plot missing for summary: {path}"); ax.text(0.5,0.5,f'{title}\nNot Found',ha='center',va='center',color='red')
        except Exception as e: logger.error(f"Err loading {path} for summary: {e}"); ax.text(0.5,0.5,f'Err {title}',ha='center',va='center',color='red')
        ax.axis('off')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(output_path); plt.close(fig)
    logger.info(f"Summary image saved: {output_path}")

summary_img_path_rf_tuned = PLOT_DIR / f"{log_filename_base}_SUMMARY.png"
create_summary_image(metrics, str(cm_plot_path_rf_tuned), str(roc_plot_path_rf_tuned), 
                     str(pr_plot_path_rf_tuned), str(fi_plot_path_rf_tuned),
                     str(summary_img_path_rf_tuned), f"Tuned Random Forest (15 Features)")

logger.info(f"\n--- Tuned RF Modeling Script (v3) Complete ---")
logger.info(f"Log: {log_filepath.resolve()}")

if __name__ == "__main__":
    pass