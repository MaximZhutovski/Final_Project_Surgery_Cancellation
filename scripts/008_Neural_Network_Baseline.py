# 009_NN_Hyperparameter_Tuning.py
"""
Uses KerasTuner to automatically find the best hyperparameters for a Neural
Network on the optimized 15-feature set. After the search, it trains the best
model and generates a full visual report, including a final summary image.
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import keras_tuner as kt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Setup (Paths, Logging, etc.)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
try:
    from config import ENGINEERED_DATA_XLSX, RESULTS_DIR, PLOT_DIR, MODEL_DIR
    print("Successfully imported paths for 009 (NN Hyperparameter Tuning)")
    for p in [RESULTS_DIR, PLOT_DIR, MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)
except ImportError:
    print("CRITICAL (009-NN-Tune): Config import error.")
    scripts_dir=Path(__file__).resolve().parent; project_root_alt=scripts_dir.parent
    ENGINEERED_DATA_XLSX=project_root_alt/"data"/"surgery_data_engineered_v3.xlsx"
    RESULTS_DIR=project_root_alt/"results"; PLOT_DIR=project_root_alt/"plots"; MODEL_DIR=project_root_alt/"models"
    for p in [RESULTS_DIR, PLOT_DIR, MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)

def get_next_filename(base_dir, prefix, suffix=".txt"):
    base_name = prefix; counter = 0
    while True:
        f_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not f_path.exists(): return f_path
        counter += 1

log_filename_base = Path(__file__).stem
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
logger = logging.getLogger(__name__)
if logger.hasHandlers(): logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])

# ---------------------------------------------------------------------------
# Helper function for summary image - It's here!
# ---------------------------------------------------------------------------
def create_summary_image(metrics_dict, cm_path, roc_path, pr_path, fi_path, output_path, model_name="Model"):
    fig, axs = plt.subplots(3, 2, figsize=(16, 20), gridspec_kw={'height_ratios': [1, 3, 3]})
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=20, y=0.99)
    ax_text = axs[0, 0]; ax_text.axis('off')
    metrics_text = "Key Performance Metrics:\n\n"
    for k, v in metrics_dict.items(): metrics_text += f"{k}: {v:.4f}\n"
    ax_text.text(0.05, 0.95, metrics_text, transform=ax_text.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))
    axs[0, 1].axis('off')
    plot_details = [('Confusion Matrix', axs[1, 0], cm_path), ('ROC Curve', axs[1, 1], roc_path), ('Precision-Recall Curve', axs[2, 0], pr_path), ('Feature Importances', axs[2, 1], fi_path)]
    for title, ax, path_str in plot_details:
        path = Path(path_str)
        try:
            if path.exists(): img = mpimg.imread(path); ax.imshow(img); ax.set_title(title, fontsize=14)
            else: ax.text(0.5, 0.5, f'{title}\nNot Found', ha='center', va='center', color='red')
        except Exception: ax.text(0.5, 0.5, f'Err loading {title}', ha='center', va='center', color='red')
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(output_path); plt.close(fig)

# --- Start of Script ---
logger.info(f"--- Neural Network Hyperparameter Tuning with KerasTuner ---")

# 1. Load and Prepare Data (on 15 features)
logger.info("\n=== 1. Loading and Preparing Data ===")
input_sheet_name = "features_focused_v1"
try: df_full = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name=input_sheet_name)
except Exception as e: logger.error(f"Error loading Excel: {e}"); sys.exit(1)
target_col = 'was_canceled'
df_full.dropna(subset=[target_col], inplace=True)
df_full[target_col] = df_full[target_col].map({0: 1, 1: 0})
y = df_full[target_col]
cols_to_drop_for_X = [
    target_col, 'plan_id', 'procedure_code', 'age_bucket', 'city',
    'distance_bucket', 'is_weekend', 'wait_days_category'
]
X_base = df_full.drop(columns=cols_to_drop_for_X, errors='ignore')
if X_base.isnull().sum().any():
    valid_indices = X_base.dropna().index
    X = X_base.loc[valid_indices]; y = y.loc[valid_indices]
else: X = X_base
logger.info(f"Final Features Used ({len(X.columns)}): {X.columns.tolist()}")

# 2. Train-Test Split and Feature Scaling
logger.info("\n=== 2. Train-Test Split and Feature Scaling ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define the HyperModel build function
logger.info("\n=== 3. Defining the HyperModel for KerasTuner ===")
def build_model(hp):
    model = Sequential()
    hp_units_1 = hp.Int('units_1', min_value=16, max_value=128, step=16)
    model.add(Dense(units=hp_units_1, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(BatchNormalization())
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout_1))
    for i in range(hp.Int('num_layers', 1, 3)):
        hp_units = hp.Int(f'units_{i+2}', min_value=16, max_value=128, step=16)
        model.add(Dense(units=hp_units, activation='relu'))
        model.add(BatchNormalization())
        hp_dropout = hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(rate=hp_dropout))
    model.add(Dense(1, activation='sigmoid'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Instantiate the Tuner and perform the search
logger.info("\n=== 4. Performing Hyperparameter Search ===")
tuner_dir = Path('keras_tuner_dir')
if tuner_dir.exists(): shutil.rmtree(tuner_dir)
tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=20, factor=3,
                     directory=tuner_dir, project_name='surgery_cancellation')
stop_early = EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train_scaled, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
logger.info("\n--- Optimal Hyperparameters Found ---")
for key, value in best_hps.values.items():
    logger.info(f"{key}: {value}")

# 5. Build and Train the Final Model with the best hyperparameters
logger.info("\n=== 5. Training the Final Model with Best Hyperparameters ===")
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=150,
                         batch_size=32, callbacks=[stop_early], verbose=2)

# 6. Evaluate the Final Model and Generate Reports
logger.info("\n=== 6. Evaluating Final Model and Generating Reports ===")
y_pred_proba = best_model.predict(X_test_scaled).ravel()
y_pred_class = (y_pred_proba > 0.5).astype(int)
metrics_summary = {'Accuracy': accuracy_score(y_test, y_pred_class), 'ROC AUC': roc_auc_score(y_test, y_pred_proba), 'Avg Precision (PR)': average_precision_score(y_test, y_pred_proba)}
logger.info("\n--- FINAL RESULTS (Tuned NN - 15 FEATURES) ---")
for key, val in metrics_summary.items(): logger.info(f"{key}: {val:.4f}")
logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred_class, target_names=['Not Canceled (0)', 'Canceled (1)']))

# --- Generate Individual Plots ---
cm = confusion_matrix(y_test, y_pred_class)
cm_path = PLOT_DIR / f"{log_filename_base}_confusion_matrix.png"
plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Not Canceled', 'Pred Canceled'], yticklabels=['Actual Not Canceled', 'Actual Canceled']); plt.title('Confusion Matrix - Tuned NN'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(cm_path); plt.close()
logger.info(f"CM plot saved: {cm_path}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_path = PLOT_DIR / f"{log_filename_base}_roc_curve.png"
plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics_summary["ROC AUC"]:.2f})'); plt.plot([0,1],[0,1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve - Tuned NN'); plt.legend(); plt.savefig(roc_path); plt.close()
logger.info(f"ROC plot saved: {roc_path}")

prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
pr_path = PLOT_DIR / f"{log_filename_base}_pr_curve.png"
plt.figure(figsize=(8,6)); plt.plot(rec, prec, label=f'PR (AP = {metrics_summary["Avg Precision (PR)"]:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve - Tuned NN'); plt.legend(); plt.savefig(pr_path); plt.close()
logger.info(f"PR plot saved: {pr_path}")

# Manual Permutation Importance
logger.info("\n=== 7. Manually Calculating Permutation Feature Importance ===")
baseline_accuracy = metrics_summary['Accuracy']
importances = {}
for i, col in enumerate(X.columns):
    logger.info(f"Calculating importance for feature: {col}...")
    X_test_permuted = X_test_scaled.copy()
    np.random.seed(42); np.random.shuffle(X_test_permuted[:, i])
    permuted_accuracy = best_model.evaluate(X_test_permuted, y_test, verbose=0)[1]
    importances[col] = baseline_accuracy - permuted_accuracy
fi_df = pd.DataFrame.from_dict(importances, orient='index', columns=['importance_decrease'])
fi_df.sort_values(by='importance_decrease', ascending=False, inplace=True)
logger.info("Top Features (Manual Permutation Importance):\n")
logger.info(fi_df.to_string())
fi_path = PLOT_DIR / f"{log_filename_base}_feature_importance.png"
plt.figure(figsize=(12, 8)); sns.barplot(x=fi_df['importance_decrease'], y=fi_df.index, palette='viridis'); plt.title('Permutation Feature Importance - Tuned NN'); plt.xlabel("Importance (Mean Accuracy Decrease)"); plt.tight_layout(); plt.savefig(fi_path); plt.close()
logger.info(f"Feature importance plot saved: {fi_path}")

# --- Generate Summary Image ---
summary_img_path = PLOT_DIR / f"{log_filename_base}_SUMMARY.png"
create_summary_image(metrics_summary, cm_path, roc_path, pr_path, fi_path, summary_img_path, "Tuned Neural Network (15 Features)")
logger.info(f"Summary image saved: {summary_img_path}")

model_path = MODEL_DIR / f"{log_filename_base}_model.keras"
best_model.save(model_path)
logger.info(f"Final tuned model saved to: {model_path}")
logger.info(f"\n--- Script {log_filename_base} Complete ---")