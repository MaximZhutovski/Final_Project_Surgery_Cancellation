# 009_Hybrid_NN_with_Embeddings.py
"""
Builds a state-of-the-art hybrid Neural Network for tabular data.
This version generates a full visual report including a summary image,
and saves all test set predictions to a CSV file for detailed analysis.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Setup (Paths, Logging, etc.)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
try:
    from config import ENGINEERED_DATA_XLSX, RESULTS_DIR, PLOT_DIR, MODEL_DIR
    print("Successfully imported paths for 010 (Hybrid NN - Full Output)")
    for p in [RESULTS_DIR, PLOT_DIR, MODEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)
except ImportError:
    print("CRITICAL (010-Hybrid-NN): Config import error.")
    scripts_dir=Path(__file__).resolve().parent; project_root_alt=scripts_dir.parent
    ENGINEERED_DATA_XLSX=project_root_alt/"data"/"surgery_data_engineered_v3.xlsx"
    RESULTS_DIR=project_root_alt/"results"; PLOT_DIR=project_root_alt/"plots"; MODEL_DIR=project_root_alt/"models"
    for p in [RESULTS_DIR, PLOT_DIR, MODEL_DIR]: p.mkdir(parents=True, exist_ok=True)

def get_next_filename(base_dir, prefix, suffix=".txt"):
    base_name = prefix; counter = 0
    while True:
        f_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not f_path.exists():
            return f_path
        counter += 1

log_filename_base = Path(__file__).stem
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])

# ---------------------------------------------------------------------------
# Helper function for summary image
# ---------------------------------------------------------------------------
def create_summary_image(metrics_dict, cm_path, roc_path, pr_path, output_path, model_name="Model"):
    # Simplified version for this model, as there's no single feature importance plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=20, y=0.98)

    # Metrics Text
    ax_text = axs[0, 0]
    ax_text.axis('off')
    metrics_text = "Key Performance Metrics:\n\n"
    for k, v in metrics_dict.items():
        metrics_text += f"{k}: {v:.4f}\n"
    ax_text.text(0.05, 0.95, metrics_text, transform=ax_text.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    # Placeholder for the 4th plot area
    axs[1,1].axis('off')

    plot_details = [
        ('Confusion Matrix', axs[0, 1], cm_path),
        ('ROC Curve', axs[1, 0], roc_path),
        # Move PR curve to bottom right for better layout
        ('Precision-Recall Curve', axs[1,1], pr_path)
    ]

    for title, ax, path_str in plot_details:
        path = Path(path_str)
        try:
            if path.exists():
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.set_title(title, fontsize=14)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'{title}\nNot Found', ha='center', va='center', color='red')
                ax.axis('off')
        except Exception as e:
            logger.error(f"Error loading {path_str} for summary: {e}")
            ax.text(0.5, 0.5, f'Error loading plot', ha='center', va='center', color='red')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)

# --- Start of Script ---
logger.info(f"--- Hybrid Neural Network with Embeddings (Full Output) ---")

# 1. Load and Prepare Data
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
categorical_features = [col for col in X.columns if X[col].nunique() < 25 and col not in ['near_holiday']]
numerical_features = [col for col in X.columns if col not in categorical_features]
logger.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")
logger.info(f"Identified {len(numerical_features)} numerical features: {numerical_features}")

# 2. Train-Test Split
logger.info("\n=== 2. Train-Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Preprocess Data
logger.info("\n=== 3. Preprocessing Data for Hybrid Model ===")
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_num_scaled = scaler.transform(X_test[numerical_features])
X_train_inputs = [X_train_num_scaled] + [X_train[col].values for col in categorical_features]
X_test_inputs = [X_test_num_scaled] + [X_test[col].values for col in categorical_features]

# 4. Build the Hybrid Model with Embeddings
logger.info("\n=== 4. Building the Hybrid Model ===")
numerical_input = Input(shape=(len(numerical_features),), name='numerical_input')
categorical_inputs = [Input(shape=(1,), name=f'{col}_input') for col in categorical_features]
num_path = Dense(32, activation='relu')(numerical_input)
num_path = BatchNormalization()(num_path)
embedding_paths = []
for i, col in enumerate(categorical_features):
    num_unique_values = X[col].nunique()
    embedding_size = int(min(np.ceil(num_unique_values / 2), 50))
    embedding_layer = Embedding(input_dim=num_unique_values + 1, output_dim=embedding_size, name=f'{col}_embedding')(categorical_inputs[i])
    flattened_embedding = Flatten()(embedding_layer)
    embedding_paths.append(flattened_embedding)
all_paths = [num_path] + embedding_paths
concatenated = Concatenate()(all_paths)
x = Dense(128, activation='relu')(concatenated)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])
model.summary(print_fn=logger.info)

# 5. Train the Model
logger.info("\n=== 5. Training the Hybrid Model ===")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
history = model.fit(X_train_inputs, y_train, validation_split=0.2, epochs=150,
                    batch_size=64, callbacks=[early_stopping, reduce_lr], verbose=2)

# 6. Evaluate and Report
logger.info("\n=== 6. Evaluating Final Model and Generating Reports ===")
y_pred_proba = model.predict(X_test_inputs).ravel()
y_pred_class = (y_pred_proba > 0.5).astype(int)
metrics_summary = {'Accuracy': accuracy_score(y_test, y_pred_class), 'ROC AUC': roc_auc_score(y_test, y_pred_proba), 'Avg Precision (PR)': average_precision_score(y_test, y_pred_proba)}
logger.info("\n--- FINAL RESULTS (Hybrid NN with Embeddings) ---")
for key, val in metrics_summary.items(): logger.info(f"{key}: {val:.4f}")
logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred_class, target_names=['Not Canceled (0)', 'Canceled (1)']))

# --- Generate Visual Reports ---
cm = confusion_matrix(y_test, y_pred_class)
cm_path = PLOT_DIR / f"{log_filename_base}_confusion_matrix.png"
plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Not Canceled', 'Pred Canceled'], yticklabels=['Actual Not Canceled', 'Actual Canceled']); plt.title('Confusion Matrix - Hybrid NN'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(cm_path); plt.close()
logger.info(f"CM plot saved: {cm_path}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_path = PLOT_DIR / f"{log_filename_base}_roc_curve.png"
plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics_summary["ROC AUC"]:.2f})'); plt.plot([0,1],[0,1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve - Hybrid NN'); plt.legend(); plt.savefig(roc_path); plt.close()
logger.info(f"ROC plot saved: {roc_path}")

prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
pr_path = PLOT_DIR / f"{log_filename_base}_pr_curve.png"
plt.figure(figsize=(8,6)); plt.plot(rec, prec, label=f'PR (AP = {metrics_summary["Avg Precision (PR)"]:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve - Hybrid NN'); plt.legend(); plt.savefig(pr_path); plt.close()
logger.info(f"PR plot saved: {pr_path}")

# --- Generate Summary Image ---
summary_img_path = PLOT_DIR / f"{log_filename_base}_SUMMARY.png"
# Note: Feature Importance is not applicable for this complex model, so we pass an empty path.
create_summary_image(metrics_summary, cm_path, roc_path, pr_path, "", summary_img_path, "Hybrid NN with Embeddings")
logger.info(f"Summary image saved: {summary_img_path}")

# --- Save Predictions to CSV ---
logger.info("\n=== 7. Saving Test Set Predictions to CSV ===")
# Create a DataFrame with the original test data and the predictions
predictions_df = X_test.copy()
predictions_df['actual_cancellation'] = y_test
predictions_df['predicted_cancellation_probability'] = y_pred_proba
predictions_df['predicted_cancellation_class'] = y_pred_class

# Save to a CSV file in the results directory
predictions_csv_path = RESULTS_DIR / f"{log_filename_base}_predictions.csv"
try:
    predictions_df.to_csv(predictions_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Successfully saved all test predictions to: {predictions_csv_path}")
except Exception as e:
    logger.error(f"Failed to save predictions CSV: {e}")

# Save the trained model
model_path = MODEL_DIR / f"{log_filename_base}_model.keras"
model.save(model_path)
logger.info(f"Final model saved to: {model_path}")
logger.info(f"\n--- Script {log_filename_base} Complete ---")