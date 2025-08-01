# 005_XGBoost_Modeling.py
"""
Loads cleaned data, trains an XGBoost Classifier, 
evaluates performance using default threshold (0.5), 
saves individual result plots, and a final summary image.
Threshold Tuning part is removed.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # For summary image
import seaborn as sns
import joblib

# --- Setup (Paths, Logging) - ללא שינוי ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
try:
    from config import (ENGINEERED_DATA_XLSX, RESULTS_DIR, PLOT_DIR, MODEL_DIR,
                        X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH, X_TEST_PROCESSED_PATH, Y_TEST_PATH) # שמות קבצים אלה יכולים להיות כלליים
    print("Successfully imported paths for 005 (XGBoost - Default Threshold)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
except ImportError:
    print("CRITICAL (005-XGBoost): Config import error.")
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    ENGINEERED_DATA_XLSX = project_root_alt/"data"/"surgery_data_engineered_v3.xlsx"; RESULTS_DIR = project_root_alt/"results"
    PLOT_DIR = project_root_alt/"plots"; MODEL_DIR = project_root_alt/"models"
    X_TRAIN_PROCESSED_PATH = project_root_alt/"data"/"005_X_train_for_xgboost_default_thresh.joblib" 
    Y_TRAIN_PATH = project_root_alt/"data"/"005_y_train_for_xgboost_default_thresh.joblib"
    X_TEST_PROCESSED_PATH = project_root_alt/"data"/"005_X_test_for_xgboost_default_thresh.joblib"
    Y_TEST_PATH = project_root_alt/"data"/"005_y_test_for_xgboost_default_thresh.joblib"
    for p in [RESULTS_DIR, PLOT_DIR, MODEL_DIR, project_root_alt/"data"]: p.mkdir(parents=True, exist_ok=True)
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".txt") -> Path:
    base_name = prefix; counter = 0
    while True:
        f_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not f_path.exists(): return f_path
        counter += 1
log_filename_base = Path(__file__).stem.replace("_and_Threshold_Tuning", "") # הסרת החלק של הטרשהולד מהשם
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_filepath, encoding='utf-8'), logging.StreamHandler(sys.stdout)])

# --- Start ---
logger.info(f"--- XGBoost Modeling (Default Threshold) (Script: {log_filename_base}) ---") # עדכון כותרת
logger.info(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Input: {ENGINEERED_DATA_XLSX}"); input_sheet_name = "features_focused_v1"
logger.info(f"Sheet: {input_sheet_name}")
if not ENGINEERED_DATA_XLSX.exists(): logger.error(f"Data file not found: {ENGINEERED_DATA_XLSX.resolve()}"); sys.exit(1)

# --- 1. Load & 1a. Handle Target NaNs & 2. Normalize - (ללא שינוי מהגרסה הקודמת) ---
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

# --- 3. Define X,y & 4. Handle NaNs in X - (ללא שינוי מהגרסה הקודמת) ---
logger.info("\n=== 3&4. Define X,y & Handle X NaNs ===")
y = df_model_input[target_col]
X = df_model_input.drop(columns=[target_col, 'plan_id', 'procedure_code'], errors='ignore')
if X.isnull().sum().any():
    logger.warning(f"NaNs in X:\n{X.isnull().sum()[X.isnull().sum()>0]}")
    X.dropna(inplace=True); y = y.loc[X.index]
    logger.info(f"Removed rows with NaNs in X. New X shape: {X.shape}")
if X.isnull().sum().any(): logger.error(f"CRITICAL: NaNs still in X. Halting."); sys.exit(1)
if len(X)==0: logger.error("X empty. Halting."); sys.exit(1)
logger.info(f"Final X shape: {X.shape}, y: {y.shape}. Features: {X.columns.tolist()}")

# --- 5. Train-Test Split - (ללא שינוי מהגרסה הקודמת) ---
logger.info("\n=== 5. Train-Test Split ===")
if y.nunique()<2: logger.error(f"Target has <2 unique values. Halting."); sys.exit(1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- 6. XGBoost Model Training - (ללא שינוי מהגרסה הקודמת) ---
logger.info("\n=== 6. Training XGBoost Classifier ===")
count_class_0, count_class_1 = y_train.value_counts().sort_index()
scale_pos_weight_val = count_class_0 / count_class_1 if count_class_1 > 0 else 1
logger.info(f"Calculated scale_pos_weight: {scale_pos_weight_val:.2f}")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight_val,
                              n_estimators=150,max_depth=7,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,
                              eval_metric='logloss',random_state=42,n_jobs=-1) # הסרתי use_label_encoder, כבר לא נתמך
logger.info(f"Training XGBoost with params: {xgb_model.get_params()}")
xgb_model.fit(X_train, y_train)
logger.info("XGBoost training complete.")

# --- 7. Model Evaluation (Default Threshold 0.5) & Individual Plots ---
# הסרנו את סעיף 7 (Threshold Tuning) המקורי, וסעיף 8 הופך להיות 7
logger.info("\n=== 7. XGBoost Evaluation (Default Threshold 0.5) & Individual Plots ===")
y_pred_xgb_default = xgb_model.predict(X_test) # שימוש ב-predict() שמשתמש בסף 0.5
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1] # עדיין צריך את ההסתברויות ל-ROC AUC ו-PR Curve

accuracy_xgb = accuracy_score(y_test, y_pred_xgb_default)
precision_xgb = precision_score(y_test, y_pred_xgb_default, pos_label=1, zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb_default, pos_label=1, zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb_default, pos_label=1, zero_division=0)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
avg_precision_xgb_curve = average_precision_score(y_test, y_pred_proba_xgb) # לשימוש במקרא של PR Curve

logger.info(f"Evaluation using default threshold (0.5)")
logger.info(f"Accuracy: {accuracy_xgb:.4f}, Precision: {precision_xgb:.4f}, Recall: {recall_xgb:.4f}, F1: {f1_xgb:.4f}, ROC AUC: {roc_auc_xgb:.4f}")
logger.info("\nClassification Report (Default Thresh):\n" + classification_report(y_test, y_pred_xgb_default, target_names=['Not Canceled(0)','Canceled(1)'], zero_division=0))
cm_xgb = confusion_matrix(y_test, y_pred_xgb_default)
logger.info(f"\nConfusion Matrix (Default Thresh):\n{cm_xgb}")

# גרפים נשארים זהים, רק הכותרות ישקפו שזה סף ברירת מחדל
cm_plot_path_xgb = PLOT_DIR / f"{log_filename_base}_confusion_matrix_default_thresh.png" # שינוי שם קובץ
plt.figure(figsize=(8,6)); sns.heatmap(cm_xgb,annot=True,fmt='d',cmap='Blues',xticklabels=['Pred Not Canceled','Pred Canceled'],yticklabels=['Actual Not Canceled','Actual Canceled'])
plt.title(f'CM - XGBoost (Default Thresh=0.5)'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(cm_plot_path_xgb); plt.close()
logger.info(f"Default CM plot saved: {cm_plot_path_xgb}")

roc_plot_path_xgb = PLOT_DIR / f"{log_filename_base}_roc_curve.png"
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.figure(figsize=(8,6)); plt.plot(fpr_xgb,tpr_xgb,color='green',lw=2,label=f'XGB ROC (AUC = {roc_auc_xgb:.2f})')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05])
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve - XGBoost'); plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.savefig(roc_plot_path_xgb); plt.close()
logger.info(f"XGBoost ROC plot saved: {roc_plot_path_xgb}")

# גרף ה-Precision-Recall המקורי (ללא הדגשת סף אופטימלי)
pr_plot_path_xgb_default = PLOT_DIR / f"{log_filename_base}_precision_recall_curve_default.png" # שינוי שם קובץ
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
plt.figure(figsize=(8,6)); plt.plot(recall_vals, precision_vals, label=f'PR curve (AP = {avg_precision_xgb_curve:.2f})', color='b', alpha=0.7)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve - XGBoost'); plt.legend(loc="lower left"); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(pr_plot_path_xgb_default); plt.close()
logger.info(f"PR curve (default) plot saved: {pr_plot_path_xgb_default}")

proba_dist_plot_path_xgb = PLOT_DIR / f"{log_filename_base}_pred_proba_dist.png"
plt.figure(figsize=(10,6)); 
sns.histplot(y_pred_proba_xgb[y_test == 0], label='Actual: Not Canceled (0)', color='skyblue', kde=True, stat="density", common_norm=False)
sns.histplot(y_pred_proba_xgb[y_test == 1], label='Actual: Canceled (1)', color='salmon', kde=True, stat="density", common_norm=False)
plt.title('Predicted Probabilities Distribution - XGB'); plt.xlabel('P(Canceled)'); plt.ylabel('Density'); plt.legend(); plt.grid(alpha=0.3); plt.savefig(proba_dist_plot_path_xgb); plt.close()
logger.info(f"XGBoost Proba dist plot saved: {proba_dist_plot_path_xgb}")

# --- 8. Feature Importances (XGBoost) --- (הופך להיות סעיף 8)
logger.info("\n=== 8. XGBoost Feature Importances ===")
# ... (הקוד לחשיבות משתנים נשאר זהה) ...
importances_xgb_vals = xgb_model.feature_importances_
fi_df_xgb = pd.DataFrame({'feature':X_train.columns, 'importance':importances_xgb_vals}).sort_values('importance',ascending=False)
logger.info("Top 20 FI (XGBoost):\n" + fi_df_xgb.head(20).to_string())
fi_plot_path_xgb = PLOT_DIR / f"{log_filename_base}_feature_importances.png"
plt.figure(figsize=(12,max(6, len(fi_df_xgb.head(20))*0.4))); sns.barplot(x='importance', y='feature', data=fi_df_xgb.head(20), palette='mako')
plt.title('Top 20 Feature Importances - XGBoost'); plt.tight_layout(); plt.savefig(fi_plot_path_xgb); plt.close()
logger.info(f"XGBoost FI plot saved: {fi_plot_path_xgb}")

# --- 9. Saving Data and Model --- (הופך להיות סעיף 9)
logger.info("\n=== 9. Saving Data & XGBoost Model ===")
# ... (הקוד לשמירת נתונים ומודל נשאר זהה, רק ללא שמירת הסף האופטימלי) ...
try:
    joblib.dump(X_train, X_TRAIN_PROCESSED_PATH); joblib.dump(y_train, Y_TRAIN_PATH)
    joblib.dump(X_test, X_TEST_PROCESSED_PATH); joblib.dump(y_test, Y_TEST_PATH)
    logger.info(f"Saved final train/test data splits to {X_TRAIN_PROCESSED_PATH.parent}")
    model_xgb_filename = MODEL_DIR / f"{log_filename_base}_model.joblib"; joblib.dump(xgb_model, model_xgb_filename) # שם קובץ המודל יכול להישאר גנרי או לציין xgb
    logger.info(f"XGBoost model saved: {model_xgb_filename}")
except Exception as e: logger.error(f"Error saving data or XGBoost model: {e}")

# --- 10. Create Summary Image --- (הופך להיות סעיף 10)
logger.info("\n=== 10. Creating Summary Image - XGBoost (Default Threshold) ===")
# ... (פונקציית create_summary_image נשארת זהה) ...
def create_summary_image(metrics_dict, cm_path, roc_path, pr_path, fi_path, proba_dist_path, output_path, model_name="Model"):
    fig, axs = plt.subplots(3, 2, figsize=(16, 20), gridspec_kw={'height_ratios': [1, 3, 3]})
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=20, y=0.99)
    ax_text = axs[0, 0]; ax_text.axis('off') 
    metrics_text = "Key Performance Metrics:\n\n"; 
    for k, v in metrics_dict.items(): metrics_text += f"{k}: {v:.4f}\n" if isinstance(v, (float,int)) else f"{k}: {v}\n"
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

xgb_metrics_summary_default = {"Accuracy": accuracy_xgb, "ROC AUC": roc_auc_xgb, "Precision (Canceled)": precision_xgb, 
                               "Recall (Canceled)": recall_xgb, "F1-score (Canceled)": f1_xgb, 
                               "Avg Precision (PR)": avg_precision_xgb_curve} # הסרתי את Optimal Threshold
summary_img_path_xgb = PLOT_DIR / f"{log_filename_base}_SUMMARY_default_thresh.png" # שינוי שם קובץ

create_summary_image(xgb_metrics_summary_default, str(cm_plot_path_xgb), str(roc_plot_path_xgb), 
                     str(pr_plot_path_xgb_default), str(fi_plot_path_xgb), # שימוש בגרף PR ללא הדגשת סף
                     str(proba_dist_plot_path_xgb), 
                     str(summary_img_path_xgb), f"XGBoost (Default Thresh=0.5)")

logger.info(f"\n--- XGBoost Script (Default Threshold) Complete ---")
logger.info(f"Log: {log_filepath.resolve()}")

if __name__ == "__main__":
    pass