# config.py – Resilient project-root detection and shared paths
"""
Shared path configuration for the Surgery_Cancellation project.
This version aligns with the re-numbered main script pipeline (001-009).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Detect project root
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent if script_dir.name.lower() == "scripts" else script_dir
PROJECT_ROOT = Path(os.getenv("BASE_DIR", PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard directories
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
PLOT_DIR = Path(os.getenv("PLOT_DIR", PROJECT_ROOT / "plots"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", DATA_DIR / "models"))

# ---------------------------------------------------------------------------
# Script 001: Initial Data Understanding
# ---------------------------------------------------------------------------
RAW_DATA_XLSX = DATA_DIR / "Surgery_Data.xlsx" # Input for 001

# ---------------------------------------------------------------------------
# Script 002: Feature Engineering
# ---------------------------------------------------------------------------
# Input for 002 is RAW_DATA_XLSX
ENGINEERED_DATA_XLSX = DATA_DIR / "surgery_data_engineered_v3.xlsx" # Output of 002 (v3)
# Example path for LMS data (ensure this file exists and names are correct if used in 002)
LMS_SOCIOECONOMIC_XLSX = DATA_DIR / "socioeconomic_lms_2017.xlsx" # Needs to be user-provided
LMS_SHEET_NAME = "אשכולות ויישובים וקודי יישובים" # Example, adjust
LMS_CITY_COL = "שם יישוב" # Example, adjust
LMS_CLUSTER_COL = "ממוצע מדד חברתי-כלכלי 2017" # Example, adjust


# ---------------------------------------------------------------------------
# Script 003: Engineered Data Analysis
# ---------------------------------------------------------------------------
# Input for 003 is ENGINEERED_DATA_XLSX

# ---------------------------------------------------------------------------
# Script 004: Data Final Preprocessing
# ---------------------------------------------------------------------------
# Input for 004 is ENGINEERED_DATA_XLSX
PREPROCESSED_X_INTERMEDIATE_XLSX = DATA_DIR / "004_X_pre_pipeline.xlsx" # Intermediate Excel output from 004

# Output of 004 (processed data for models)
X_TRAIN_PROCESSED_PATH = DATA_DIR / "004_X_train_processed.joblib"
Y_TRAIN_PATH = DATA_DIR / "004_y_train.joblib"
X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib"
Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"
PREPROCESSOR_PIPELINE_PATH = DATA_DIR / "004_preprocessor_pipeline.joblib"
PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "004_processed_feature_names.joblib"

# ---------------------------------------------------------------------------
# Script 005: XGBoost Baseline Model
# ---------------------------------------------------------------------------
# Inputs: X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH, X_TEST_PROCESSED_PATH, Y_TEST_PATH
XGB_BASELINE_MODEL_PATH = MODEL_DIR / "005_xgb_baseline_model.joblib"

# ---------------------------------------------------------------------------
# Script 006: XGBoost Hyperparameter Tuning
# ---------------------------------------------------------------------------
# Inputs: X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH, X_TEST_PROCESSED_PATH, Y_TEST_PATH
BEST_XGB_PARAMS_PATH = RESULTS_DIR / "006_best_xgb_params.joblib"
XGB_TUNED_MODEL_PATH = MODEL_DIR / "006_xgb_tuned_model.joblib"

# ---------------------------------------------------------------------------
# Script 007: XGBoost Threshold Tuning
# ---------------------------------------------------------------------------
# Inputs: X_TEST_PROCESSED_PATH, Y_TEST_PATH, XGB_TUNED_MODEL_PATH (from 006)

# ---------------------------------------------------------------------------
# Script 008: LightGBM Optuna Tuning
# ---------------------------------------------------------------------------
# Inputs: X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH, X_TEST_PROCESSED_PATH, Y_TEST_PATH
BEST_LGBM_PARAMS_OPTUNA_PATH = RESULTS_DIR / "008_best_lgbm_params_optuna.joblib"
LGBM_OPTUNA_TUNED_MODEL_PATH = MODEL_DIR / "008_lgbm_optuna_tuned_model.joblib"

# ---------------------------------------------------------------------------
# Script 009: LightGBM Threshold Tuning
# ---------------------------------------------------------------------------
# Inputs: X_TEST_PROCESSED_PATH, Y_TEST_PATH, LGBM_OPTUNA_TUNED_MODEL_PATH (from 008)

# ---------------------------------------------------------------------------
# Column Mapping (from 002)
# ---------------------------------------------------------------------------
COLS_MAP = {
    "ת\"ז מותממת": "patient_id", "תאריך פתיחת בקשה": "request_date",
    "תאריך ביצוע ניתוח": "surgery_date", "מספר ניתוח תכנון": "plan_id",
    "חדר": "room", "הרדמה": "anesthesia", "קוד ניתוח": "procedure_code",
    "גיל המטופל בזמן הניתוח": "age", "BMI": "bmi",
    "תרופות קבועות": "medications", "אבחנות רקע": "diagnoses",
    "ALBUMIN": "albumin", "PT-INR": "pt_inr", "POTASSIUM": "potassium",
    "SODIUM": "sodium", "HB": "hb", "WBC": "wbc", "PLT": "plt",
    "עיר": "city", "אתר ניתוח": "surgery_site", "distance_km": "distance_km",
    "מין": "gender", "גורם משלם": "payer", "מצב משפחתי": "marital_status",
    "מחלקה מנתחת": "department"
}

# ---------------------------------------------------------------------------
# Ensure folders exist
# ---------------------------------------------------------------------------
for d in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Configuration Paths ---")
    print(f"PROJECT_ROOT: {PROJECT_ROOT.resolve()}")
    print(f"DATA_DIR    : {DATA_DIR.resolve()}")
    print(f"PLOT_DIR    : {PLOT_DIR.resolve()}")
    print(f"RESULTS_DIR : {RESULTS_DIR.resolve()}")
    print(f"MODEL_DIR   : {MODEL_DIR.resolve()}")
    print("-" * 20)
    print("Script 001 Input:")
    print(f"  RAW_DATA_XLSX : {RAW_DATA_XLSX.resolve()} (Exists? {RAW_DATA_XLSX.exists()})")
    print("-" * 20)
    print("Script 002 Output / 003 & 004 Input:")
    print(f"  ENGINEERED_DATA_XLSX : {ENGINEERED_DATA_XLSX.resolve()} (Exists? {ENGINEERED_DATA_XLSX.exists()})")
    print("-" * 20)
    print("Script 004 Outputs (Data for Models):")
    print(f"  PREPROCESSED_X_INTERMEDIATE_XLSX: {PREPROCESSED_X_INTERMEDIATE_XLSX.resolve()} (Exists? {PREPROCESSED_X_INTERMEDIATE_XLSX.exists()})")
    print(f"  X_TRAIN_PROCESSED_PATH : {X_TRAIN_PROCESSED_PATH.resolve()} (Exists? {X_TRAIN_PROCESSED_PATH.exists()})")
    print(f"  PREPROCESSOR_PIPELINE_PATH: {PREPROCESSOR_PIPELINE_PATH.resolve()} (Exists? {PREPROCESSOR_PIPELINE_PATH.exists()})")
    print("-" * 20)
    print("Script 005 Output (XGB Baseline):")
    print(f"  XGB_BASELINE_MODEL_PATH: {XGB_BASELINE_MODEL_PATH.resolve()} (Exists? {XGB_BASELINE_MODEL_PATH.exists()})")
    print("-" * 20)
    print("Script 006 Outputs (XGB Tuned):")
    print(f"  BEST_XGB_PARAMS_PATH : {BEST_XGB_PARAMS_PATH.resolve()} (Exists? {BEST_XGB_PARAMS_PATH.exists()})")
    print(f"  XGB_TUNED_MODEL_PATH : {XGB_TUNED_MODEL_PATH.resolve()} (Exists? {XGB_TUNED_MODEL_PATH.exists()})")
    print("-" * 20)
    print("Script 008 Outputs (LGBM Tuned):")
    print(f"  BEST_LGBM_PARAMS_OPTUNA_PATH : {BEST_LGBM_PARAMS_OPTUNA_PATH.resolve()} (Exists? {BEST_LGBM_PARAMS_OPTUNA_PATH.exists()})")
    print(f"  LGBM_OPTUNA_TUNED_MODEL_PATH : {LGBM_OPTUNA_TUNED_MODEL_PATH.resolve()} (Exists? {LGBM_OPTUNA_TUNED_MODEL_PATH.exists()})")
    print("-" * 20)
    print("COLS_MAP defined with keys:", list(COLS_MAP.keys()) if 'COLS_MAP' in globals() else "COLS_MAP not defined")