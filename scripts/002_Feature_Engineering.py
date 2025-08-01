# 002_Feature_Engineering.py
"""
Performs feature engineering for the surgery-cancellation dataset.
Outputs a focused set of features, including a custom 'age_bucket'.
Labels for 'wait_days_category' and 'distance_bucket' are modified to be numeric ranges.
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import holidays
import logging
# import re # Only if 'site_room' or similar regex cleaning is needed

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import RAW_DATA_XLSX, ENGINEERED_DATA_XLSX, COLS_MAP
    # Add other config imports like LMS_... if you decide to use them later
    print("Successfully imported paths from config.py for 002 (Focused Output)")
except ImportError:
    print("CRITICAL: Could not import 'config'. Ensure config.py is in the 'scripts' directory or key variables are missing.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: An unexpected error occurred during config import: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__) # Using standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger.info(f"--- Feature Engineering (Script 002 - Focused Output with Age Bucket) ---")
logger.info(f"Using raw data input: {RAW_DATA_XLSX}")
logger.info(f"Engineered data output will be: {ENGINEERED_DATA_XLSX}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISTANCE_BINS = [0, 5, 10, 20, 50, 100, 200, 500, np.inf]
DISTANCE_LABELS_MODIFIED = ["0-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200-500", "500+"]

WAIT_DAYS_BINS = [-np.inf, 7, 30, 90, 365, np.inf]
WAIT_DAYS_LABELS_MODIFIED = ["0-7", "8-30", "31-90", "91-365", "365+"]

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_START_YEAR = 2018 
DATA_END_YEAR = 2025   
try:
    il_holidays_generator = holidays.IL(years=range(DATA_START_YEAR, DATA_END_YEAR + 1))
    HOLIDAYS_IL_DYNAMIC = set(pd.to_datetime(list(il_holidays_generator.keys())).date)
    logger.info(f"Generated {len(HOLIDAYS_IL_DYNAMIC)} Israeli holiday dates for years {DATA_START_YEAR}-{DATA_END_YEAR}")
except Exception as e:
    logger.error(f"Error creating dynamic holiday list: {e}. Proceeding with an empty list.")
    HOLIDAYS_IL_DYNAMIC = set()

# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------
def main() -> None:
    if not RAW_DATA_XLSX.exists():
        logger.error(f"Input Excel file not found: {RAW_DATA_XLSX}")
        raise FileNotFoundError(f"Input Excel file not found: {RAW_DATA_XLSX}")

    logger.info("Loading data...")
    xls = pd.ExcelFile(RAW_DATA_XLSX)
    df_all = pd.read_excel(xls, sheet_name=0) 
    # Apply COLS_MAP for renaming Hebrew columns to English if they exist
    # Filter COLS_MAP to only include columns present in df_all to avoid KeyErrors
    rename_map = {k: v for k, v in COLS_MAP.items() if k in df_all.columns}
    df_all.rename(columns=rename_map, inplace=True)
    
    df_cancel = pd.read_excel(xls, sheet_name=1)
    logger.info(f"Loaded {len(df_all):,} records from the main sheet.")
    logger.info(f"Loaded {len(df_cancel):,} records from the cancellation sheet.")

    # Ensure 'plan_id' exists in df_all (might be named 'מספר ניתוח תכנון' originally)
    if 'plan_id' not in df_all.columns and "מספר ניתוח תכנון" in df_all.columns:
        df_all.rename(columns={"מספר ניתוח תכנון": "plan_id"}, inplace=True)
        logger.info("Renamed 'מספר ניתוח תכנון' to 'plan_id' in df_all.")
    elif 'plan_id' not in df_all.columns:
        logger.error("'plan_id' (or 'מספר ניתוח תכנון') not found in df_all after attempting to use COLS_MAP.")
        raise KeyError("'plan_id' not found in df_all.")

    # --- 1. Create 'was_canceled' ---
    logger.info("Calculating 'was_canceled'...")
    original_cancel_id_col = "מספר ניתוח תכנון" # Name in df_cancel
    if original_cancel_id_col not in df_cancel.columns:
         logger.error(f"'{original_cancel_id_col}' not found in df_cancel.")
         raise KeyError(f"'{original_cancel_id_col}' not found in df_cancel.")

    df_all["plan_id"] = df_all["plan_id"].astype(str).str.strip()
    df_cancel[original_cancel_id_col] = df_cancel[original_cancel_id_col].astype(str).str.strip()
    
    plans_cancel_set = set(df_cancel[original_cancel_id_col])
    df_all["was_canceled"] = df_all["plan_id"].isin(plans_cancel_set)
    logger.info(f"Marked {df_all['was_canceled'].sum():,} records as canceled.")

    # --- 2. Process dates and create base calendar features ---
    logger.info("Processing dates and calendar features...")
    # Ensure date columns have English names (handled by COLS_MAP or direct rename if needed)
    date_cols_map = {"request_date": "תאריך פתיחת בקשה", "surgery_date": "תאריך ביצוע ניתוח"}
    for eng_name, heb_name in date_cols_map.items():
        if eng_name not in df_all.columns and heb_name in df_all.columns:
            df_all.rename(columns={heb_name: eng_name}, inplace=True)
            logger.info(f"Renamed '{heb_name}' to '{eng_name}'.")

    df_all["request_date"] = pd.to_datetime(df_all.get("request_date"), errors="coerce")
    df_all["surgery_date"] = pd.to_datetime(df_all.get("surgery_date"), errors="coerce")

    # --- 3. Create 'wait_days' and 'wait_days_category' ---
    if 'request_date' in df_all.columns and 'surgery_date' in df_all.columns and \
       df_all["request_date"].notna().any() and df_all["surgery_date"].notna().any():
        df_all["wait_days"] = (df_all["surgery_date"] - df_all["request_date"]).dt.days
        df_all["wait_days_category"] = pd.cut(df_all["wait_days"], bins=WAIT_DAYS_BINS, labels=WAIT_DAYS_LABELS_MODIFIED, right=False).astype(object)
        df_all["wait_days_category"].fillna("__MISSING__", inplace=True)
        logger.info(f"Calculated 'wait_days' and 'wait_days_category' (modified labels).")
    else:
        logger.warning("Could not calculate 'wait_days'. 'request_date' or 'surgery_date' missing or all NaNs.")
        df_all["wait_days"] = np.nan
        df_all["wait_days_category"] = "__MISSING__"

    # --- 4. Create 'surgery_weekday', 'is_weekend', 'season' ---
    valid_dates_mask = df_all["surgery_date"].notna() if "surgery_date" in df_all.columns else pd.Series([False]*len(df_all))
    
    df_all["surgery_weekday"] = pd.NA
    if "surgery_date" in df_all.columns:
        df_all.loc[valid_dates_mask, "surgery_weekday"] = df_all.loc[valid_dates_mask, "surgery_date"].dt.day_name()
    
    df_all["is_weekend"] = pd.NA
    if "surgery_weekday" in df_all.columns and df_all["surgery_weekday"].notna().any():
        df_all["is_weekend"] = df_all["surgery_weekday"].isin(["Friday", "Saturday"])
    
    month_to_season = {1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer",
                       7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall", 12:"Winter"}
    df_all["season"] = pd.NA
    if "surgery_date" in df_all.columns:
        df_all.loc[valid_dates_mask, "season"] = df_all.loc[valid_dates_mask, "surgery_date"].dt.month.map(month_to_season)
    
    # --- 5. Create 'near_holiday' ---
    logger.info("Calculating 'near_holiday'...")
    df_all["near_holiday"] = False 
    if HOLIDAYS_IL_DYNAMIC and "surgery_date" in df_all.columns:
        surg_dates_series = df_all.loc[valid_dates_mask, "surgery_date"].dt.date
        for index, row_date in surg_dates_series.items():
            if pd.notna(row_date):
                 df_all.loc[index, "near_holiday"] = any(abs((row_date - hol_date).days) <= 3 for hol_date in HOLIDAYS_IL_DYNAMIC)

    # --- 6. Create 'num_medications', 'num_diagnoses' ---
    logger.info("Creating count features for medications and diagnoses...")
    def count_items(text_series):
        if text_series is None or not isinstance(text_series, pd.Series):
            return pd.Series([0] * len(df_all), index=df_all.index)
        return text_series.fillna('').astype(str).apply(lambda x: len(x.split(',')) if x.strip() else 0)

    # Ensure 'medications' and 'diagnoses' columns exist (possibly renamed by COLS_MAP)
    med_col_name = "medications" if "medications" in df_all.columns else ("תרופות קבועות" if "תרופות קבועות" in df_all.columns else None)
    diag_col_name = "diagnoses" if "diagnoses" in df_all.columns else ("אבחנות רקע" if "אבחנות רקע" in df_all.columns else None)

    df_all["num_medications"] = count_items(df_all.get(med_col_name))
    df_all["num_diagnoses"] = count_items(df_all.get(diag_col_name))

    # --- 7. Create 'distance_bucket' ---
    logger.info("Creating 'distance_bucket'...")
    if 'distance_km' in df_all.columns:
        df_all['distance_km'] = pd.to_numeric(df_all['distance_km'], errors='coerce')
        df_all["distance_bucket"] = pd.cut(df_all["distance_km"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS_MODIFIED, right=False, include_lowest=True).astype(object)
        df_all["distance_bucket"].fillna("__MISSING__", inplace=True)
        logger.info(f"Calculated 'distance_bucket' (modified labels).")
    else:
        logger.warning("'distance_km' column not found. Cannot create 'distance_bucket'.")
        df_all["distance_bucket"] = "__MISSING__"

    # --- 8. Create 'age_bucket' (NEW) ---
    if 'age' in df_all.columns:
        logger.info("Creating 'age_bucket' column...")
        if not pd.api.types.is_numeric_dtype(df_all['age']):
            df_all['age'] = pd.to_numeric(df_all['age'], errors='coerce')
            logger.info("Converted 'age' column to numeric for bucketing.")
        
        def assign_age_bucket(age_value):
            if pd.isna(age_value): return np.nan 
            if age_value <= 12: return "0-12"
            elif age_value <= 18: return "12-18"
            elif age_value <= 40: return "18-40"
            elif age_value <= 65: return "40-65"
            elif age_value <= 80: return "65-80"
            else: return "80+"

        df_all["age_bucket"] = df_all['age'].apply(assign_age_bucket).astype(object)
        df_all["age_bucket"].fillna("__MISSING__", inplace=True) # Handle NaNs resulting from NaN age
        logger.info("Created 'age_bucket' with requested ranges.")
    else:
        logger.warning("'age' column not found. Cannot create 'age_bucket'.")
        df_all["age_bucket"] = "__MISSING__"


    # --- 9. Check for surgeries on Saturday ---
    if "surgery_weekday" in df_all.columns:
        surgeries_on_saturday = df_all[df_all["surgery_weekday"] == "Saturday"]
        if not surgeries_on_saturday.empty:
            logger.warning(f"Found {len(surgeries_on_saturday)} surgeries scheduled on Saturday.")
        else:
            logger.info("Check: No surgeries found scheduled on Saturday.")
    
    # --- 10. Select final columns for output ---
    # These are the columns you specified you want in the output of 002
    # Plus any base columns needed for them (e.g. 'age' for 'age_bucket' if 'age' itself is not requested)
    # And essential identifiers like 'plan_id' if needed for merging in 004 before dropping
    requested_original_cols = [
        'department', 'surgery_site', 'room', 'procedure_code', 'anesthesia',
        'age', 'gender', 'city', 'payer', 'marital_status', 'distance_km',
        'plan_id' # Keep plan_id for potential use before 004 drops it
    ]
    
    created_cols = [
        'was_canceled', 'wait_days', 'wait_days_category', 'surgery_weekday',
        'is_weekend', 'season', 'near_holiday', 'num_medications',
        'num_diagnoses', 'distance_bucket', 'age_bucket'
    ]

    final_output_columns = []
    # Ensure original columns exist (after COLS_MAP and any direct renames)
    for col_name in requested_original_cols:
        if col_name in df_all.columns:
            final_output_columns.append(col_name)
        else: # Try to find its Hebrew original if not mapped by COLS_MAP
            original_heb_name = None
            for k_heb, v_eng in COLS_MAP.items():
                if v_eng == col_name:
                    original_heb_name = k_heb
                    break
            if original_heb_name and original_heb_name in df_all.columns:
                df_all.rename(columns={original_heb_name: col_name}, inplace=True)
                final_output_columns.append(col_name)
                logger.info(f"Renamed '{original_heb_name}' to '{col_name}' for output.")
            else:
                logger.warning(f"Requested original column '{col_name}' not found. It will be missing.")
    
    # Add created columns
    for col_name in created_cols:
        if col_name in df_all.columns:
            final_output_columns.append(col_name)
        else:
            logger.warning(f"Created column '{col_name}' is unexpectedly missing. It will be missing.")
            
    # Remove duplicates and sort for consistency (optional)
    final_output_columns = sorted(list(set(final_output_columns)))
    
    # Ensure target column is present if it's in final_output_columns, or add it if not but was created
    if 'was_canceled' not in final_output_columns and 'was_canceled' in df_all.columns:
        final_output_columns.append('was_canceled')


    df_output = df_all[final_output_columns].copy()
    logger.info(f"Selected final columns for output ({len(df_output.columns)}): {df_output.columns.tolist()}")

    # --- 11. Save the engineered file ---
    logger.info("\nPreview of some features from the final selected DataFrame (first 5 rows):")
    preview_cols_subset = [col for col in ['plan_id', 'was_canceled', 'age_bucket', 'wait_days_category', 'distance_bucket'] if col in df_output.columns]
    if not preview_cols_subset and df_output.shape[1] > 0:
        preview_cols_subset = df_output.columns.tolist()[:5]
    if preview_cols_subset:
        # Convert to string for markdown if mixed types cause issues
        # logger.info(df_output[preview_cols_subset].astype(str).head().to_markdown(index=False, numalign="left", stralign="left"))
        logger.info("\n" + df_output[preview_cols_subset].head().to_string())

    else:
        logger.info("No columns available for preview in df_output.")

    choice = input("\n💬 Save the engineered file (with selected features & age_bucket) to Excel? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            ENGINEERED_DATA_XLSX.parent.mkdir(parents=True, exist_ok=True)
            output_sheet_name = "features_focused_v1" # Or your preferred sheet name
            logger.info(f"Saving file to: {ENGINEERED_DATA_XLSX} (sheet: {output_sheet_name})...")
            
            # Convert object columns that might contain mixed types (like np.nan and strings) to string for safer Excel export
            for col in df_output.select_dtypes(include=['object']).columns:
                # Check if column is not purely string already
                if not all(isinstance(x, str) for x in df_output[col].dropna()):
                     df_output[col] = df_output[col].astype(str).replace({'<NA>':'', 'nan':''}) # Replace pandas NA and string 'nan' with empty string

            df_output.to_excel(ENGINEERED_DATA_XLSX, sheet_name=output_sheet_name, index=False, engine='openpyxl')
            logger.info(f"✅ Successfully saved → {ENGINEERED_DATA_XLSX.resolve()}")
        except Exception as e:
            logger.error(f"❌ Error saving file: {e}")
    else:
        logger.info("❎ File was not saved.")

if __name__ == "__main__":
    main()