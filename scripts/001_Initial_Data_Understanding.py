# 001_Initial_Data_Understanding.py
"""
Loads the raw surgery data and performs comprehensive initial statistical analysis,
including a deeper look into potential duplicates in the cancellation log based on dates.
Distinguishes between exact duplicate rows and multiple cancellation events for the same Plan ID.
Analyzes the characteristics of multiple cancellation events.
Outputs results to both console and a timestamped text file in the results directory.

Run from the project root:
    python -m scripts.001_Initial_Data_Understanding
or from *scripts/* directly:
    python 001_Initial_Data_Understanding.py
"""
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import INPUT_XLSX, RESULTS_DIR
except ImportError:
    print("Warning: config.py not found or required paths (INPUT_XLSX, RESULTS_DIR) not defined.")
    print("Using default relative paths assuming script is in 'scripts' folder.")
    scripts_dir = Path(__file__).resolve().parent
    project_root_alt = scripts_dir.parent
    INPUT_XLSX = project_root_alt / "data" / "Surgery_Data.xlsx"
    RESULTS_DIR = project_root_alt / "results"
    (project_root_alt / "data").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup Output Logging to File and Console
# ---------------------------------------------------------------------------
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".log") -> Path:
    """Finds the next available filename like prefix_1.log, prefix_2.log etc."""
    base_name = prefix
    counter = 0
    while True:
        if counter == 0:
            file_path = base_dir / f"{base_name}{suffix}"
        else:
            file_path = base_dir / f"{base_name}_{counter}{suffix}"
        if not file_path.exists():
            return file_path
        counter += 1

log_filename_base = Path(__file__).stem
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")

log_formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()

# --- Prevent handler accumulation in interactive environments ---
# Clear existing handlers attached to the root logger
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# ----------------------------------------------------------------

logger.setLevel(logging.INFO) # Set level AFTER clearing handlers if needed

# File Handler
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# --- Start of Analysis ---
logger.info(f"--- Initial Data Understanding ---")
logger.info(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Using input file: {INPUT_XLSX}")
if not INPUT_XLSX.exists():
    logger.error(f"Input Excel file not found at: {INPUT_XLSX.resolve()}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Function to capture DataFrame info into a string
# ---------------------------------------------------------------------------
def get_df_info(df: pd.DataFrame) -> str:
    """Captures df.info() output into a string."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# Load Data (Raw)
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Loading Raw Data ===")
try:
    xls = pd.ExcelFile(INPUT_XLSX)
    df_all_raw = pd.read_excel(xls, sheet_name=0)
    logger.info(f"Loaded df_all_raw: {df_all_raw.shape[0]:,} rows, {df_all_raw.shape[1]} columns")
    df_cancel_raw = pd.read_excel(xls, sheet_name=1)
    logger.info(f"Loaded df_cancel_raw: {df_cancel_raw.shape[0]:,} rows, {df_cancel_raw.shape[1]} columns")
except Exception as e:
    logger.error(f"Error loading Excel file: {e}")
    sys.exit(1)

# Define key column names for cancellation analysis
cancel_id_col_raw = "מספר ניתוח תכנון"
cancel_date_col_raw = "תאריך ביטול ניתוח"
planned_date_at_cancel_col = "תאריך הניתוח המתוכנן בזמן הביטול"

# Validate existence of key columns in df_cancel_raw
required_cancel_cols = [cancel_id_col_raw, cancel_date_col_raw, planned_date_at_cancel_col]
missing_cancel_cols = [col for col in required_cancel_cols if col not in df_cancel_raw.columns]
if missing_cancel_cols:
    logger.error(f"Error: Missing required columns in cancellation sheet (df_cancel_raw): {missing_cancel_cols}")
    logger.error(f"Available columns: {df_cancel_raw.columns.tolist()}")
    sys.exit(1)

# Convert date columns in cancellation sheet to datetime
try:
    df_cancel_raw[cancel_date_col_raw] = pd.to_datetime(df_cancel_raw[cancel_date_col_raw], errors='coerce')
    df_cancel_raw[planned_date_at_cancel_col] = pd.to_datetime(df_cancel_raw[planned_date_at_cancel_col], errors='coerce')
    logger.info(f"Converted '{cancel_date_col_raw}' and '{planned_date_at_cancel_col}' in df_cancel_raw to datetime.")
except Exception as e:
    logger.error(f"Error converting date columns in df_cancel_raw: {e}")
    sys.exit(1)

# Clean Plan IDs (assuming they should be treated as strings)
df_cancel_raw[cancel_id_col_raw] = df_cancel_raw[cancel_id_col_raw].astype(str).str.strip()
plan_id_col_all = "מספר ניתוח תכנון" # Assuming this is the name in df_all_raw as well
if plan_id_col_all not in df_all_raw.columns:
    logger.error(f"Error: Expected plan ID column '{plan_id_col_all}' not found in df_all_raw!")
    logger.error(f"Available columns: {df_all_raw.columns.tolist()}")
    sys.exit(1)
df_all_raw[plan_id_col_all] = df_all_raw[plan_id_col_all].astype(str).str.strip()


# ---------------------------------------------------------------------------
# Basic Structure and Info (Summary)
# ---------------------------------------------------------------------------
logger.info("\n=== 2. Basic Structure & Info (Summary) ===")
logger.info(f"df_all_raw Shape: {df_all_raw.shape}")
logger.info(f"df_cancel_raw Shape: {df_cancel_raw.shape}")

# ---------------------------------------------------------------------------
# Cancellation Log Analysis (Consistency & Duplicates - REVISED LOGIC)
# ---------------------------------------------------------------------------
logger.info("\n=== 3. Cancellation Log Analysis (Consistency & Duplicates) ===")

# --- 3a. Basic Counts ---
logger.info("\n--- 3a. Basic Counts ---")
total_cancel_rows = len(df_cancel_raw)
unique_canceled_ids_count = df_cancel_raw[cancel_id_col_raw].nunique()
logger.info(f"Total rows in df_cancel_raw: {total_cancel_rows:,}")
logger.info(f"Unique '{cancel_id_col_raw}' in df_cancel_raw: {unique_canceled_ids_count:,}")

# --- 3b. Identify EXACT Duplicate Rows ---
logger.info("\n--- 3b. Exact Duplicate Row Analysis ---")
exact_duplicate_rows = df_cancel_raw[df_cancel_raw.duplicated(keep=False)]
num_exact_duplicate_rows = len(exact_duplicate_rows)
num_unique_rows = len(df_cancel_raw.drop_duplicates())
num_rows_removed_if_deduped = total_cancel_rows - num_unique_rows

logger.info(f"Number of rows identified as EXACT duplicates (identical across all columns): {num_rows_removed_if_deduped:,}")
if num_rows_removed_if_deduped > 0:
    logger.info("These likely represent true data entry errors.")
    logger.info("\nExamples of exact duplicate rows (showing pairs/groups):")
    if num_exact_duplicate_rows > 0:
         first_few_exact_duplicates = exact_duplicate_rows.sort_values(by=df_cancel_raw.columns.tolist()).head(10)
         logger.info(first_few_exact_duplicates.to_markdown(index=False))
    else:
         logger.info("No exact duplicate rows found to show examples.")

# --- 3c. Identify Multiple Cancellation EVENTS (Same Plan ID, Different Data) ---
logger.info("\n--- 3c. Multiple Cancellation Event Analysis (Same Plan ID, Different Row Data) ---")
df_cancel_unique_rows = df_cancel_raw.drop_duplicates()
plan_id_counts_after_exact_dedup = df_cancel_unique_rows.groupby(cancel_id_col_raw).size()
plan_ids_with_multiple_events = plan_id_counts_after_exact_dedup[plan_id_counts_after_exact_dedup > 1].index
num_plan_ids_with_multiple_events = len(plan_ids_with_multiple_events)
rows_in_multiple_events = df_cancel_unique_rows[df_cancel_unique_rows[cancel_id_col_raw].isin(plan_ids_with_multiple_events)]
num_rows_in_multiple_events = len(rows_in_multiple_events)

logger.info(f"Number of unique '{cancel_id_col_raw}' associated with multiple distinct cancellation/change events: {num_plan_ids_with_multiple_events:,}")
logger.info(f"Total rows involved in these multiple distinct events: {num_rows_in_multiple_events:,}")
if num_plan_ids_with_multiple_events > 0:
    logger.info("These represent Plan IDs where the cancellation log captured different information (e.g., different cancel date, planned date) over time.")
    logger.info("\nExamples of multiple distinct events for the same Plan ID:")
    example_ids_multi = plan_ids_with_multiple_events[:3].tolist()
    logger.info(f"(Showing details for Plan IDs: {example_ids_multi})")
    example_df_multi = rows_in_multiple_events[rows_in_multiple_events[cancel_id_col_raw].isin(example_ids_multi)].sort_values(by=[cancel_id_col_raw, cancel_date_col_raw])
    if not example_df_multi.empty:
        logger.info(example_df_multi.to_markdown(index=False))
    else:
        logger.info("Could not retrieve example multiple event entries.")

# --- 3d. Matching with Main Dataset (Using Unique Plan IDs from Cancel Log) ---
logger.info("\n--- 3d. Matching with Main Dataset (df_all_raw) ---")
canceled_ids_raw_set = set(df_cancel_raw[cancel_id_col_raw].unique())
all_plan_ids_series = df_all_raw[plan_id_col_all]
df_all_raw['was_canceled_accurate'] = all_plan_ids_series.isin(canceled_ids_raw_set)
num_marked_as_canceled = df_all_raw['was_canceled_accurate'].sum()
logger.info(f"Rows marked as canceled in df_all_raw (based on *any* entry in cancel log): {num_marked_as_canceled:,}")

all_plan_ids_set = set(all_plan_ids_series.unique())
canceled_ids_found_in_all = canceled_ids_raw_set.intersection(all_plan_ids_set)
num_canceled_ids_found = len(canceled_ids_found_in_all)
logger.info(f"Unique canceled Plan IDs from df_cancel_raw found in df_all_raw: {num_canceled_ids_found:,}")
num_missing_in_all = unique_canceled_ids_count - num_canceled_ids_found
logger.info(f"-> {num_missing_in_all:,} unique canceled Plan IDs from df_cancel_raw are NOT PRESENT in df_all_raw.")


# --- Sections 4-9 remain the same as the previous correct version ---
# ---------------------------------------------------------------------------
# Missing Value Analysis (Section 4)
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Missing Value Analysis (df_all_raw) ===")
missing_percentage = (df_all_raw.isnull().sum() / len(df_all_raw) * 100).round(2)
missing_info = pd.DataFrame({'Missing Count': df_all_raw.isnull().sum(), 'Missing Percentage (%)': missing_percentage})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Percentage (%)', ascending=False)
if not missing_info.empty:
    logger.info(missing_info.to_string())
else:
    logger.info("No missing values found in df_all_raw.")

# ---------------------------------------------------------------------------
# Target Variable Analysis (Section 5)
# ---------------------------------------------------------------------------
logger.info("\n=== 5. Target Variable Analysis ('was_canceled_accurate') ===")
if 'was_canceled_accurate' in df_all_raw.columns:
    cancel_rate = df_all_raw['was_canceled_accurate'].mean() * 100
    logger.info(f"Overall Accurate Cancellation Rate: {cancel_rate:.2f}%")
    logger.info("\nDistribution of Target Variable:")
    logger.info(df_all_raw['was_canceled_accurate'].value_counts(normalize=True).round(4).to_string())
    logger.info("\nCounts:")
    logger.info(df_all_raw['was_canceled_accurate'].value_counts().to_string())
    imbalance_ratio = df_all_raw['was_canceled_accurate'].value_counts().min() / df_all_raw['was_canceled_accurate'].value_counts().max() if df_all_raw['was_canceled_accurate'].value_counts().max() > 0 else 0
    logger.info(f"(Dataset Imbalance Ratio: ~{imbalance_ratio:.2f})")
else:
     logger.error("Column 'was_canceled_accurate' was not created successfully.")


# ---------------------------------------------------------------------------
# Numerical Feature Analysis (Section 6)
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Numerical Feature Analysis (df_all_raw) ===")
numeric_cols_guess = ['גיל המטופל בזמן הניתוח', 'BMI', 'ALBUMIN', 'PT-INR', 'POTASSIUM', 'SODIUM', 'HB', 'WBC', 'PLT', 'ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח']
if 'distance_km' in df_all_raw.columns:
    numeric_cols_guess.append('distance_km')
    try: df_all_raw['distance_km'] = pd.to_numeric(df_all_raw['distance_km'], errors='coerce')
    except Exception as e: logger.warning(f"Could not convert 'distance_km' to numeric: {e}")

numeric_cols_exist = [col for col in numeric_cols_guess if col in df_all_raw.columns]
for col in numeric_cols_exist:
    if pd.api.types.is_numeric_dtype(df_all_raw[col]) or pd.api.types.is_datetime64_any_dtype(df_all_raw[col]): continue
    original_dtype = df_all_raw[col].dtype
    try:
        df_all_raw[col] = pd.to_numeric(df_all_raw[col], errors='coerce')
        if df_all_raw[col].isnull().all(): logger.warning(f"Column '{col}' became all NaNs after numeric conversion from {original_dtype}.")
    except Exception as e: logger.warning(f"Could not convert '{col}' (dtype: {original_dtype}) to numeric: {e}")

numeric_cols_final = df_all_raw[numeric_cols_exist].select_dtypes(include=np.number).columns.tolist()
logger.info(f"Analyzing numerical columns: {numeric_cols_final}")
if numeric_cols_final:
    logger.info("\nDescriptive Statistics (Overall):")
    logger.info(df_all_raw[numeric_cols_final].describe().round(2).to_string())
    if 'was_canceled_accurate' in df_all_raw.columns:
        logger.info("\nDescriptive Statistics (Grouped by 'was_canceled_accurate'):")
        try:
            grouped_desc = df_all_raw.groupby('was_canceled_accurate')[numeric_cols_final].describe().round(2)
            logger.info(grouped_desc.to_string())
        except Exception as e: logger.error(f"Error generating grouped descriptives for numerical: {e}")
    else: logger.warning("Grouping skipped: 'was_canceled_accurate' column not found.")
else: logger.info("No numerical columns available for analysis after type conversion.")


# ---------------------------------------------------------------------------
# Date Feature Analysis (Section 7)
# ---------------------------------------------------------------------------
logger.info("\n=== 7. Date Feature Analysis (df_all_raw) ===")
date_cols = ['תאריך פתיחת בקשה', 'תאריך ביצוע ניתוח']
existing_date_cols = [col for col in date_cols if col in df_all_raw.columns]
for col in existing_date_cols:
    if not pd.api.types.is_datetime64_any_dtype(df_all_raw[col]):
        original_dtype = df_all_raw[col].dtype
        try:
            df_all_raw[col] = pd.to_datetime(df_all_raw[col], errors='coerce')
            logger.info(f"Converted '{col}' to datetime from {original_dtype} (errors coerced).")
        except Exception as e: logger.warning(f"Could not convert '{col}' to datetime: {e}")

if len(existing_date_cols) == 2 and \
   pd.api.types.is_datetime64_any_dtype(df_all_raw[existing_date_cols[0]]) and \
   pd.api.types.is_datetime64_any_dtype(df_all_raw[existing_date_cols[1]]):
    req_date_col = existing_date_cols[0]
    surg_date_col = existing_date_cols[1]
    logger.info(f"\nDate Range ({req_date_col}): {df_all_raw[req_date_col].min()} to {df_all_raw[req_date_col].max()}")
    logger.info(f"Date Range ({surg_date_col}): {df_all_raw[surg_date_col].min()} to {df_all_raw[surg_date_col].max()}")
    wait_days_col_orig = 'ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח'
    if wait_days_col_orig not in numeric_cols_final:
        logger.info("Calculating 'wait_days' from date columns.")
        df_all_raw['wait_days_calc'] = (df_all_raw[surg_date_col] - df_all_raw[req_date_col]).dt.days
        wait_days_col_to_analyze = 'wait_days_calc'
    else:
        logger.info(f"Using existing column '{wait_days_col_orig}' for wait time analysis.")
        wait_days_col_to_analyze = wait_days_col_orig
    if wait_days_col_to_analyze in df_all_raw.columns:
        logger.info("\nWait Time (Days from Request to Surgery) Statistics:")
        logger.info(df_all_raw[wait_days_col_to_analyze].describe().round(1).to_string())
        if 'was_canceled_accurate' in df_all_raw.columns:
             logger.info("\nWait Time Statistics (Grouped by 'was_canceled_accurate'):")
             try: logger.info(df_all_raw.groupby('was_canceled_accurate')[wait_days_col_to_analyze].describe().round(1).to_string())
             except Exception as e: logger.error(f"Error generating grouped descriptives for wait time: {e}")
        else: logger.warning("Grouping skipped: 'was_canceled_accurate' column not found.")
    try:
        df_all_raw['surgery_weekday'] = df_all_raw[surg_date_col].dt.day_name()
        month_to_season = {1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer",
                           7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall", 12:"Winter"}
        df_all_raw['season'] = df_all_raw[surg_date_col].dt.month.map(month_to_season)
        logger.info("\nGenerated basic calendar features: 'surgery_weekday', 'season'")
    except Exception as e: logger.warning(f"Could not generate calendar features: {e}")
else: logger.info("Could not find both required date columns or they are not datetime type for analysis.")

# ---------------------------------------------------------------------------
# Categorical Feature Analysis (Section 8)
# ---------------------------------------------------------------------------
logger.info("\n=== 8. Categorical Feature Analysis (df_all_raw) ===")
cat_cols_guess = ['מחלקה מנתחת', 'אתר ניתוח', 'חדר', 'הרדמה', 'קוד ניתוח', 'מין', 'עיר', 'גורם משלם', 'מצב משפחתי', 'surgery_weekday', 'season']
cat_cols_exist = [col for col in cat_cols_guess if col in df_all_raw.columns]
cat_cols_final = df_all_raw[cat_cols_exist].select_dtypes(exclude=np.number).columns.tolist()
if cat_cols_final:
    logger.info(f"Analyzing categorical columns: {cat_cols_final}")
    for col in cat_cols_final:
        logger.info(f"\n--- Analysis for: {col} ---")
        try:
            col_series = df_all_raw[col].fillna('__MISSING__')
            num_unique = col_series.nunique()
            logger.info(f"Number of Unique Values: {num_unique}")
            if num_unique > 50:
                 logger.info(f"Top 20 Most Frequent Values:")
                 logger.info(col_series.value_counts().head(20).to_string())
            else:
                logger.info("Value Counts:")
                logger.info(col_series.value_counts().to_string())
            if 'was_canceled_accurate' in df_all_raw.columns:
                logger.info(f"\nCancellation Rate by {col}:")
                cancel_rate_by_cat = df_all_raw.groupby(col, dropna=False)['was_canceled_accurate'].agg(['mean', 'count']).round(3)
                cancel_rate_by_cat = cancel_rate_by_cat.rename(columns={'mean': 'Cancel Rate', 'count': 'Total Count'})
                cancel_rate_by_cat = cancel_rate_by_cat.sort_values(['Total Count', 'Cancel Rate'], ascending=[False, False])
                if num_unique > 50:
                    logger.info(cancel_rate_by_cat.head(20).to_string())
                    logger.info("... (showing top 20 by count)")
                else:
                    logger.info(cancel_rate_by_cat.to_string())
            else: logger.warning("'was_canceled_accurate' column not found, cannot calculate rates.")
        except Exception as e: logger.error(f"Error analyzing categorical column '{col}': {e}")
else: logger.info("No categorical columns identified or available for analysis.")


# ---------------------------------------------------------------------------
# Text Feature Analysis (Section 9)
# ---------------------------------------------------------------------------
logger.info("\n=== 9. Text Feature Analysis (Counts) ===")
text_cols = ['תרופות קבועות', 'אבחנות רקע']
existing_text_cols = [col for col in text_cols if col in df_all_raw.columns]
if existing_text_cols:
    logger.info(f"Analyzing text complexity (counts) for: {existing_text_cols}")
    def count_items(text_series):
        return text_series.fillna('').astype(str).apply(lambda x: len(x.split(',')) if x.strip() else 0)
    for col in existing_text_cols:
        count_col_name = f"num_{col.replace(' ', '_')}"
        try:
            df_all_raw[count_col_name] = count_items(df_all_raw[col])
            logger.info(f"\n--- Statistics for {count_col_name} ---")
            logger.info(f"(Calculated by splitting '{col}' by comma)")
            logger.info("\nOverall Statistics:")
            logger.info(df_all_raw[count_col_name].describe().round(1).to_string())
            if 'was_canceled_accurate' in df_all_raw.columns:
                 logger.info("\nStatistics Grouped by 'was_canceled_accurate':")
                 logger.info(df_all_raw.groupby('was_canceled_accurate')[count_col_name].describe().round(1).to_string())
            else: logger.warning("'was_canceled_accurate' column not found, cannot group.")
            logger.info(f"\nExample raw text content for '{col}' (first 5 non-empty):")
            example_series = df_all_raw.loc[df_all_raw[col].fillna('').str.strip() != '', col]
            logger.info(example_series.head().to_string(index=False))
        except Exception as e: logger.error(f"Error processing text column '{col}': {e}")
else: logger.info("Text columns ('תרופות קבועות', 'אבחנות רקע') not found.")

# ---------------------------------------------------------------------------
# Analysis of Multiple Cancellation Events (Section 10)
# ---------------------------------------------------------------------------
logger.info("\n=== 10. Analysis of Multiple Cancellation Events ===")
if 'plan_ids_with_multiple_events' in locals() and num_plan_ids_with_multiple_events > 0: # Check if variable exists
    logger.info(f"Analyzing the {num_plan_ids_with_multiple_events:,} unique Plan IDs identified with multiple distinct cancellation events.")

    # Use the dataframe containing only these rows (after removing exact duplicates)
    # Need df_cancel_unique_rows defined earlier
    if 'df_cancel_unique_rows' not in locals():
         df_cancel_unique_rows = df_cancel_raw.drop_duplicates()

    df_multi_events = df_cancel_unique_rows[df_cancel_unique_rows[cancel_id_col_raw].isin(plan_ids_with_multiple_events)].copy()

    if not df_multi_events.empty:
        # Calculate time differences within each group
        df_multi_events.sort_values(by=[cancel_id_col_raw, cancel_date_col_raw], inplace=True)
        # Time between consecutive cancellation dates for the same Plan ID
        df_multi_events['days_since_last_cancel'] = df_multi_events.groupby(cancel_id_col_raw)[cancel_date_col_raw].diff().dt.days
        # Time difference between planned date and cancel date for each event
        df_multi_events['days_planned_to_cancel'] = (df_multi_events[planned_date_at_cancel_col] - df_multi_events[cancel_date_col_raw]).dt.days
        # Count the number of events per Plan ID
        df_multi_events['cancel_event_count'] = df_multi_events.groupby(cancel_id_col_raw)[cancel_date_col_raw].cumcount() + 1 # Use a consistent column for cumcount

        logger.info("\n--- Statistics on Timing of Multiple Events ---")
        logger.info("Days Since Last Cancellation (for the same Plan ID):")
        logger.info(df_multi_events['days_since_last_cancel'].describe().round(1).to_string())
        logger.info("\nDays from Cancellation Date to Planned Surgery Date:")
        logger.info(df_multi_events['days_planned_to_cancel'].describe().round(1).to_string())
        logger.info("\nNumber of Cancellation Events per Plan ID (for those with >1 event):")
        logger.info(df_multi_events.groupby(cancel_id_col_raw)['cancel_event_count'].max().describe().round(1).to_string())

        # Example characteristics of patients/surgeries with multiple events
        logger.info("\n--- Characteristics of Patients/Surgeries with Multiple Events ---")
        multi_event_plan_ids = df_multi_events[cancel_id_col_raw].unique()
        df_all_multi_event_subset = df_all_raw[df_all_raw[plan_id_col_all].isin(multi_event_plan_ids)]

        if not df_all_multi_event_subset.empty:
            logger.info(f"Found {len(df_all_multi_event_subset):,} matching entries in df_all_raw for the {len(multi_event_plan_ids):,} Plan IDs with multiple events.")
            logger.info("\nComparison of Averages (Multiple Events vs. Overall):")
            # Define wait_days_col_to_analyze correctly before using it here
            wait_days_col_orig = 'ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח'
            wait_days_col_calc = 'wait_days_calc'
            if wait_days_col_calc in df_all_raw.columns: wait_days_col_to_analyze = wait_days_col_calc
            elif wait_days_col_orig in numeric_cols_final: wait_days_col_to_analyze = wait_days_col_orig
            else: wait_days_col_to_analyze = None

            comparison = {
                'Metric': ['Avg Age', 'Avg BMI (Raw*)', 'Avg Wait Days', 'Cancellation Rate (%)'],
                'Multiple Events': [
                    df_all_multi_event_subset['גיל המטופל בזמן הניתוח'].mean(),
                    pd.to_numeric(df_all_multi_event_subset['BMI'], errors='coerce').mean(),
                    df_all_multi_event_subset[wait_days_col_to_analyze].mean() if wait_days_col_to_analyze and wait_days_col_to_analyze in df_all_multi_event_subset else np.nan,
                    df_all_multi_event_subset['was_canceled_accurate'].mean() * 100
                ],
                'Overall': [
                    df_all_raw['גיל המטופל בזמן הניתוח'].mean(),
                    pd.to_numeric(df_all_raw['BMI'], errors='coerce').mean(),
                    df_all_raw[wait_days_col_to_analyze].mean() if wait_days_col_to_analyze and wait_days_col_to_analyze in df_all_raw else np.nan,
                    df_all_raw['was_canceled_accurate'].mean() * 100
                ]
            }
            df_comparison = pd.DataFrame(comparison)
            logger.info(df_comparison.round(2).to_string(index=False))
            logger.info("(* BMI Averages are indicative only due to data quality issues)")

            logger.info("\nMost Common Departments ('מחלקה מנתחת') for Multiple Events:")
            logger.info(df_all_multi_event_subset['מחלקה מנתחת'].value_counts().head(10).to_string())

        else:
            logger.info("Could not find matching entries in df_all_raw for Plan IDs with multiple events.")
    else:
        logger.info("DataFrame for multiple events analysis is empty.")
else:
    logger.info("No Plan IDs with multiple distinct cancellation events were identified (variable 'num_plan_ids_with_multiple_events' not > 0 or not defined).")


# ---------------------------------------------------------------------------
# Key Insights Summary (Section 11)
# ---------------------------------------------------------------------------
logger.info("\n=== 11. Key Insights and Actionable Points from Raw Data ===")
# Use f-string for dynamic summary values
summary_text = f"""
Based on the analysis of the raw data ('Surgery_Data.xlsx'):

**1. Data Scope & Cancellation Consistency:**
   - Main dataset (df_all_raw): {df_all_raw.shape[0]:,} records, {df_all_raw.shape[1]} columns.
   - Cancellation log (df_cancel_raw): {total_cancel_rows:,} records. Contains {unique_canceled_ids_count:,} unique Plan IDs.
   - Exact duplicate rows (identical across all columns) identified: {num_rows_removed_if_deduped:,}. These are likely errors.
   - Multiple distinct cancellation events (same Plan ID, different row data) identified for {num_plan_ids_with_multiple_events:,} unique Plan IDs, involving {num_rows_in_multiple_events:,} rows after removing exact duplicates. These likely represent real rescheduling/cancellation sequences.
   - Crucially, only ~{num_canceled_ids_found:,} unique Plan IDs from the ENTIRE cancellation log are present in the main dataset (df_all_raw).
   - The derived 'was_canceled_accurate' flag ({df_all_raw['was_canceled_accurate'].mean()*100:.2f}% True) uses the presence of *any* entry for a Plan ID in the cancel log for matching.

**2. Missing Values - Major Concern:** Remains a key issue. High percentages missing in labs, BMI, diagnoses, medications, etc. Action Point: Imputation/flagging strategy needed in 002.

**3. Target Variable Imbalance:** Remains the same (~1:4 ratio). Action Point: Use appropriate metrics (Precision, Recall, AUC) and consider imbalance techniques.

**4. Numerical Features Insights & Issues:**
   - Age: Similar average age between groups. Max age (124) needs check.
   - **BMI:** Still contains highly erroneous values. Critical cleaning needed (Action Point).
   - Lab tests & Sodium: As before, minor differences, check potential text in Sodium during cleaning (Action Point).
   - Wait Time: Still shows a significant difference (longer wait for canceled). Ensure inclusion in engineered features (Action Point).

**5. Categorical Features Insights:** Room, Anesthesia, Procedure Code, Site show strongest rate differences. High cardinality needs addressing (Action Point).

**6. Text Features (Counts):** Counts themselves remain weak predictors.

**7. Multiple Cancellation Events (NEW INSIGHTS):**
   - A significant number of Plan IDs ({num_plan_ids_with_multiple_events:,}) have multiple distinct entries in the cancellation log.
   - Analysis of timing and characteristics of these patients/surgeries may reveal patterns. This data is valuable. Action Point: Consider how to leverage this in feature engineering (e.g., 'num_previous_cancel_events').

**8. Overall Next Steps:**
   - Proceed to feature engineering (002).
   - **Critical:** Clean BMI.
   - **Important:** Include 'wait_days'.
   - Address Missing Values (Imputation strategy).
   - Address High Cardinality features.
   - **Consider:** How to leverage multiple cancellation event info.
   - Re-run detailed analysis (003) *after* feature engineering.
"""
logger.info(summary_text)


# ---------------------------------------------------------------------------
# Final Message
# ---------------------------------------------------------------------------
logger.info("\n=== 12. End of Initial Data Understanding Report ===") # Renumbered
logger.info("This report provides a statistical overview of the raw data, including duplicate analysis.")
logger.info("Review the missing values, distributions, cancellation rates, and multiple event analysis.")
logger.info("The 'was_canceled_accurate' column used here is derived based on *any* matching plan ID in the cancel log.")
logger.info("Next logical step is Feature Engineering (like in 002_Feature_Engineering.py).")
logger.info(f"\n--- Analysis complete. Full output saved to: {log_filepath.resolve()} ---")