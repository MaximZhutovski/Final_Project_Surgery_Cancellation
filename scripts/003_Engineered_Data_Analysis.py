# 003_Final_Cleaned_Data_EDA.py
"""
Loads the user's FINAL CLEANED and MANUALLY ENCODED engineered surgery data.
Performs comprehensive statistical analysis and generates requested visualizations.
Graphs will display the numerical codes as defined by the user.
"""
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ... (Setup: Add project root to PYTHONPATH and import config - ללא שינוי) ...
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import ENGINEERED_DATA_XLSX, RESULTS_DIR, PLOT_DIR 
    print("Successfully imported paths from config.py for 003 (Final Cleaned Data EDA - Numeric Labels)")
except ImportError:
    print("CRITICAL (003): config.py not found or essential paths missing.")
    scripts_dir = Path(__file__).resolve().parent
    project_root_alt = scripts_dir.parent
    ENGINEERED_DATA_XLSX = project_root_alt / "data" / "surgery_data_engineered_v3.xlsx" 
    RESULTS_DIR = project_root_alt / "results"
    PLOT_DIR = project_root_alt / "plots"
    print(f"Warning (003): Using fallback paths. Input data path set to: {ENGINEERED_DATA_XLSX}")
    (project_root_alt / "data").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (003): An unexpected error occurred during config import: {e}")
    sys.exit(1)

# ... (Setup Output Logging - ללא שינוי) ...
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".txt") -> Path:
    base_name = prefix; counter = 0
    while True:
        file_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not file_path.exists(): return file_path
        counter += 1
log_filename_base = Path(__file__).stem
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler(log_filepath, encoding='utf-8'),
                              logging.StreamHandler(sys.stdout)])

# --- Start of Analysis ---
logger.info(f"--- Final Cleaned Data EDA (Numeric Labels in Plots) (Script: {log_filename_base}) ---")
logger.info(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Using input file: {ENGINEERED_DATA_XLSX}")
input_sheet_name = "features_focused_v1" 
logger.info(f"Reading from sheet: {input_sheet_name}")
if not ENGINEERED_DATA_XLSX.exists():
    logger.error(f"Data file not found at: {ENGINEERED_DATA_XLSX.resolve()}"); sys.exit(1)

def get_df_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO(); df.info(buf=buffer, verbose=True, show_counts=True); return buffer.getvalue()

logger.info("\n=== 1. Loading User's Final Cleaned and Encoded Data ===")
try:
    df_final_cleaned = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name=input_sheet_name)
    logger.info(f"Loaded data from '{input_sheet_name}': {df_final_cleaned.shape[0]:,} rows, {df_final_cleaned.shape[1]} columns")
except Exception as e:
    logger.error(f"Error loading the Excel file from sheet '{input_sheet_name}': {e}"); sys.exit(1)

# --- Target Column & Boolean-like Columns Normalization (ללא שינוי מהגרסה הקודמת) ---
target_col = 'was_canceled' 
logger.info(f"\n=== Normalizing Target & Boolean-like Columns (Ensuring 1=True, 0=False) ===")
if target_col in df_final_cleaned.columns:
    unique_target_values = set(df_final_cleaned[target_col].dropna().unique())
    logger.info(f"Original '{target_col}' unique values: {unique_target_values}")
    if unique_target_values <= {0, 1} or unique_target_values <= {0.0, 1.0}:
        map_to_standard_boolean = {0: 1, 1: 0} 
        df_final_cleaned[target_col] = df_final_cleaned[target_col].map(map_to_standard_boolean)
        logger.info(f"Normalized '{target_col}' from 0/1 (0=True) to 1/0 (1=True).")
    elif unique_target_values == {True, False}:
        df_final_cleaned[target_col] = df_final_cleaned[target_col].astype(int)
        logger.info(f"Converted boolean '{target_col}' to 0/1 int (True=1, False=0).")
    df_final_cleaned[target_col] = pd.to_numeric(df_final_cleaned[target_col], errors='coerce').fillna(-1).astype(int)
    if -1 in df_final_cleaned[target_col].unique(): logger.warning(f"NaNs found in '{target_col}' after conversion became -1.")
else: logger.error(f"Target column '{target_col}' not found!"); sys.exit(1)

boolean_like_cols_to_normalize = {'is_weekend': {0: 1, 1: 0}, 'near_holiday': {0: 1, 1: 0}}
for col, col_map_if_inverted in boolean_like_cols_to_normalize.items():
    if col in df_final_cleaned.columns:
        unique_col_values = set(df_final_cleaned[col].dropna().unique())
        if unique_col_values <= {0,1} or unique_col_values <= {0.0, 1.0}:
            df_final_cleaned[col] = df_final_cleaned[col].map(col_map_if_inverted)
        elif unique_col_values == {True, False}: df_final_cleaned[col] = df_final_cleaned[col].astype(int)
        df_final_cleaned[col] = pd.to_numeric(df_final_cleaned[col], errors='coerce').fillna(-1).astype(int)
        if -1 in df_final_cleaned[col].unique(): logger.warning(f"NaNs in '{col}' became -1.")

# --- Basic Structure, Info, Missing Values, Target Analysis (ללא שינוי מהותי מהגרסה הקודמת) ---
logger.info("\n=== 2. Basic Structure & Info (Post-Normalization) ===")
logger.info(f"Shape: {df_final_cleaned.shape}\n{get_df_info(df_final_cleaned)}")
all_features = [col for col in df_final_cleaned.columns if col != target_col and col != 'plan_id'] 
logger.info(f"Features for analysis: {all_features}\nFirst 5 rows:\n{df_final_cleaned.head().to_string()}")
logger.info("\n=== 3. Missing Value Analysis ===")
missing_counts = df_final_cleaned.isnull().sum()
missing_info_df = pd.DataFrame({'Missing Count': missing_counts[missing_counts > 0], 
                                'Missing Percentage (%)': (missing_counts[missing_counts > 0] / len(df_final_cleaned) * 100).round(2)})
if not missing_info_df.empty: logger.info(f"Columns with NaN Values:\n{missing_info_df.sort_values('Missing Percentage (%)', ascending=False).to_string()}")
else: logger.info("No NaN values found.")
logger.info(f"\n=== 4. Target Variable Analysis ('{target_col}') (Post-Normalization) ===")
cancel_rate = df_final_cleaned[target_col].mean(skipna=True) * 100
logger.info(f"Overall Cancellation Rate: {cancel_rate:.2f}% (1 means Canceled)")
logger.info(f"Distribution:\n{df_final_cleaned[target_col].value_counts(dropna=False, normalize=True).round(4).to_string()}")
logger.info(f"\n=== 5. Feature Analysis (Numerically Encoded Features) ===")
features_to_analyze = [col for col in all_features if col in df_final_cleaned.columns]
if features_to_analyze:
    non_numeric = [col for col in features_to_analyze if not pd.api.types.is_numeric_dtype(df_final_cleaned[col])]
    if non_numeric: logger.warning(f"Non-numeric features: {non_numeric}")
    try: logger.info(f"\nOverall Descriptive Stats:\n{df_final_cleaned[features_to_analyze].describe(include='all').round(3).to_string()}")
    except Exception as e: logger.error(f"Error in overall describe: {e}")
    if target_col in df_final_cleaned.columns:
        try:
            numeric_for_grouping = [f for f in features_to_analyze if pd.api.types.is_numeric_dtype(df_final_cleaned[f])]
            if numeric_for_grouping: logger.info(f"\nGrouped Mean by '{target_col}':\n{df_final_cleaned.groupby(target_col)[numeric_for_grouping].mean().round(3).to_string()}")
        except Exception as e: logger.error(f"Error in grouped describe: {e}")
else: logger.info("No features for detailed analysis.")

# ---------------------------------------------------------------------------
# Section 6: Visualizations (עם תוויות מספריות כפי שביקשת)
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Generating Visualizations (Displaying Numerical Codes) ===")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_numeric_codes_cancellations(df, column_name, target_column, plot_dir, top_n=20):
    """
    Generates a bar plot for numerically encoded categorical features,
    displaying the numerical codes themselves on the x-axis.
    """
    logger.info(f"Generating plot for numerically-encoded column '{column_name}'...")
    plt.figure(figsize=(14, 8))
    
    temp_df = df.copy()
    # העמודה כבר אמורה להיות עם הקודים המספריים שלך
    # נמיר למחרוזת לצורך קיבוץ נכון בגרף, כדי ש-1 ו-1.0 יטופלו כאותה קטגוריה
    plot_column_display = temp_df[column_name].astype(str) 
    
    x_label_rotation = 45 
    # אם יש מעט קטגוריות, אין צורך לסובב את התוויות
    if plot_column_display.nunique() <= 7 : x_label_rotation = 0
    
    plot_title = f'Surgeries & Cancellations by {column_name} (Numerical Codes)'
    
    df_for_plot = pd.DataFrame({'category_plot_col': plot_column_display, target_column: temp_df[target_column]})
    total_counts = df_for_plot['category_plot_col'].value_counts()
    
    # אם יש יותר מדי ערכים ייחודיים, הצג top_n
    # (אבל מכיוון שאתה כבר סיננת וקידדת, אולי זה לא נחוץ לרוב העמודות)
    categories_to_plot = total_counts.index
    if total_counts.nunique() > top_n:
        logger.info(f"Column '{column_name}' has {total_counts.nunique()} unique codes. Showing top {top_n} by frequency.")
        categories_to_plot = total_counts.nlargest(top_n).index
        df_for_plot = df_for_plot[df_for_plot['category_plot_col'].isin(categories_to_plot)]
        # עדכן את total_counts אחרי הסינון
        total_counts = df_for_plot['category_plot_col'].value_counts()

    # ספירת הביטולים (כאשר המטרה היא 1)
    canceled_counts = df_for_plot[df_for_plot[target_column] == 1]['category_plot_col'].value_counts()
    
    plot_data = pd.DataFrame({'Total Surgeries': total_counts, 'Canceled Surgeries': canceled_counts}).fillna(0)
    plot_data['Canceled Surgeries'] = plot_data['Canceled Surgeries'].astype(int)

       # מיון הגרף: בדרך כלל לפי סך הניתוחים, מהגבוה לנמוך
    # אלא אם כן יש מעט מאוד קטגוריות ואז מיון לפי הקוד יכול להיות קריא יותר
    # כרגע, נתעדף מיון לפי סך הכל עבור כל המקרים, אלא אם כן תרצה לשנות עבור מקרים ספציפיים.
    
    # אם יש יותר מ-top_n קטגוריות, plot_data כבר ממוין לפי Total Surgeries מהקטע הקודם של לקיחת ה-top_n
    # אם יש פחות או שווה ל-top_n קטגוריות, נמיין כאן
    if total_counts.nunique() <= top_n : # אם לא סיננו ל-top_n, נמיין עכשיו
         plot_data = plot_data.sort_values(by='Total Surgeries', ascending=False)
    # אם סיננו ל-top_n, plot_data כבר אמור להיות ממוין נכון (כי לקחנו nlargest מ-total_counts
    # ואז יצרנו את plot_data. אם לא, אפשר להוסיף כאן מיון נוסף).
    # כדי לוודא, אפשר פשוט למיין תמיד:
    # plot_data = plot_data.sort_values(by='Total Surgeries', ascending=False)

    # בשורה התחתונה, כדי להבטיח מיון מהגדול לקטן לפי סך הניתוחים:
    plot_data = plot_data.sort_values(by='Total Surgeries', ascending=False)
    if plot_data.empty:
        logger.warning(f"No data to plot for {column_name}. Skipping plot.")
        plt.close(); return

    ax = plot_data['Total Surgeries'].plot(kind='bar', color='skyblue', label='Total Surgeries', alpha=0.7, width=0.4, position=1)
    ax = plot_data['Canceled Surgeries'].plot(kind='bar', color='salmon', label='Canceled Surgeries', alpha=0.7, width=0.4, position=0, ax=ax)

    plt.title(plot_title, fontsize=16)
    plt.xlabel(f"{column_name} (Code)", fontsize=13) # הוספתי "(Code)"
    plt.ylabel('Number of Surgeries', fontsize=13)
    plt.xticks(rotation=x_label_rotation, ha="right" if x_label_rotation > 0 else "center", fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title='Status', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    safe_column_name = "".join(c if c.isalnum() else "_" for c in column_name)
    plot_filename = PLOT_DIR / f"plot_numericcodes_{safe_column_name}_cancellations.png"
    try:
        plt.savefig(plot_filename)
        logger.info(f"Saved plot to {plot_filename}")
    except Exception as e:
        logger.error(f"Error saving plot for {column_name}: {e}")
    plt.close()

# רשימת העמודות שאתה רוצה להן גרפים (אלו העמודות שקידדת ידנית)
# וגם עמודות שהן נומריות מטבען
columns_for_plotting_numeric_codes = [
    'payer', 'anesthesia', 'age_bucket', 'city', 'near_holiday', 'is_weekend',
    'season', 'surgery_weekday', 'marital_status', 'gender', 'surgery_site',
    'wait_days_category', 'distance_bucket',
    # עמודות נומריות רציפות שגם עבורן אולי תרצה לראות גרף דומה (אם כי היסטוגרמה תתאים להן יותר)
    # או עמודות כמו department, procedure_code אם גם אותן קידדת למספרים.
    'age', 'wait_days', 'distance_km', 'num_medications', 'num_diagnoses', 
    'room', 'department', 'procedure_code' 
]
# סנן רק את אלו שקיימות ב-DataFrame
columns_to_actually_plot = [col for col in columns_for_plotting_numeric_codes if col in df_final_cleaned.columns]

for col_name in columns_to_actually_plot:
    # קבע top_n לפי סוג העמודה אם צריך (למשל, אם city עדיין מכיל הרבה קודים אחרי הסינון שלך)
    current_top_n = 20 # ברירת מחדל
    if col_name == 'city': current_top_n = 15 
    elif col_name in ['procedure_code', 'department'] and df_final_cleaned[col_name].nunique() > 25: 
        current_top_n = 25
        
    plot_numeric_codes_cancellations(
        df_final_cleaned, col_name, target_col, PLOT_DIR, 
        top_n=current_top_n
    )

# --- Correlation Heatmap (ללא שינוי מהגרסה הקודמת) ---
logger.info("\n=== Generating Correlation Heatmap ===")
features_for_heatmap = [col for col in df_final_cleaned.columns if col != 'plan_id']
numeric_features_for_heatmap = df_final_cleaned[features_for_heatmap].select_dtypes(include=np.number).columns.tolist()
if not numeric_features_for_heatmap:
    logger.warning("No numeric features found for correlation heatmap. Skipping.")
else:
    if target_col not in numeric_features_for_heatmap and pd.api.types.is_numeric_dtype(df_final_cleaned[target_col]):
        numeric_features_for_heatmap.append(target_col)
    correlation_matrix = df_final_cleaned[numeric_features_for_heatmap].corr()
    plt.figure(figsize=(max(12, len(numeric_features_for_heatmap)*0.5), max(10, len(numeric_features_for_heatmap)*0.4)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 7})
    plt.title('Correlation Heatmap of Features (Including Target)', fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=9); plt.yticks(fontsize=9)
    plt.tight_layout()
    heatmap_filename = PLOT_DIR / "correlation_heatmap.png"
    try: plt.savefig(heatmap_filename); logger.info(f"Saved correlation heatmap to {heatmap_filename}")
    except Exception as e: logger.error(f"Error saving heatmap: {e}")
    plt.close()

# --- Section 7: Data Readiness Summary (ללא שינוי מהותי מהגרסה הקודמת) ---
logger.info(f"\n=== 7. Data Readiness Summary for Next Step (004 - Modeling) ===")
# ... (הסיכום נשאר דומה) ...
num_rows_s7, num_cols_s7 = df_final_cleaned.shape
features_for_model_s7 = [col for col in df_final_cleaned.columns if col != target_col and col != 'plan_id']
num_features_for_model_s7 = len(features_for_model_s7)
missing_summary_s7 = "No NaN values found in features for modeling (as per user's cleaning and normalization)."
if 'missing_info_df' in locals() and not missing_info_df.empty:
    relevant_missing_s7 = missing_info_df[missing_info_df.index.isin(features_for_model_s7)]
    if not relevant_missing_s7.empty:
        top_missing_col_s7 = relevant_missing_s7.index[0]; top_missing_pct_s7 = relevant_missing_s7.iloc[0,1]
        missing_summary_s7 = f"{len(relevant_missing_s7)} relevant feature columns with NaN values. Highest: {top_missing_col_s7} ({top_missing_pct_s7}%)"
summary_text_recommendations = f"""
**1. Data Structure (User Cleaned & Encoded, Target Normalized):**
   - Shape: {num_rows_s7:,} rows, {num_cols_s7} columns.
   - Features for modeling: {num_features_for_model_s7}
   - All features numerically encoded. Target '{target_col}' is 1 (Canceled) or 0 (Not Canceled).
**2. Missing Values in Features for Modeling:**
   - Status: {missing_summary_s7}
   - **Action for 004 (Modeling Script):** Address any remaining NaNs.
**3. Target Variable ('{target_col}'):**
   - Rate: {df_final_cleaned[target_col].mean(skipna=True)*100:.2f}% Canceled. Consider imbalance.
**4. Next Steps in 004 (Random Forest Modeling):**
   - Load this final cleaned dataset. Verify no unexpected NaNs/types in X.
   - Define X and y. Train-Test Split. Train RandomForestClassifier. Evaluate. Feature Importance.
"""
logger.info(summary_text_recommendations)
logger.info(f"\n--- End of Final Cleaned Data EDA (Numeric Labels) (Script: {log_filename_base}) ---")
logger.info(f"Full output saved to: {log_filepath.resolve()}")
logger.info(f"Plots saved in: {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    pass