import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# File paths
input_file = Path(r'c:\Users\pdaadh\Desktop\KT digiarvi\V3_1b polished and anonymized (1).xlsx')
output_file = Path('preprocessed_kt_data.csv')

print(f"Reading data from {input_file}...")
# Read the Excel file
df = pd.read_excel(input_file, sheet_name=0)

# Define exercise columns (all columns starting with M3S)
exercise_cols = [col for col in df.columns if col.startswith('M3S')]
print(f"Found {len(exercise_cols)} exercise columns")

# Create a mapping of exercise codes to their categories
exercise_categories = {
    'M3S201a': 'Rally1 2x3',
    'M3S201b': 'Rally2 5x4',
    'M3S201c': 'Rally3 10x8',
    'M3S201d': 'Rally4 9x3',
    'M3S201e': 'Rally5 3x7',
    'M3S201f': 'Rally6 4x4',
    'M3S201g': 'Rally7 4x6',
    'M3S201h': 'Rally8 6x7',
    'M3S201i': 'Rally9 7x4',
    'M3S201j': 'Rally10 8x6',
    'M3S202': 'Subtractions 1',
    'M3S203': 'Divisions 1',
    'M3S501': 'Geometry: Basic concepts',
    'M3S601': 'Rain statistics (daily rainfall amounts)',
    'M3S301': 'Missing number: Subtraction',
    'M3S204': 'Additions 2',
    'M3S205': 'Subtractions 2',
    'M3S101': 'Word puzzle – bars',
    'M3S206': 'Subtractions 3',
    'M3S207': 'Multiplications 2',
    'M3S502': 'Distance on a map',
    'M3S302': 'Missing number: Division',
    'M3S102': 'Logical reasoning',
    'M3S208': 'Divisions 3',
    'M3S602': 'Fix the robot’s code',
}

# Coarse (group-level) categories for RL action space
exercise_categories_coarse = {
    # Multiplication
    'M3S201a': 'Multiplication', 'M3S201b': 'Multiplication', 'M3S201c': 'Multiplication',
    'M3S201d': 'Multiplication', 'M3S201e': 'Multiplication', 'M3S201f': 'Multiplication',
    'M3S201g': 'Multiplication', 'M3S201h': 'Multiplication', 'M3S201i': 'Multiplication',
    'M3S201j': 'Multiplication', 'M3S207': 'Multiplication',
    # Subtraction
    'M3S202': 'Subtraction', 'M3S205': 'Subtraction', 'M3S206': 'Subtraction',
    # Division
    'M3S203': 'Division', 'M3S208': 'Division',
    # Addition
    'M3S204': 'Addition',
    # Geometry
    'M3S501': 'Geometry', 'M3S502': 'Geometry',
    # Data Analysis
    'M3S601': 'Data_Analysis',
    # Missing Numbers
    'M3S301': 'Missing_Numbers', 'M3S302': 'Missing_Numbers',
    # Word Problems
    'M3S101': 'Word_Problems',
    # Logical Reasoning
    'M3S102': 'Logical_Reasoning',
    # Coding
    'M3S602': 'Coding'
}

# Create a mapping of exercise codes to their order in the curriculum
# Max possible per exercise (for performance comparison)
exercise_max = {
    'M3S201a': 1, 'M3S201b': 1, 'M3S201c': 1, 'M3S201d': 1, 'M3S201e': 1,
    'M3S201f': 1, 'M3S201g': 1, 'M3S201h': 1, 'M3S201i': 1, 'M3S201j': 1,
    'M3S202': 3, 'M3S203': 3, 'M3S501': 5, 'M3S601': 3, 'M3S301': 3,
    'M3S204': 4, 'M3S205': 4, 'M3S101': 2, 'M3S206': 3, 'M3S207': 4,
    'M3S502': 3, 'M3S302': 2, 'M3S102': 4, 'M3S208': 3, 'M3S602': 1
}

# Create a mapping of exercise codes to their order in the curriculum
exercise_order = {code: i+1 for i, code in enumerate(exercise_cols)}

# Helpers
def _standardize_sex(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("boy", "male", "m"):
        return "Boy"
    if s in ("girl", "gir", "female", "f"):
        return "Girl"
    return str(val).strip().capitalize()

print("Processing student-exercise interactions...")
# Melt the dataframe to long format (student-exercise interactions)
kt_data = []
for _, row in df.iterrows():
    student_id = row['Orig_order']
    for exer in exercise_cols:
        # Treat NaN values as 0
        score = 0 if pd.isna(row[exer]) else int(row[exer])
        max_val = exercise_max.get(exer, np.nan)
        normalized_score = (score / max_val) if (pd.notna(max_val) and max_val > 0) else np.nan
        sex_std = _standardize_sex(row.get('sex', None))
        # language match feature
        hl = row.get('home_lang', np.nan)
        sl = row.get('school_lang', np.nan)
        if pd.notna(hl) and pd.notna(sl):
            home_school_lang_match = int(str(hl).strip().lower() == str(sl).strip().lower())
        else:
            home_school_lang_match = np.nan
        kt_data.append({
            'student_id': student_id,
            'exercise_id': exer,
            # Descriptive label for category
            'category': exercise_categories.get(exer, 'Other'),
            # Coarse grouping used by RL
            'category_group': exercise_categories_coarse.get(exer, 'Other'),
            'order': exercise_order[exer],
            'score': score,  # Store the raw score (0-5), with NaN treated as 0
            'max': max_val,  # Max value for this exercise
            'normalized_score': normalized_score,
            'reward': normalized_score,
            'flag_score_exceeds_max': int(score > max_val) if pd.notna(max_val) else 0,
            'grade': row['grade'],
            'sex': sex_std,  # Standardized sex label
            'mean_perception': row.get('MEAN_PERCPT', None),  # Add mean perception score
            # Version / context
            'ver_oplm': row.get('Ver_OPLM', None),
            'ver_da': row.get('Ver_DA', None),
            'digiarvi_version': row.get('digiarvi_version', None),
            # Language context
            'school_lang': row.get('school_lang', None),
            'home_lang': row.get('home_lang', None),
            'strong_lang': row.get('strong_lang', None),
            'friend_lang': row.get('friend_lang', None),
            'home_school_lang_match': home_school_lang_match,
            # Missingness / engagement
            'missing_all': row.get('missing_all', None),
            'missing_beginning30': row.get('missing_beginning30', None),
            'missing_last50': row.get('missing_last50', None),
            # Aggregates / ability proxies (primarily for evaluation)
            'pSUM': row.get('pSUM', None),
            'SUM_Rally': row.get('SUM_Rally', None),
            'SUM': row.get('SUM', None),
            'STD_TOTAL_theta_25': row.get('STD_TOTAL_theta_25', None),
            'T10_TOTAL_theta_25': row.get('T10_TOTAL_theta_25', None)
        })

# Convert to DataFrame
kt_df = pd.DataFrame(kt_data)

 # No additional derived metrics added here. Kept only core fields and per-exercise max.

# Save to CSV
print(f"Saving preprocessed data to {output_file}...")
try:
    kt_df.to_csv(output_file, index=False)
except PermissionError:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback = Path(f"preprocessed_kt_data_{ts}.csv")
    print(f"Permission denied writing {output_file}. Saving to {fallback} instead. Close the CSV if open to overwrite default.")
    kt_df.to_csv(fallback, index=False)
    output_file = fallback

print("\nPreprocessing complete!")
print(f"Total interactions: {len(kt_df)}")
print(f"Unique students: {kt_df['student_id'].nunique()}")
print(f"Unique exercises: {kt_df['exercise_id'].nunique()}")
print("\nSample of preprocessed data:")
print(kt_df.head())

