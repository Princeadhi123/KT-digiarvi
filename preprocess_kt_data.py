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

print("Processing student-exercise interactions...")
# Melt the dataframe to long format (student-exercise interactions)
kt_data = []
for _, row in df.iterrows():
    student_id = row['Orig_order']
    for exer in exercise_cols:
        # Treat NaN values as 0
        score = 0 if pd.isna(row[exer]) else int(row[exer])
        kt_data.append({
            'student_id': student_id,
            'exercise_id': exer,
            'category': exercise_categories.get(exer, 'Other'),
            'order': exercise_order[exer],
            'score': score,  # Store the raw score (0-5), with NaN treated as 0
            'max': exercise_max.get(exer, np.nan),  # Max value for this exercise
            'grade': row['grade'],
            'sex': row['sex'],  # Add sex information
            'mean_perception': row.get('MEAN_PERCPT', None)  # Add mean perception score
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

