import pandas as pd

# Load the data
df = pd.read_csv('../UK online student engagement.csv', index_col=0)

# Create mapping dictionaries
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5}
dropout_mapping = {'Y': 1, 'N': 0}
year_mapping = {'first': 1, 'second': 2, 'third': 3}
external_mapping = {'Y': 1, 'N': 0}

# Apply mappings
# Convert External column
df['External'] = df['External'].map(external_mapping)

# Convert Year column
df['Year'] = df['Year'].map(year_mapping)

# Convert dropout column
df['dropout'] = df['dropout'].map(dropout_mapping)

# Convert grade columns - find all columns that might contain letter grades
grade_columns = ['test 1', 'test 2', 'test 3', 'ind cw', 'group cw', 'final grade']
for col in grade_columns:
    if col in df.columns:
        df[col] = df[col].map(grade_mapping)

# Save the processed data
df.to_csv('../processed_student_engagement.csv')

print("Data processing complete. Output saved to 'processed_student_engagement.csv'")
