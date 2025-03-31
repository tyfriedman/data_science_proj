import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

### Data Understanding
# Load the data
data = pd.read_csv('./UK online student engagement.csv', index_col=0)

# check data structure
print("Data structure:")
print(data.head())
print(data.info())
print(data.describe())

# check missing values before cleaning
print(f"\n\nMissing values before cleaning:")
print(data.isnull().sum())

### Data Cleaning
# drop missing values
full_data = data.dropna()
null_data = data[data.isnull().any(axis=1)]

# check missing values again
print(f"\n\nMissing values after cleaning:")
print(full_data.isnull().sum())

# check data structure
print(f"\n\nData structure after cleaning:")
print(full_data.head())
print(full_data.info())
print(full_data.describe())

# check dropout rate distribution
print(f"\n\nDropout rate distribution (before cleaning):")
print(data['dropout'].value_counts(normalize=True))

# check dropout rate distribution with cleaned data
print(f"\n\nDropout rate distribution (after cleaning):")
print(full_data['dropout'].value_counts(normalize=True))

### Data Preprocessing
# Convert categorical variables to proper data types
data['External'] = data['External'].map({'Y': 1, 'N': 0})
data['dropout'] = data['dropout'].map({'Y': 1, 'N': 0})
data['Year'] = data['Year'].astype('category')

# Convert letter grades to numeric values for analysis
def convert_grade(grade):
    if pd.isna(grade):
        return np.nan
    grade_map = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
    return grade_map.get(grade, np.nan)  # Return NaN for any unexpected values

grade_columns = ['test 1', 'test 2', 'test 3', 'ind cw', 'group cw', 'final grade']
for col in grade_columns:
    data[f'{col}_numeric'] = data[col].apply(convert_grade)

# Check the converted data
print("\nData after preprocessing:")
print(data.head())

# When calculating group means, use only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Summary statistics by groups (using only numeric columns)
print("\nSummary statistics by year:")
year_stats = data.groupby('Year')[numeric_data.columns].mean()
print(year_stats)

print("\nSummary statistics by External/Internal status:")
external_stats = data.groupby('External')[numeric_data.columns].mean()
print(external_stats)

print("\nSummary statistics by dropout status:")
dropout_stats = data.groupby('dropout')[numeric_data.columns].mean()
print(dropout_stats)

### Univariate Analysis
# Create a function to plot histograms for numeric columns
def plot_histograms(df, columns, bins=10, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns)//3 + 1, 3, i)
        df[column].hist(bins=bins)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig('numeric_distributions.png')
    plt.close()

# Select numeric columns for analysis
numeric_columns = ['session 1', 'session 2', 'session 3', 'session 4', 'session 5', 'session 6', 
                  'fourm Q', 'fourm A', 'office hour visits']
plot_histograms(data, numeric_columns)

# Analyze categorical variables
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
data['Year'].value_counts().plot(kind='bar')
plt.title('Distribution by Year')

plt.subplot(1, 3, 2)
data['External'].value_counts().plot(kind='bar')
plt.title('External vs Internal Students')

plt.subplot(1, 3, 3)
data['dropout'].value_counts().plot(kind='bar')
plt.title('Dropout Distribution')

plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.close()

### Bivariate Analysis
# Correlation matrix and heatmap
plt.figure(figsize=(12, 10))
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Analyze relationship between session attendance and dropout
session_cols = [col for col in data.columns if 'session' in col]
plt.figure(figsize=(12, 8))
for i, col in enumerate(session_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='dropout', y=col, data=data)
    plt.title(f'{col} by Dropout Status')
plt.tight_layout()
plt.savefig('session_attendance_by_dropout.png')
plt.close()

# Forum engagement by dropout status
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='dropout', y='fourm Q', data=data)
plt.title('Forum Questions by Dropout Status')

plt.subplot(1, 2, 2)
sns.boxplot(x='dropout', y='fourm A', data=data)
plt.title('Forum Answers by Dropout Status')

plt.tight_layout()
plt.savefig('forum_engagement_by_dropout.png')
plt.close()

### Advanced Analytics
# Feature engineering - create engagement metrics
data['attendance_rate'] = data[session_cols].count(axis=1) / len(session_cols)
data['forum_engagement'] = data['fourm Q'] + data['fourm A']
data['engagement_score'] = (
    data['attendance_rate'] * 0.5 + 
    data['forum_engagement'] / data['forum_engagement'].max() * 0.3 + 
    data['office hour visits'] / data['office hour visits'].max() * 0.2
)

# Analyze the new features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='dropout', y='attendance_rate', data=data)
plt.title('Attendance Rate by Dropout Status')

plt.subplot(1, 3, 2)
sns.boxplot(x='dropout', y='forum_engagement', data=data)
plt.title('Forum Engagement by Dropout Status')

plt.subplot(1, 3, 3)
sns.boxplot(x='dropout', y='engagement_score', data=data)
plt.title('Overall Engagement Score by Dropout Status')

plt.tight_layout()
plt.savefig('engagement_metrics.png')
plt.close()

# Pairplot for key variables
sns.pairplot(data=data, 
             vars=['session 1', 'fourm Q', 'fourm A', 'office hour visits'], 
             hue='dropout')
plt.savefig('key_variables_pairplot.png')
plt.close()

### Temporal Analysis
# Analyze attendance patterns across sessions
session_data = data[session_cols].copy()
session_data.columns = range(1, len(session_cols) + 1)

# Plot attendance over time by dropout status
plt.figure(figsize=(10, 6))
for dropout_status in [0, 1]:
    subset = data[data['dropout'] == dropout_status]
    attendance = [subset[col].mean() for col in session_cols if col in data.columns]
    plt.plot(range(1, len(attendance) + 1), attendance, 
             marker='o', 
             label=f'Dropout: {"Yes" if dropout_status == 1 else "No"}')

plt.xlabel('Session Number')
plt.ylabel('Average Attendance')
plt.title('Attendance Trends Over Time by Dropout Status')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('attendance_trends.png')
plt.close()

### Student Performance Analysis
# Analyze test performance
test_cols = [col for col in data.columns if 'test' in col and '_numeric' in col]
if test_cols:
    plt.figure(figsize=(10, 6))
    for dropout_status in [0, 1]:
        subset = data[data['dropout'] == dropout_status]
        test_scores = [subset[col].mean() for col in test_cols]
        plt.plot(range(1, len(test_scores) + 1), test_scores, 
                marker='o', 
                label=f'Dropout: {"Yes" if dropout_status == 1 else "No"}')

    plt.xlabel('Test Number')
    plt.ylabel('Average Score')
    plt.title('Test Performance Over Time by Dropout Status')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('test_performance.png')
    plt.close()

# Compare performance by year
if 'final grade_numeric' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Year', y='final grade_numeric', hue='dropout', data=data)
    plt.title('Final Grade by Year and Dropout Status')
    plt.savefig('grade_by_year.png')
    plt.close()

### Additional Analyses
# Identify patterns in dropout risk
# Create a function to calculate engagement percentiles
def engagement_percentile(row):
    thresholds = {
        'low': 25,
        'medium': 50,
        'high': 75
    }
    score = row['engagement_score']
    if pd.isna(score):
        return 'unknown'
    elif score <= thresholds['low']:
        return 'very low'
    elif score <= thresholds['medium']:
        return 'low'
    elif score <= thresholds['high']:
        return 'medium'
    else:
        return 'high'

data['engagement_level'] = data.apply(engagement_percentile, axis=1)

# Analyze dropout rates by engagement level
plt.figure(figsize=(10, 6))
dropout_by_engagement = data.groupby('engagement_level')['dropout'].mean() * 100
dropout_by_engagement.plot(kind='bar')
plt.title('Dropout Rate by Engagement Level')
plt.ylabel('Dropout Rate (%)')
plt.ylim(0, 100)
for i, v in enumerate(dropout_by_engagement):
    plt.text(i, v + 5, f"{v:.1f}%", ha='center')
plt.savefig('dropout_by_engagement.png')
plt.close()

# Save the processed data for modeling
data.to_csv('processed_student_data.csv')
print("\nProcessed data saved to 'processed_student_data.csv'")