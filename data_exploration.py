import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

### Overview of dropout rates
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
    plt.savefig('exploration_graphs/numeric_distributions.png')
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
plt.savefig('exploration_graphs/categorical_distributions.png')
plt.close()

### Bivariate Analysis
# Correlation matrix and heatmap
plt.figure(figsize=(12, 10))
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('exploration_graphs/correlation_heatmap.png')
plt.close()

# Analyze relationship between session attendance and dropout
session_cols = [col for col in data.columns if 'session' in col]
plt.figure(figsize=(12, 8))
for i, col in enumerate(session_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='dropout', y=col, data=data)
    plt.title(f'{col} by Dropout Status')
plt.tight_layout()
plt.savefig('exploration_graphs/session_attendance_by_dropout.png')
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
plt.savefig('exploration_graphs/forum_engagement_by_dropout.png')
plt.close()

### Advanced Analytics
# Feature engineering - create engagement metrics
session_cols = [col for col in data.columns if 'session' in col]

# Calculate attendance rate based on time spent in sessions
# First, replace 0 values with NaN since they represent non-attendance
for col in session_cols:
    data[col] = data[col].replace(0, np.nan)

# Calculate the average attendance percentage across all sessions
# For each student, sum the attendance percentages and divide by the number of sessions
data['attendance_quality'] = data[session_cols].mean(axis=1, skipna=True) / 100  # Divide by 100 to get 0-1 scale

# Calculate the attendance rate (proportion of sessions attended)
data['attendance_rate'] = data[session_cols].count(axis=1) / len(session_cols)

# Create a combined attendance metric that considers both rate and quality
data['attendance_score'] = data['attendance_rate'] * data['attendance_quality']

# Calculate forum engagement
data['forum_engagement'] = data['fourm Q'] + data['fourm A']

# Normalize the metrics to 0-1 range to ensure proper weighting
max_forum = data['forum_engagement'].max()
max_office = data['office hour visits'].max()

# Update engagement score to use the new attendance_score
data['engagement_score'] = (
    data['attendance_score'] * 0.5 + 
    (data['forum_engagement'] / max_forum if max_forum > 0 else 0) * 0.3 + 
    (data['office hour visits'] / max_office if max_office > 0 else 0) * 0.2
)

# Analyze the new features
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='dropout', y='attendance_rate', data=data)
plt.title('Attendance Rate by Dropout Status')

plt.subplot(2, 2, 2)
sns.boxplot(x='dropout', y='attendance_quality', data=data)
plt.title('Attendance Quality by Dropout Status')

plt.subplot(2, 2, 3)
sns.boxplot(x='dropout', y='attendance_score', data=data)
plt.title('Combined Attendance Score by Dropout Status')

plt.subplot(2, 2, 4)
sns.boxplot(x='dropout', y='engagement_score', data=data)
plt.title('Overall Engagement Score by Dropout Status')

plt.tight_layout()
plt.savefig('exploration_graphs/attendance_metrics.png')
plt.close()

# Additional visualization to compare the different attendance metrics
plt.figure(figsize=(10, 6))
attendance_metrics = ['attendance_rate', 'attendance_quality', 'attendance_score']
for dropout_status in [0, 1]:
    subset = data[data['dropout'] == dropout_status]
    values = [subset[metric].mean() for metric in attendance_metrics]
    plt.bar(
        [f"{metric} ({'Dropout' if dropout_status == 1 else 'Non-Dropout'})" 
         for metric in attendance_metrics], 
        values,
        alpha=0.7,
        color='red' if dropout_status == 1 else 'blue'
    )

plt.title('Comparison of Attendance Metrics by Dropout Status')
plt.ylabel('Average Value (0-1 scale)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('exploration_graphs/attendance_metrics_comparison.png')
plt.close()

# Pairplot for key variables
sns.pairplot(data=data, 
             vars=['session 1', 'fourm Q', 'fourm A', 'office hour visits'], 
             hue='dropout')
plt.savefig('exploration_graphs/key_variables_pairplot.png')
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
plt.savefig('exploration_graphs/attendance_trends.png')
plt.close()

### Student Performance Analysis
# Analyze test performance
test_cols_original = [col for col in data.columns if 'test' in col and '_numeric' not in col]
test_cols_numeric = [col for col in data.columns if 'test' in col and '_numeric' in col]

# Function to find the last test before dropout
def get_last_test_before_dropout(row):
    # For students who dropped out
    if row['dropout'] == 1:
        # Get the test scores (both original and numeric)
        test_scores = row[test_cols_original]
        test_scores_numeric = row[test_cols_numeric]
        
        # Find the last non-null test
        last_test_idx = None
        for i, score in enumerate(test_scores):
            if pd.notna(score):
                last_test_idx = i
            else:
                # Stop once we hit the first null (dropout happened after this)
                break
                
        if last_test_idx is not None:
            return test_scores_numeric.iloc[last_test_idx]
    
    return np.nan

# Apply the function to get the last test score before dropout
data['last_test_before_dropout'] = data.apply(get_last_test_before_dropout, axis=1)

# Calculate average test scores for non-dropout students
non_dropout_avg_scores = data[data['dropout'] == 0][test_cols_numeric].mean(axis=1)
data.loc[data['dropout'] == 0, 'avg_test_score'] = non_dropout_avg_scores

# Compare the distributions
plt.figure(figsize=(12, 6))

# Plot 1: Compare distributions
plt.subplot(1, 2, 1)
# Last test before dropout
dropout_scores = data[data['dropout'] == 1]['last_test_before_dropout'].dropna()
# Average test scores for non-dropouts
non_dropout_scores = data[data['dropout'] == 0]['avg_test_score'].dropna()

# Create a DataFrame for seaborn
plot_data = pd.DataFrame({
    'Score': pd.concat([dropout_scores, non_dropout_scores]),
    'Category': ['Last Test Before Dropout'] * len(dropout_scores) + 
                ['Average Test (Non-Dropout)'] * len(non_dropout_scores)
})

sns.boxplot(x='Category', y='Score', data=plot_data)
plt.title('Last Test Before Dropout vs. Average Test Score (Non-Dropouts)')
plt.ylabel('Test Score (0-4 scale)')

# Plot 2: Show the actual score distributions
plt.subplot(1, 2, 2)

# Define common bins for both histograms
bins = [0, 1, 2, 3, 4]  # Using grade-level bins (F, D, C, B, A)
bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
bin_width = 0.35  # Width of each bar

# Calculate histograms manually
dropout_hist, _ = np.histogram(dropout_scores, bins=bins)
non_dropout_hist, _ = np.histogram(non_dropout_scores, bins=bins)

# Plot side-by-side bars
plt.bar([x - bin_width/2 for x in bin_centers], dropout_hist, 
        width=bin_width, color='red', alpha=0.7, label='Last Test Before Dropout')
plt.bar([x + bin_width/2 for x in bin_centers], non_dropout_hist, 
        width=bin_width, color='blue', alpha=0.7, label='Average Test (Non-Dropouts)')

# Add labels and formatting
plt.title('Distribution of Test Scores')
plt.xlabel('Test Score (0-4 scale)')
plt.ylabel('Frequency')
plt.xticks(bin_centers, ['F (0-1)', 'D (1-2)', 'C (2-3)', 'B (3-4)'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('exploration_graphs/pre_dropout_test_comparison.png')
plt.close()

# Calculate and print statistics
print(f"Average last test score before dropout: {dropout_scores.mean():.2f}")
print(f"Average test score for non-dropouts: {non_dropout_scores.mean():.2f}")
print(f"Difference: {dropout_scores.mean() - non_dropout_scores.mean():.2f}")

# Test if the difference is statistically significant
t_stat, p_value = stats.ttest_ind(dropout_scores, non_dropout_scores, equal_var=False)
print(f"T-test results: t={t_stat:.2f}, p={p_value:.4f}")
print(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")

# Create a more detailed analysis by year
plt.figure(figsize=(10, 6))
for year in data['Year'].unique():
    # Last test before dropout for this year
    year_dropout_scores = data[(data['dropout'] == 1) & (data['Year'] == year)]['last_test_before_dropout'].dropna()
    # Average test scores for non-dropouts in this year
    year_non_dropout_scores = data[(data['dropout'] == 0) & (data['Year'] == year)]['avg_test_score'].dropna()
    
    if len(year_dropout_scores) > 0 and len(year_non_dropout_scores) > 0:
        print(f"\nYear: {year}")
        print(f"  Average last test score before dropout: {year_dropout_scores.mean():.2f}")
        print(f"  Average test score for non-dropouts: {year_non_dropout_scores.mean():.2f}")
        print(f"  Difference: {year_dropout_scores.mean() - year_non_dropout_scores.mean():.2f}")
        
        # Plot the data
        plt.bar(
            [f"{year} - Dropout", f"{year} - Non-Dropout"], 
            [year_dropout_scores.mean(), year_non_dropout_scores.mean()],
            yerr=[year_dropout_scores.std(), year_non_dropout_scores.std()],
            capsize=10,
            alpha=0.7
        )

plt.title('Test Performance (average/last test before dropout) by Year and Dropout Status')
plt.ylabel('Average Test Score (0-4 scale)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('exploration_graphs/test_performance_by_year.png')
plt.close()

### Additional Analyses
# Identify patterns in dropout risk
# Create a function to calculate engagement percentiles
def engagement_percentile(row):
    # First check if engagement_score exists in the row
    if 'engagement_score' not in row:
        return 'unknown'
    
    score = row['engagement_score']
    if pd.isna(score):
        return 'unknown'
    
    # Calculate percentiles based on the actual distribution
    # instead of fixed thresholds
    if score <= data['engagement_score'].quantile(0.25):
        return 'very low'
    elif score <= data['engagement_score'].quantile(0.50):
        return 'low'
    elif score <= data['engagement_score'].quantile(0.75):
        return 'medium'
    else:
        return 'high'

# Now apply the engagement level categorization
data['engagement_level'] = data.apply(engagement_percentile, axis=1)

# Check the distribution of engagement levels
print("\nEngagement level distribution:")
print(data['engagement_level'].value_counts())

# Analyze dropout rates by engagement level
plt.figure(figsize=(10, 6))
dropout_by_engagement = data.groupby('engagement_level')['dropout'].mean() * 100

# Sort the categories in a logical order
order = ['very low', 'low', 'medium', 'high', 'unknown']
dropout_by_engagement = dropout_by_engagement.reindex(order)

# Plot with the correct order
dropout_by_engagement.plot(kind='bar')
plt.title('Dropout Rate by Engagement Level')
plt.ylabel('Dropout Rate (%)')
plt.ylim(0, 100)
for i, v in enumerate(dropout_by_engagement):
    if not pd.isna(v):  # Only add text for non-NaN values
        plt.text(i, v + 5, f"{v:.1f}%", ha='center')
plt.savefig('exploration_graphs/dropout_by_engagement.png')
plt.close()

# Save the processed data for modeling
data.to_csv('processed_student_data.csv')
print("\nProcessed data saved to 'processed_student_data.csv'")