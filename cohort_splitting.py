import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_checkpoints(df):
    """
    Analyze the dataset to identify checkpoints where students commonly drop out.
    Returns a list of checkpoint column names.
    """
    # Calculate the percentage of null values in each column
    null_percentages = df.isnull().mean() * 100

    # Create a graph to show null percentages for each column and save it as a png file
    plt.figure(figsize=(12, 8))
    plt.bar(null_percentages.index, null_percentages.values)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.xlabel('Columns')
    plt.ylabel('Null Percentage')
    plt.title('Null Percentages of Each Column')
    plt.savefig('null_percentages.png')
    plt.close()

    # Identify potential checkpoints (columns with significant increase in null values)
    checkpoints = []
    prev_null_percentage = 0

    for index, null_percentage in enumerate(null_percentages.values):
        # If there's a significant jump in null values (e.g., >5% increase)
        if null_percentage - prev_null_percentage > 5:
            checkpoints.append(index)
        prev_null_percentage = null_percentage
    
    return checkpoints

def create_cohorts(df, checkpoints):
    """
    Split the dataframe into chunks based on checkpoints.
    Uses column indices instead of column names.
    """
    df_cols = df.columns.tolist()

    cohorts = []
    
    for i in range(len(checkpoints)):
        if i == 0:
            cohorts.append(df[df_cols[:checkpoints[i]]])
        else:
            cohorts.append(df[df_cols[checkpoints[i-1]:checkpoints[i]]])

    # Dropout labels
    dropout_labels = df['dropout']

    # Save cohorts to csv files
    for i, cohort in enumerate(cohorts):
        cohort.to_csv(f'cohort_data/cohort_{i+1}.csv')
        cohort['dropout'] = dropout_labels
        cohort.to_csv(f'cohort_data/cohort_{i+1}_labelled.csv')

    return cohorts

def generate_checkpoint_indicators(df, checkpoints):
    """
    Add binary columns indicating if a student made it past each checkpoint.
    A student passes a checkpoint if they have NO null values in ALL columns leading up to that checkpoint.
    
    Returns the original dataframe with added binary indicator columns.
    """
    df_cols = df.columns.tolist()
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Create binary columns for each checkpoint
    for i in range(len(checkpoints)):
        checkpoint_name = f"passed_checkpoint_{i+1}"
        
        # Check all columns up to the next checkpoint
        if i < len(checkpoints) - 1:
            cols_to_check = df_cols[:checkpoints[i + 1]]
        else:
            cols_to_check = df_cols[:]
        
        # Student passes if they have no null values in the columns leading up to this checkpoint
        result_df[checkpoint_name] = ~result_df[cols_to_check].isnull().any(axis=1)

    # Save the result dataframe to a csv file
    result_df.to_csv('cohort_data/result_df.csv')
    
    return result_df

def main():
    # Load the data
    df = pd.read_csv('UK online student engagement.csv', index_col=0)
    
    # Analyze and identify checkpoints
    checkpoints = analyze_checkpoints(df)
    print(f"Identified checkpoints: {checkpoints}")
    
    # Create cohorts (split data into chunks)
    cohorts = create_cohorts(df, checkpoints)
    print(f"Created {len(cohorts)} cohorts")
    
    # Generate binary indicator columns
    df_with_indicators = generate_checkpoint_indicators(df, checkpoints)
    
    # Display checkpoint passing rates
    checkpoint_cols = [col for col in df_with_indicators.columns if col.startswith('passed_checkpoint')]
    print("\nCheckpoint passing rates:")
    for col in checkpoint_cols:
        pass_rate = df_with_indicators[col].mean() * 100
        print(f"{col}: {pass_rate:.2f}% of students passed")

if __name__ == "__main__":
    main()
