import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the data
    df = pd.read_csv('processed_student_engagement.csv', index_col=0)
    
    # Create Cohort 1: Everyone, only columns up to test 1
    # Add label indicating if they dropped after test 1 (session 3 is null)
    cohort_1_cols = ['External', 'Year', 'session 1', 'session 2', 'test 1']
    cohort_1 = df[cohort_1_cols].copy()
    cohort_1['dropped_after_test_1'] = df['session 3'].isna().astype(int)
    
    # Create Cohort 2: Everyone who didn't drop after test 1
    # Only include those with data up to test 2
    # Label indicates if they dropped after test 2 (session 5 is null)
    cohort_2_cols = ['External', 'Year', 'session 1', 'session 2', 'test 1', 
                     'session 3', 'session 4', 'test 2']
    cohort_2 = df[~df['session 3'].isna()][cohort_2_cols].copy()
    cohort_2['dropped_after_test_2'] = df[~df['session 3'].isna()]['session 5'].isna().astype(int)
    
    # Create Cohort 3: Everyone who didn't drop after test 2
    # Only include those with data up to test 3
    # Label indicates if they dropped after test 3 (session 6 is null)
    cohort_3_cols = ['External', 'Year', 'session 1', 'session 2', 'test 1', 
                     'session 3', 'session 4', 'test 2',
                     'session 5', 'test 3']
    cohort_3 = df[~df['session 5'].isna()][cohort_3_cols].copy()
    cohort_3['dropped_after_test_3'] = df[~df['session 5'].isna()]['session 6'].isna().astype(int)
    
    # Create Cohort 4: Everyone who completed the course
    # Include all who have session 6 data (no early dropouts)
    # Label is the same as the original dropout column
    cohort_4 = df[~df['session 6'].isna()].copy()
    cohort_4.rename(columns={'dropout': 'final_dropout'}, inplace=True)
    
    # Save cohorts to csv files
    cohort_1.to_csv('./cohorts/cohort_1.csv')
    cohort_2.to_csv('./cohorts/cohort_2.csv')
    cohort_3.to_csv('./cohorts/cohort_3.csv')
    cohort_4.to_csv('./cohorts/cohort_4.csv')
    
    print(f"Cohort 1 size: {len(cohort_1)}")
    print(f"Cohort 2 size: {len(cohort_2)}")
    print(f"Cohort 3 size: {len(cohort_3)}")
    print(f"Cohort 4 size: {len(cohort_4)}")

if __name__ == "__main__":
    main()