import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the data
    df = pd.read_csv('processed_student_engagement.csv', index_col=0)
    
    # Define completion stage for each student
    # 0: Dropped after test 1 (1/3 completion)
    # 1: Dropped after test 2 (2/3 completion)
    # 2: Dropped after test 3 (5/6 completion)
    # 3: Completed the course (full completion)
    df['completion_stage'] = 3  # default: completed everything
    df.loc[df['session 6'].isna() & ~df['session 5'].isna(), 'completion_stage'] = 2  # dropped after test 3
    df.loc[df['session 5'].isna() & ~df['session 3'].isna(), 'completion_stage'] = 1  # dropped after test 2
    df.loc[df['session 3'].isna(), 'completion_stage'] = 0  # dropped after test 1
    
    # Create Cohort 1: Everyone, only columns up to test 1
    # Add label indicating if they dropped after test 1 (session 3 is null)
    cohort_1_cols = ['External', 'Year', 'session 1', 'session 2', 'test 1', 'fourm Q', 'fourm A', 'office hour visits', 'completion_stage']
    cohort_1 = df[cohort_1_cols].copy()
    # Add scaled forum and office hours engagement
    cohort_1.loc[cohort_1['completion_stage'] == 3, 'forum_Q_phase1'] = cohort_1['fourm Q'] * 0.33
    cohort_1.loc[cohort_1['completion_stage'] == 2, 'forum_Q_phase1'] = cohort_1['fourm Q'] * 0.4
    cohort_1.loc[cohort_1['completion_stage'] == 1, 'forum_Q_phase1'] = cohort_1['fourm Q'] * 0.5
    cohort_1.loc[cohort_1['completion_stage'] == 0, 'forum_Q_phase1'] = cohort_1['fourm Q'] * 1
    cohort_1.loc[cohort_1['completion_stage'] == 3, 'forum_A_phase1'] = cohort_1['fourm A'] * 0.33
    cohort_1.loc[cohort_1['completion_stage'] == 2, 'forum_A_phase1'] = cohort_1['fourm A'] * 0.4
    cohort_1.loc[cohort_1['completion_stage'] == 1, 'forum_A_phase1'] = cohort_1['fourm A'] * 0.5
    cohort_1.loc[cohort_1['completion_stage'] == 0, 'forum_A_phase1'] = cohort_1['fourm A'] * 1
    cohort_1.loc[cohort_1['completion_stage'] == 3, 'office_hours_phase1'] = cohort_1['office hour visits'] * 0.33
    cohort_1.loc[cohort_1['completion_stage'] == 2, 'office_hours_phase1'] = cohort_1['office hour visits'] * 0.4
    cohort_1.loc[cohort_1['completion_stage'] == 1, 'office_hours_phase1'] = cohort_1['office hour visits'] * 0.5
    cohort_1.loc[cohort_1['completion_stage'] == 0, 'office_hours_phase1'] = cohort_1['office hour visits'] * 1
    cohort_1.drop(columns=['fourm Q', 'fourm A', 'office hour visits', 'completion_stage'], inplace=True)
    cohort_1['dropped_after_test_1'] = df['session 3'].isna().astype(int)
    
    # Create Cohort 2: Everyone who didn't drop after test 1
    # Only include those with data up to test 2
    # Label indicates if they dropped after test 2 (session 5 is null)
    cohort_2_cols = ['External', 'Year', 'session 1', 'session 2', 'test 1', 
                     'session 3', 'session 4', 'test 2', 'fourm Q', 'fourm A', 'office hour visits', 'completion_stage']
    cohort_2 = df[~df['session 3'].isna()][cohort_2_cols].copy()
    cohort_2.loc[cohort_2['completion_stage'] == 3, 'forum_Q_phase2'] = cohort_2['fourm Q'] * 0.67
    cohort_2.loc[cohort_2['completion_stage'] == 2, 'forum_Q_phase2'] = cohort_2['fourm Q'] * 0.8
    cohort_2.loc[cohort_2['completion_stage'] == 1, 'forum_Q_phase2'] = cohort_2['fourm Q'] * 1
    cohort_2.loc[cohort_2['completion_stage'] == 3, 'forum_A_phase2'] = cohort_2['fourm A'] * 0.67
    cohort_2.loc[cohort_2['completion_stage'] == 2, 'forum_A_phase2'] = cohort_2['fourm A'] * 0.8
    cohort_2.loc[cohort_2['completion_stage'] == 1, 'forum_A_phase2'] = cohort_2['fourm A'] * 1
    cohort_2.loc[cohort_2['completion_stage'] == 3, 'office_hours_phase2'] = cohort_2['office hour visits'] * 0.67
    cohort_2.loc[cohort_2['completion_stage'] == 2, 'office_hours_phase2'] = cohort_2['office hour visits'] * 0.8
    cohort_2.loc[cohort_2['completion_stage'] == 1, 'office_hours_phase2'] = cohort_2['office hour visits'] * 1
    cohort_2.drop(columns=['fourm Q', 'fourm A', 'office hour visits', 'completion_stage'], inplace=True)
    cohort_2['dropped_after_test_2'] = df[~df['session 3'].isna()]['session 5'].isna().astype(int)
    
    # Create Cohort 3: Everyone who didn't drop after test 2
    # Only include those with data up to test 3
    # Label indicates if they dropped after test 3 (session 6 is null)
    cohort_3_cols = ['External', 'Year', 'session 1', 'session 2', 'test 1', 
                     'session 3', 'session 4', 'test 2',
                     'session 5', 'test 3', 'fourm Q', 'fourm A', 'office hour visits', 'completion_stage']
    cohort_3 = df[~df['session 5'].isna()][cohort_3_cols].copy()
    cohort_3.loc[cohort_3['completion_stage'] == 3, 'forum_Q_phase3'] = cohort_3['fourm Q'] * 0.83
    cohort_3.loc[cohort_3['completion_stage'] == 2, 'forum_Q_phase3'] = cohort_3['fourm Q'] * 1
    cohort_3.loc[cohort_3['completion_stage'] == 3, 'forum_A_phase3'] = cohort_3['fourm A'] * 0.83
    cohort_3.loc[cohort_3['completion_stage'] == 2, 'forum_A_phase3'] = cohort_3['fourm A'] * 1
    cohort_3.loc[cohort_3['completion_stage'] == 3, 'office_hours_phase3'] = cohort_3['office hour visits'] * 0.83
    cohort_3.loc[cohort_3['completion_stage'] == 2, 'office_hours_phase3'] = cohort_3['office hour visits'] * 1
    cohort_3.drop(columns=['fourm Q', 'fourm A', 'office hour visits', 'completion_stage'], inplace=True)
    cohort_3['dropped_after_test_3'] = df[~df['session 5'].isna()]['session 6'].isna().astype(int)
    
    # Create Cohort 4: Everyone who completed the course
    # Include all who have session 6 data (no early dropouts)
    # Label is the same as the original dropout column
    cohort_4 = df[~df['session 6'].isna()].copy()
    cohort_4.rename(columns={'dropout': 'final_dropout'}, inplace=True)
    
    # Save cohorts to csv files
    cohort_1.to_csv('./cohorts/cohort1.csv')
    cohort_2.to_csv('./cohorts/cohort2.csv')
    cohort_3.to_csv('./cohorts/cohort3.csv')
    cohort_4.to_csv('./cohorts/cohort4.csv')
    
    print(f"Cohort 1 size: {len(cohort_1)}")
    print(f"Cohort 2 size: {len(cohort_2)}")
    print(f"Cohort 3 size: {len(cohort_3)}")
    print(f"Cohort 4 size: {len(cohort_4)}")

if __name__ == "__main__":
    main()