import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path='../processed_student_engagement.csv'):
    """
    Load the student engagement data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the processed student engagement CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded student engagement data
    """
    return pd.read_csv(file_path, index_col=0)

def define_completion_stages(df):
    """
    Define completion stage for each student
    0: Dropped after test 1 (1/3 completion)
    1: Dropped after test 2 (2/3 completion)
    2: Dropped after test 3 (5/6 completion)
    3: Completed the course (full completion)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data
        
    Returns:
    --------
    pd.DataFrame
        Data with completion_stage column added
    """
    df = df.copy()
    df['completion_stage'] = 3  # default: completed everything
    df.loc[df['session 6'].isna() & ~df['session 5'].isna(), 'completion_stage'] = 2  # dropped after test 3
    df.loc[df['session 5'].isna() & ~df['session 3'].isna(), 'completion_stage'] = 1  # dropped after test 2
    df.loc[df['session 3'].isna(), 'completion_stage'] = 0  # dropped after test 1
    return df

def create_cohort_1(df):
    """
    Create Cohort 1: Everyone, only columns up to test 1
    Add label indicating if they dropped after test 1 (session 3 is null)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data with completion_stage column
        
    Returns:
    --------
    pd.DataFrame
        Cohort 1 dataset
    """
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
    
    return cohort_1

def create_cohort_2(df):
    """
    Create Cohort 2: Everyone who didn't drop after test 1
    Only include those with data up to test 2
    Label indicates if they dropped after test 2 (session 5 is null)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data with completion_stage column
        
    Returns:
    --------
    pd.DataFrame
        Cohort 2 dataset
    """
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
    
    return cohort_2

def create_cohort_3(df):
    """
    Create Cohort 3: Everyone who didn't drop after test 2
    Only include those with data up to test 3
    Label indicates if they dropped after test 3 (session 6 is null)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data with completion_stage column
        
    Returns:
    --------
    pd.DataFrame
        Cohort 3 dataset
    """
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
    
    return cohort_3

def create_cohort_4(df):
    """
    Create Cohort 4: Everyone who completed the course
    Include all who have session 6 data (no early dropouts)
    Label is the same as the original dropout column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data with completion_stage column
        
    Returns:
    --------
    pd.DataFrame
        Cohort 4 dataset
    """
    cohort_4 = df[~df['session 6'].isna()].copy()
    cohort_4.rename(columns={'dropout': 'final_dropout'}, inplace=True)
    return cohort_4

def save_cohorts(cohort_1, cohort_2, cohort_3, cohort_4, output_dir='./'):
    """
    Save cohorts to CSV files
    
    Parameters:
    -----------
    cohort_1, cohort_2, cohort_3, cohort_4 : pd.DataFrame
        The cohort dataframes
    output_dir : str
        Directory to save the CSV files
    """
    cohort_1.to_csv(os.path.join(output_dir, 'cohort1.csv'))
    cohort_2.to_csv(os.path.join(output_dir, 'cohort2.csv'))
    cohort_3.to_csv(os.path.join(output_dir, 'cohort3.csv'))
    cohort_4.to_csv(os.path.join(output_dir, 'cohort4.csv'))
    
    print(f"Cohort 1 size: {len(cohort_1)}")
    print(f"Cohort 2 size: {len(cohort_2)}")
    print(f"Cohort 3 size: {len(cohort_3)}")
    print(f"Cohort 4 size: {len(cohort_4)}")

def main():
    # Load the data
    df = load_data()
    
    # Define completion stages
    df = define_completion_stages(df)
    
    # Create cohorts
    cohort_1 = create_cohort_1(df)
    cohort_2 = create_cohort_2(df)
    cohort_3 = create_cohort_3(df)
    cohort_4 = create_cohort_4(df)
    
    # Save cohorts to CSV files
    save_cohorts(cohort_1, cohort_2, cohort_3, cohort_4)

if __name__ == "__main__":
    main()