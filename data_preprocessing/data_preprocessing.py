import pandas as pd
import os

def load_data(file_path='../UK online student engagement.csv'):
    """
    Load the student engagement data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the raw student engagement CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded student engagement data
    """
    return pd.read_csv(file_path, index_col=0)

def create_mappings():
    """
    Create mapping dictionaries for various columns
    
    Returns:
    --------
    dict
        Dictionary containing all mapping dictionaries
    """
    # Create mapping dictionaries
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5}
    dropout_mapping = {'Y': 1, 'N': 0}
    year_mapping = {'first': 1, 'second': 2, 'third': 3}
    external_mapping = {'Y': 1, 'N': 0}
    
    return {
        'grade_mapping': grade_mapping,
        'dropout_mapping': dropout_mapping,
        'year_mapping': year_mapping,
        'external_mapping': external_mapping
    }

def apply_external_mapping(df, external_mapping):
    """
    Convert External column from Y/N to 1/0
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data
    external_mapping : dict
        Mapping dictionary for External column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with External column converted
    """
    df = df.copy()
    df['External'] = df['External'].map(external_mapping)
    return df

def apply_year_mapping(df, year_mapping):
    """
    Convert Year column from text to numeric
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data
    year_mapping : dict
        Mapping dictionary for Year column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Year column converted
    """
    df = df.copy()
    df['Year'] = df['Year'].map(year_mapping)
    return df

def apply_dropout_mapping(df, dropout_mapping):
    """
    Convert dropout column from Y/N to 1/0
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data
    dropout_mapping : dict
        Mapping dictionary for dropout column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with dropout column converted
    """
    df = df.copy()
    df['dropout'] = df['dropout'].map(dropout_mapping)
    return df

def apply_grade_mappings(df, grade_mapping):
    """
    Convert grade columns from letter grades to numeric values
    
    Parameters:
    -----------
    df : pd.DataFrame
        Student engagement data
    grade_mapping : dict
        Mapping dictionary for grade columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with grade columns converted
    """
    df = df.copy()
    # Convert grade columns - find all columns that might contain letter grades
    grade_columns = ['test 1', 'test 2', 'test 3', 'ind cw', 'group cw', 'final grade']
    for col in grade_columns:
        if col in df.columns:
            df[col] = df[col].map(grade_mapping)
    return df

def save_processed_data(df, output_path='../processed_student_engagement.csv'):
    """
    Save the processed data to a CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed student engagement data
    output_path : str
        Path to save the processed data
        
    Returns:
    --------
    str
        Path to the saved file
    """
    df.to_csv(output_path)
    return output_path

def process_data(input_path='../UK online student engagement.csv', 
                output_path='../processed_student_engagement.csv'):
    """
    Process the student engagement data
    
    Parameters:
    -----------
    input_path : str
        Path to the raw student engagement CSV file
    output_path : str
        Path to save the processed data
        
    Returns:
    --------
    pd.DataFrame
        Processed student engagement data
    """
    # Load the data
    df = load_data(input_path)
    
    # Create mappings
    mappings = create_mappings()
    
    # Apply mappings
    df = apply_external_mapping(df, mappings['external_mapping'])
    df = apply_year_mapping(df, mappings['year_mapping'])
    df = apply_dropout_mapping(df, mappings['dropout_mapping'])
    df = apply_grade_mappings(df, mappings['grade_mapping'])
    
    # Save the processed data
    save_processed_data(df, output_path)
    
    return df

def main():
    # Process the data
    process_data()
    print("Data processing complete. Output saved to 'processed_student_engagement.csv'")

if __name__ == "__main__":
    main()
