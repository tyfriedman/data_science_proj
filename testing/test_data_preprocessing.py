import unittest
import pandas as pd
import os
import numpy as np
import sys
sys.path.append("../cohorts")
sys.path.append("../data_preprocessing")
from cohorts import (
    load_data,
    define_completion_stages,
    create_cohort_1,
    create_cohort_2,
    create_cohort_3,
    create_cohort_4,
    save_cohorts
)
from data_preprocessing import (
    create_mappings,
    apply_external_mapping,
    apply_year_mapping,
    apply_dropout_mapping,
    apply_grade_mappings,
    save_processed_data,
    process_data
)

#########################################################
# SECTION 1: Tests for cohorts.py functions
#########################################################

class TestCohortFunctions(unittest.TestCase):
    """Tests for the individual functions in cohorts.py"""
    
    def setUp(self):
        """Create a sample dataframe for testing"""
        # Create a simple test dataframe with the required columns
        self.test_df = pd.DataFrame({
            'External': [0, 1, 1, 0],
            'Year': [1, 2, 3, 1],
            'session 1': [10, 20, 30, 40],
            'session 2': [15, 25, 35, 45],
            'test 1': [3, 4, 5, 2],
            'session 3': [20, 30, np.nan, 50],
            'session 4': [25, 35, np.nan, 55],
            'test 2': [4, 3, np.nan, 5],
            'session 5': [30, np.nan, np.nan, 60],
            'test 3': [5, np.nan, np.nan, 4],
            'session 6': [35, np.nan, np.nan, 65],
            'fourm Q': [5, 8, 3, 10],
            'fourm A': [3, 5, 1, 7],
            'office hour visits': [2, 3, 1, 5],
            'dropout': [0, 1, 1, 0]
        })
    
    def test_define_completion_stages(self):
        """Test the define_completion_stages function"""
        # Apply the function
        df_with_stages = define_completion_stages(self.test_df)
        
        # Check that completion_stage column was added
        self.assertIn('completion_stage', df_with_stages.columns)
        
        # Check values are as expected
        # Student 0: Completed everything (stage 3)
        # Student 1: Dropped after test 2 (stage 1)
        # Student 2: Dropped after test 1 (stage 0)
        # Student 3: Completed everything (stage 3)
        expected_stages = [3, 1, 0, 3]
        self.assertListEqual(list(df_with_stages['completion_stage']), expected_stages)
        
        # Verify the original dataframe wasn't modified
        self.assertNotIn('completion_stage', self.test_df.columns)
    
    def test_create_cohort_1(self):
        """Test the create_cohort_1 function"""
        # First apply completion stages
        df_with_stages = define_completion_stages(self.test_df)
        
        # Create cohort 1
        cohort1 = create_cohort_1(df_with_stages)
        
        # Check that all students are included
        self.assertEqual(len(cohort1), 4)
        
        # Check that it has the expected columns
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1',
            'forum_Q_phase1', 'forum_A_phase1', 'office_hours_phase1',
            'dropped_after_test_1'
        ]
        for col in expected_columns:
            self.assertIn(col, cohort1.columns)
        
        # Check that the dropped_after_test_1 indicator is correct
        expected_dropouts = [0, 0, 1, 0]  # Student 2 dropped after test 1
        self.assertListEqual(list(cohort1['dropped_after_test_1']), expected_dropouts)
        
        # Check that forum/office hours are scaled correctly
        # Student 0 (stage 3): forum_Q * 0.33
        self.assertAlmostEqual(cohort1.loc[0, 'forum_Q_phase1'], self.test_df.loc[0, 'fourm Q'] * 0.33)
        # Student 2 (stage 0): forum_Q * the full amount (1.0)
        self.assertAlmostEqual(cohort1.loc[2, 'forum_Q_phase1'], self.test_df.loc[2, 'fourm Q'] * 1.0)
    
    def test_create_cohort_2(self):
        """Test the create_cohort_2 function"""
        # First apply completion stages
        df_with_stages = define_completion_stages(self.test_df)
        
        # Create cohort 2
        cohort2 = create_cohort_2(df_with_stages)
        
        # Check that only students who didn't drop after test 1 are included (students 0, 1, 3)
        self.assertEqual(len(cohort2), 3)
        self.assertNotIn(2, cohort2.index)  # Student 2 should be excluded
        
        # Check that it has the expected columns
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1',
            'session 3', 'session 4', 'test 2',
            'forum_Q_phase2', 'forum_A_phase2', 'office_hours_phase2',
            'dropped_after_test_2'
        ]
        for col in expected_columns:
            self.assertIn(col, cohort2.columns)
        
        # Check that the dropped_after_test_2 indicator is correct
        # Student 1 dropped after test 2
        self.assertEqual(cohort2.loc[1, 'dropped_after_test_2'], 1)
        # Students 0 and 3 didn't drop after test 2
        self.assertEqual(cohort2.loc[0, 'dropped_after_test_2'], 0)
        self.assertEqual(cohort2.loc[3, 'dropped_after_test_2'], 0)
        
        # Check that forum/office hours are scaled correctly
        # Student 0 (stage 3): forum_Q * 0.67
        self.assertAlmostEqual(cohort2.loc[0, 'forum_Q_phase2'], self.test_df.loc[0, 'fourm Q'] * 0.67)
        # Student 1 (stage 1): forum_Q * 1.0
        self.assertAlmostEqual(cohort2.loc[1, 'forum_Q_phase2'], self.test_df.loc[1, 'fourm Q'] * 1.0)
    
    def test_create_cohort_3(self):
        """Test the create_cohort_3 function"""
        # First apply completion stages
        df_with_stages = define_completion_stages(self.test_df)
        
        # Create cohort 3
        cohort3 = create_cohort_3(df_with_stages)
        
        # Check that only students who didn't drop after test 2 are included (students 0, 3)
        self.assertEqual(len(cohort3), 2)
        self.assertNotIn(1, cohort3.index)  # Student 1 should be excluded
        self.assertNotIn(2, cohort3.index)  # Student 2 should be excluded
        
        # Check that it has the expected columns
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1',
            'session 3', 'session 4', 'test 2', 'session 5', 'test 3',
            'forum_Q_phase3', 'forum_A_phase3', 'office_hours_phase3',
            'dropped_after_test_3'
        ]
        for col in expected_columns:
            self.assertIn(col, cohort3.columns)
        
        # Check that the dropped_after_test_3 indicator is correct
        # Students 0 and 3 didn't drop after test 3
        self.assertEqual(cohort3.loc[0, 'dropped_after_test_3'], 0)
        self.assertEqual(cohort3.loc[3, 'dropped_after_test_3'], 0)
        
        # Check that forum/office hours are scaled correctly
        # Student 0 (stage 3): forum_Q * 0.83
        self.assertAlmostEqual(cohort3.loc[0, 'forum_Q_phase3'], self.test_df.loc[0, 'fourm Q'] * 0.83)
    
    def test_create_cohort_4(self):
        """Test the create_cohort_4 function"""
        # First apply completion stages
        df_with_stages = define_completion_stages(self.test_df)
        
        # Create cohort 4
        cohort4 = create_cohort_4(df_with_stages)
        
        # Check that only students who completed the course are included (students 0, 3)
        self.assertEqual(len(cohort4), 2)
        self.assertNotIn(1, cohort4.index)  # Student 1 should be excluded
        self.assertNotIn(2, cohort4.index)  # Student 2 should be excluded
        
        # Check that dropout has been renamed to final_dropout
        self.assertIn('final_dropout', cohort4.columns)
        self.assertNotIn('dropout', cohort4.columns)
        
        # Check that final_dropout values are preserved from the original dropout column
        self.assertEqual(cohort4.loc[0, 'final_dropout'], self.test_df.loc[0, 'dropout'])
        self.assertEqual(cohort4.loc[3, 'final_dropout'], self.test_df.loc[3, 'dropout'])


#########################################################
# SECTION 2: Tests for data_preprocessing.py functions
#########################################################

class TestDataPreprocessing(unittest.TestCase):
    """Tests for the data_preprocessing.py file"""
    
    def setUp(self):
        """Create a sample dataframe for testing"""
        # Create a simple test dataframe with the required columns
        self.test_df = pd.DataFrame({
            'External': ['Y', 'N', 'Y', 'N'],
            'Year': ['first', 'second', 'third', 'first'],
            'test 1': ['A', 'B', 'C', 'D'],
            'test 2': ['B', 'C', 'D', 'F'],
            'test 3': ['C', 'D', 'F', 'A'],
            'ind cw': ['D', 'F', 'A', 'B'],
            'group cw': ['F', 'A', 'B', 'C'],
            'final grade': ['A', 'B', 'C', 'D'],
            'dropout': ['Y', 'N', 'Y', 'N']
        })
        
        # Get the mappings
        self.mappings = create_mappings()
    
    def test_create_mappings(self):
        """Test the create_mappings function"""
        mappings = create_mappings()
        
        # Check that all expected mapping dictionaries are present
        self.assertIn('grade_mapping', mappings)
        self.assertIn('dropout_mapping', mappings)
        self.assertIn('year_mapping', mappings)
        self.assertIn('external_mapping', mappings)
        
        # Check values in grade_mapping
        self.assertEqual(mappings['grade_mapping']['A'], 1)
        self.assertEqual(mappings['grade_mapping']['F'], 5)
        
        # Check values in dropout_mapping
        self.assertEqual(mappings['dropout_mapping']['Y'], 1)
        self.assertEqual(mappings['dropout_mapping']['N'], 0)
        
        # Check values in year_mapping
        self.assertEqual(mappings['year_mapping']['first'], 1)
        self.assertEqual(mappings['year_mapping']['third'], 3)
        
        # Check values in external_mapping
        self.assertEqual(mappings['external_mapping']['Y'], 1)
        self.assertEqual(mappings['external_mapping']['N'], 0)
    
    def test_apply_external_mapping(self):
        """Test the apply_external_mapping function"""
        # Apply the function
        df_mapped = apply_external_mapping(self.test_df, self.mappings['external_mapping'])
        
        # Check that External column values are converted correctly
        self.assertEqual(df_mapped.loc[0, 'External'], 1)  # 'Y' -> 1
        self.assertEqual(df_mapped.loc[1, 'External'], 0)  # 'N' -> 0
        self.assertEqual(df_mapped.loc[2, 'External'], 1)  # 'Y' -> 1
        self.assertEqual(df_mapped.loc[3, 'External'], 0)  # 'N' -> 0
        
        # Verify the original dataframe wasn't modified
        self.assertEqual(self.test_df.loc[0, 'External'], 'Y')
    
    def test_apply_year_mapping(self):
        """Test the apply_year_mapping function"""
        # Apply the function
        df_mapped = apply_year_mapping(self.test_df, self.mappings['year_mapping'])
        
        # Check that Year column values are converted correctly
        self.assertEqual(df_mapped.loc[0, 'Year'], 1)  # 'first' -> 1
        self.assertEqual(df_mapped.loc[1, 'Year'], 2)  # 'second' -> 2
        self.assertEqual(df_mapped.loc[2, 'Year'], 3)  # 'third' -> 3
        self.assertEqual(df_mapped.loc[3, 'Year'], 1)  # 'first' -> 1
        
        # Verify the original dataframe wasn't modified
        self.assertEqual(self.test_df.loc[0, 'Year'], 'first')
    
    def test_apply_dropout_mapping(self):
        """Test the apply_dropout_mapping function"""
        # Apply the function
        df_mapped = apply_dropout_mapping(self.test_df, self.mappings['dropout_mapping'])
        
        # Check that dropout column values are converted correctly
        self.assertEqual(df_mapped.loc[0, 'dropout'], 1)  # 'Y' -> 1
        self.assertEqual(df_mapped.loc[1, 'dropout'], 0)  # 'N' -> 0
        self.assertEqual(df_mapped.loc[2, 'dropout'], 1)  # 'Y' -> 1
        self.assertEqual(df_mapped.loc[3, 'dropout'], 0)  # 'N' -> 0
        
        # Verify the original dataframe wasn't modified
        self.assertEqual(self.test_df.loc[0, 'dropout'], 'Y')
    
    def test_apply_grade_mappings(self):
        """Test the apply_grade_mappings function"""
        # Apply the function
        df_mapped = apply_grade_mappings(self.test_df, self.mappings['grade_mapping'])
        
        # Check that test columns values are converted correctly
        # test 1: A -> 1, B -> 2, C -> 3, D -> 4
        self.assertEqual(df_mapped.loc[0, 'test 1'], 1)
        self.assertEqual(df_mapped.loc[1, 'test 1'], 2)
        self.assertEqual(df_mapped.loc[2, 'test 1'], 3)
        self.assertEqual(df_mapped.loc[3, 'test 1'], 4)
        
        # test 2: B -> 2, C -> 3, D -> 4, F -> 5
        self.assertEqual(df_mapped.loc[0, 'test 2'], 2)
        self.assertEqual(df_mapped.loc[1, 'test 2'], 3)
        self.assertEqual(df_mapped.loc[2, 'test 2'], 4)
        self.assertEqual(df_mapped.loc[3, 'test 2'], 5)
        
        # Check that all grade columns are converted
        grade_columns = ['test 1', 'test 2', 'test 3', 'ind cw', 'group cw', 'final grade']
        for col in grade_columns:
            self.assertTrue(df_mapped[col].dtype == 'int64' or 
                           df_mapped[col].dtype == 'int32' or 
                           df_mapped[col].dtype == 'float64',
                           f"Column {col} was not converted to numeric")
        
        # Verify the original dataframe wasn't modified
        self.assertEqual(self.test_df.loc[0, 'test 1'], 'A')
    

#########################################################
# SECTION 3: Tests for actual CSV file outputs
#########################################################

class TestCohortDatasets(unittest.TestCase):
    """Tests for the cohort datasets"""

    def setUp(self):
        """Load the cohort datasets before each test"""
        cohort_dir = "../cohorts"
        self.cohort1 = pd.read_csv(os.path.join(cohort_dir, "cohort1.csv"))
        self.cohort2 = pd.read_csv(os.path.join(cohort_dir, "cohort2.csv"))
        self.cohort3 = pd.read_csv(os.path.join(cohort_dir, "cohort3.csv"))
        self.cohort4 = pd.read_csv(os.path.join(cohort_dir, "cohort4.csv"))
    
    def test_no_null_values(self):
        """Test that there are no null values in any of the cohort datasets"""
        self.assertFalse(self.cohort1.isnull().any().any(), "Cohort 1 contains null values")
        self.assertFalse(self.cohort2.isnull().any().any(), "Cohort 2 contains null values")
        self.assertFalse(self.cohort3.isnull().any().any(), "Cohort 3 contains null values")
        self.assertFalse(self.cohort4.isnull().any().any(), "Cohort 4 contains null values")
    
    def test_expected_columns_cohort1(self):
        """Test that cohort 1 has the expected columns"""
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1', 
            'forum_Q_phase1', 'forum_A_phase1', 'office_hours_phase1', 
            'dropped_after_test_1'
        ]
        for col in expected_columns:
            self.assertIn(col, self.cohort1.columns, f"Column {col} missing from cohort 1")
    
    def test_expected_columns_cohort2(self):
        """Test that cohort 2 has the expected columns"""
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1',
            'session 3', 'session 4', 'test 2',
            'forum_Q_phase2', 'forum_A_phase2', 'office_hours_phase2',
            'dropped_after_test_2'
        ]
        for col in expected_columns:
            self.assertIn(col, self.cohort2.columns, f"Column {col} missing from cohort 2")
    
    def test_expected_columns_cohort3(self):
        """Test that cohort 3 has the expected columns"""
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1',
            'session 3', 'session 4', 'test 2', 'session 5', 'test 3',
            'forum_Q_phase3', 'forum_A_phase3', 'office_hours_phase3',
            'dropped_after_test_3'
        ]
        for col in expected_columns:
            self.assertIn(col, self.cohort3.columns, f"Column {col} missing from cohort 3")
    
    def test_expected_columns_cohort4(self):
        """Test that cohort 4 has the expected columns"""
        expected_columns = [
            'External', 'Year', 'session 1', 'session 2', 'test 1',
            'session 3', 'session 4', 'test 2', 'session 5', 'test 3',
            'session 6', 'ind cw', 'group cw', 'final grade',
            'fourm Q', 'fourm A', 'office hour visits', 
            'final_dropout', 'completion_stage'
        ]
        for col in expected_columns:
            self.assertIn(col, self.cohort4.columns, f"Column {col} missing from cohort 4")
    
    def test_binary_dropout_indicators(self):
        """Test that dropout indicator columns contain only binary values (0 or 1)"""
        self.assertTrue(self.cohort1['dropped_after_test_1'].isin([0, 1]).all(), 
                        "Cohort 1 dropout indicator contains non-binary values")
        
        self.assertTrue(self.cohort2['dropped_after_test_2'].isin([0, 1]).all(), 
                        "Cohort 2 dropout indicator contains non-binary values")
        
        self.assertTrue(self.cohort3['dropped_after_test_3'].isin([0, 1]).all(), 
                        "Cohort 3 dropout indicator contains non-binary values")
        
        self.assertTrue(self.cohort4['final_dropout'].isin([0, 1]).all(), 
                        "Cohort 4 dropout indicator contains non-binary values")
    
    def test_cohort_relationships(self):
        """Test expected relationships between cohorts based on dropout pattern"""
        # Cohort 2 should contain fewer students than cohort 1 (due to dropouts)
        self.assertLess(len(self.cohort2), len(self.cohort1), 
                        "Cohort 2 should have fewer students than Cohort 1")
        
        # Cohort 3 should contain fewer students than cohort 2
        self.assertLess(len(self.cohort3), len(self.cohort2), 
                        "Cohort 3 should have fewer students than Cohort 2")
        
        # Cohort 4 should only contain students with completion_stage=3
        self.assertTrue((self.cohort4['completion_stage'] == 3).all(),
                       "All students in Cohort 4 should have completion_stage=3")
    
    def test_valid_data_ranges(self):
        """Test that important numeric fields are within expected ranges"""
        # Test scores should be between 1-5
        for cohort in [self.cohort1, self.cohort2, self.cohort3, self.cohort4]:
            self.assertTrue((cohort['test 1'] >= 1).all() and (cohort['test 1'] <= 5).all(), 
                           "Test 1 scores outside valid range (1-5)")
            
        for cohort in [self.cohort2, self.cohort3, self.cohort4]:
            self.assertTrue((cohort['test 2'] >= 1).all() and (cohort['test 2'] <= 5).all(), 
                           "Test 2 scores outside valid range (1-5)")
            
        for cohort in [self.cohort3, self.cohort4]:
            self.assertTrue((cohort['test 3'] >= 1).all() and (cohort['test 3'] <= 5).all(), 
                           "Test 3 scores outside valid range (1-5)")
        
        # External should be binary (0 or 1)
        for cohort in [self.cohort1, self.cohort2, self.cohort3, self.cohort4]:
            self.assertTrue(cohort['External'].isin([0, 1]).all(), 
                           "External column contains non-binary values")
            
        # Year should be 1, 2, or 3
        for cohort in [self.cohort1, self.cohort2, self.cohort3, self.cohort4]:
            self.assertTrue(cohort['Year'].isin([1, 2, 3]).all(), 
                           "Year column contains invalid values (should be 1, 2, or 3)")
    
    def test_forum_office_hours_scaling(self):
        """Test that forum activity and office hours are scaled appropriately"""
        # For cohort 1, all forum_Q values should be non-negative
        self.assertTrue((self.cohort1['forum_Q_phase1'] >= 0).all(), 
                        "Negative forum_Q values found in Cohort 1")
        
        # For cohort 2, all forum_Q values should be non-negative
        self.assertTrue((self.cohort2['forum_Q_phase2'] >= 0).all(), 
                        "Negative forum_Q values found in Cohort 2")
        
        # For cohort 3, all forum_Q values should be non-negative
        self.assertTrue((self.cohort3['forum_Q_phase3'] >= 0).all(), 
                        "Negative forum_Q values found in Cohort 3")

if __name__ == '__main__':
    unittest.main()
